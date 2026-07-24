"""Fine-tune one configured KG seed from the shared pretrain checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

from lipid_screening_agent import __version__
from lipid_screening_agent.artifacts import NodeStatus
from lipid_screening_agent.cli import (
    CommonRunnerArguments,
    add_common_runner_arguments,
    load_common_runner_environment,
)
from lipid_screening_agent.config.models import KGFinetuneConfig
from lipid_screening_agent.runtime import InputError, RunContext, atomic_write_json, sha256_file
from lipid_screening_agent.runtime.execution import NodeExecution, execute_node

from ..input_prepare._common import add_execution_identity_arguments, execution_identity
from ._training_common import (
    atomic_dataframe_csv,
    atomic_torch_save,
    check_unique_seeds,
    environment_snapshot,
    load_training_dependencies,
    output_path,
    resolve_committed_artifact,
    resolve_device,
    resolve_run_file,
    seed_configuration_hash,
    set_random_seed,
)
from .pretrain import CHECKPOINT_FORMAT

NODE_ID = "kg_finetune_seed"


def _binary_metrics(pos, neg, torch, roc_auc_score, average_precision_score):
    positive = torch.sigmoid(pos.detach()).reshape(-1).cpu().numpy()
    negative = torch.sigmoid(neg.detach()).reshape(-1).cpu().numpy()
    labels = [1] * len(positive) + [0] * len(negative)
    scores = list(positive) + list(negative)
    if not positive.size or not negative.size:
        raise InputError("fine-tune metrics require positive and negative examples")
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "num_pos": int(len(positive)),
        "num_neg": int(len(negative)),
    }


def _evaluate(model, message_graph, positive_graph, negative_graph, etypes, name, deps):
    torch = deps["torch"]
    from sklearn.metrics import average_precision_score, roc_auc_score

    model.eval()
    with torch.no_grad():
        pred_pos, pred_neg, _, _ = model(
            message_graph,
            negative_graph,
            positive_graph,
            pretrain_mode=False,
            mode=name,
        )
    pos = torch.cat([pred_pos[etype].reshape(-1) for etype in etypes])
    neg = torch.cat([pred_neg[etype].reshape(-1) for etype in etypes])
    labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
    logits = torch.cat([pos, neg])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).item()
    all_values = _binary_metrics(pos, neg, torch, roc_auc_score, average_precision_score)
    per_relation = [
        _binary_metrics(
            pred_pos[etype],
            pred_neg[etype],
            torch,
            roc_auc_score,
            average_precision_score,
        )
        for etype in etypes
    ]
    return {
        "loss": float(loss),
        "micro_auroc": all_values["auroc"],
        "micro_auprc": all_values["auprc"],
        "macro_auroc": float(sum(item["auroc"] for item in per_relation) / len(per_relation)),
        "macro_auprc": float(sum(item["auprc"] for item in per_relation) / len(per_relation)),
        "forward_auroc": per_relation[0]["auroc"],
        "forward_auprc": per_relation[0]["auprc"],
        "reverse_auroc": per_relation[1]["auroc"],
        "reverse_auprc": per_relation[1]["auprc"],
        "num_pos": all_values["num_pos"],
        "num_neg": all_values["num_neg"],
    }


def _model_from_config(model_class, graph, constructor, device):
    return model_class(
        graph,
        in_size=constructor["n_inp"],
        hidden_size=constructor["n_hid"],
        out_size=constructor["n_out"],
        attention=constructor["attention"],
        proto=constructor["proto"],
        proto_num=constructor.get("proto_num", 3),
        sim_measure=constructor.get("sim_measure", "all_nodes_profile"),
        bert_measure=constructor.get("bert_measure", "disease_name"),
        agg_measure=constructor.get("agg_measure", "rarity"),
        num_walks=constructor.get("num_walks", 200),
        walk_mode=constructor.get("walk_mode", "bit"),
        path_length=constructor.get("path_length", 2),
        split="custom_finetune_seed",
        data_folder="",
        exp_lambda=constructor.get("exp_lambda", 0.7),
        device=device,
    ).to(device)


def _compound_ids(source_ids: object, tag: str):
    values = []
    for item in str(source_ids or "").split(";"):
        item = item.strip()
        if item.startswith(tag):
            value = item[len(tag) :].strip()
            if value and value not in values:
                values.append(value)
    return values


def _rank(model, graph, candidates, disease_index, forward_etype, positive_ids, deps):
    torch, numpy = deps["torch"], deps["numpy"]
    model.eval()
    with torch.no_grad():
        embeddings = model(graph, graph, return_h=True)
        indices = torch.tensor(
            candidates["type_local_index"].astype(int).tolist(),
            device=model.w_rels.device,
        )
        drug = embeddings["drug"][indices]
        disease = embeddings["disease"][disease_index].unsqueeze(0)
        relation = model.w_rels[model.pred.rel2idx[forward_etype]]
        scores = torch.sigmoid(torch.sum(drug * relation * disease, dim=1)).cpu().numpy()
    ranking = candidates.copy()
    ranking["score"] = scores
    ranking["is_custom_disease_train_positive"] = ranking.node_id.isin(positive_ids)
    ranking = ranking.sort_values(["score", "node_id"], ascending=[False, True]).reset_index(
        drop=True
    )
    ranking.insert(0, "rank", numpy.arange(1, len(ranking) + 1))
    return ranking


def _cache_matches(paths, cache_key):
    required = ["ranking", "history", "summary", "seed_manifest"]
    if not all(paths[name].is_file() for name in required):
        return False
    try:
        payload = json.loads(paths["seed_manifest"].read_text(encoding="utf-8"))
        if payload.get("cache_key") != cache_key:
            return False
        return all(
            payload.get("file_sha256", {}).get(name) == sha256_file(paths[name])
            for name in ("ranking", "history", "summary")
        )
    except Exception:
        return False


def run_finetune(
    *,
    seed,
    nodes,
    directed,
    training_manifest,
    pretrain_config,
    checkpoint_path,
    settings,
    paths,
    device,
    deps,
    environment,
    cache_key,
    allowed_root,
):
    numpy, torch, dgl = deps["numpy"], deps["torch"], deps["dgl"]
    from .model import (
        Full_Graph_NegSampler,
        HeteroRGCN,
        create_dgl_graph,
        evaluate_graph_construct,
        initialize_node_embedding,
    )

    set_random_seed(seed, numpy=numpy, torch=torch, dgl=dgl)
    nodes["node_id"] = nodes.node_id.astype(str)
    nodes["type_local_index"] = nodes.type_local_index.astype(int)
    directed["x_id"] = directed.x_id.astype(str)
    directed["y_id"] = directed.y_id.astype(str)
    directed[["x_idx", "y_idx"]] = directed[["x_idx", "y_idx"]].astype(int)
    forward = tuple(settings.relations.forward)
    reverse = tuple(settings.relations.reverse)
    if len(forward) != 3 or len(reverse) != 3:
        raise InputError("fine-tune forward/reverse relations must be canonical triples")
    target_relations = {forward[1], reverse[1]}
    train_all = directed[directed.split == "train"].copy()
    splits = {
        name: directed[(directed.split == name) & directed.relation.isin(target_relations)]
        .drop_duplicates(["x_id", "relation", "y_id"])
        .reset_index(drop=True)
        for name in ("train", "valid", "test")
    }
    if any(frame.empty for frame in splits.values()):
        raise InputError(
            "fine-tuning requires drug_disease edges in every split",
            details={name: len(frame) for name, frame in splits.items()},
        )
    custom_id = str(training_manifest["custom_disease_id"])
    custom_train = splits["train"][
        (splits["train"].relation == forward[1]) & (splits["train"].y_id == custom_id)
    ]
    leaked = directed[
        (directed.split != "train")
        & (directed.relation == forward[1])
        & (directed.y_id == custom_id)
    ]
    if not custom_train.empty and not leaked.empty:
        raise InputError("custom disease positive edges leaked outside the train split")
    columns = ["x_type", "relation", "y_type", "x_idx", "y_idx"]
    message = train_all.drop_duplicates(["x_id", "relation", "y_id"]).reset_index(drop=True)
    graph = create_dgl_graph(message[columns], directed[columns])
    graph = initialize_node_embedding(graph, pretrain_config["model_constructor"]["n_inp"])
    expected = [tuple(value) for value in pretrain_config["canonical_etypes"]]
    if list(graph.canonical_etypes) != expected:
        raise InputError(
            "canonical etypes changed relative to pretraining",
            details={"expected": expected, "observed": list(graph.canonical_etypes)},
        )
    dd_graph = create_dgl_graph(splits["train"][columns], directed[columns])
    dd_etypes = [etype for etype in dd_graph.canonical_etypes if etype in {forward, reverse}]
    if set(dd_etypes) != {forward, reverse}:
        raise InputError("fine-tune graph is missing a configured drug_disease direction")
    valid_pos, valid_neg = evaluate_graph_construct(
        splits["valid"][["x_idx", "relation", "y_idx"]],
        graph,
        settings.negative_method,
        settings.negative_samples_per_positive,
        device,
    )
    test_pos, test_neg = evaluate_graph_construct(
        splits["test"][["x_idx", "relation", "y_idx"]],
        graph,
        settings.negative_method,
        settings.negative_samples_per_positive,
        device,
    )
    graph = graph.to(device)
    valid_pos, valid_neg = valid_pos.to(device), valid_neg.to(device)
    test_pos, test_neg = test_pos.to(device), test_neg.to(device)
    model = _model_from_config(HeteroRGCN, graph, pretrain_config["model_constructor"], device)
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # older Torch
        state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.pred.etypes_dd = dd_etypes
    model.pred.node_types_dd = ["disease", "drug"]
    if settings.reset_decoder:
        torch.nn.init.xavier_uniform_(model.w_rels)
    if (
        settings.optimizer.casefold() != "adamw"
        or settings.loss != "binary_cross_entropy_with_logits"
    ):
        raise InputError("fine-tune checkpoint contract requires AdamW and BCE-with-logits")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=settings.scheduler_factor,
        patience=settings.scheduler_patience,
    )
    sampler = Full_Graph_NegSampler(
        graph, settings.negative_samples_per_positive, settings.negative_method, device
    )
    best_metric, best_epoch, best_state, best_valid = float("-inf"), 0, None, None
    patience = 0
    history = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, settings.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        negative = sampler(graph)
        pred_pos, pred_neg, _, _ = model(graph, negative, pretrain_mode=False, mode="train")
        pos = torch.cat([pred_pos[etype].reshape(-1) for etype in dd_etypes])
        neg = torch.cat([pred_neg[etype].reshape(-1) for etype in dd_etypes])
        labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
        logits = torch.cat([pos, neg])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        if settings.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
        optimizer.step()
        scheduler.step(loss.item())
        record = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(loss.item()),
        }
        validate = (
            settings.validate_first_epoch and epoch == 1
        ) or epoch % settings.validate_every_epochs == 0
        if validate:
            valid = _evaluate(model, graph, valid_pos, valid_neg, dd_etypes, "valid", deps)
            record["valid"] = valid
            current = valid.get(settings.best_model_metric)
            if current is None:
                raise InputError("configured fine-tune selection metric is unavailable")
            if current > best_metric:
                best_metric, best_epoch, best_valid = float(current), epoch, copy.deepcopy(valid)
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                patience = 0
            else:
                patience += 1
        record["elapsed_seconds"] = float(time.perf_counter() - epoch_started)
        if device.type == "cuda":
            record["gpu_max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        history.append(record)
        atomic_write_json(paths["history"], history, allowed_root=allowed_root)
        if (
            settings.early_stopping_patience is not None
            and patience >= settings.early_stopping_patience
        ):
            break
    if best_state is None:
        best_state = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
        best_epoch = len(history)
    model.load_state_dict(best_state, strict=True)
    test = _evaluate(model, graph, test_pos, test_neg, dd_etypes, "test", deps)
    disease_rows = nodes[nodes.node_id == custom_id]
    if len(disease_rows) != 1:
        raise InputError("custom disease node is missing or duplicated")
    tag = str(training_manifest.get("candidate_source_tag", "UserLibrary:"))
    candidates = nodes[nodes.node_type == "drug"].copy()
    candidates["compound_ids"] = candidates.source_ids.map(lambda value: _compound_ids(value, tag))
    candidates = (
        candidates[candidates.compound_ids.map(bool)]
        .sort_values("type_local_index")
        .reset_index(drop=True)
    )
    if candidates.empty:
        raise InputError("no candidate drug nodes match candidate_source_tag", details={"tag": tag})
    ranking = _rank(
        model,
        graph,
        candidates,
        int(disease_rows.iloc[0].type_local_index),
        forward,
        set(custom_train.x_id),
        deps,
    )
    ranking["compound_ids"] = ranking.compound_ids.map(
        lambda values: json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    )
    keep = [
        "node_id",
        "node_name",
        "source_ids",
        "compound_ids",
        "rank",
        "score",
        "is_custom_disease_train_positive",
    ]
    for column in keep:
        if column not in ranking:
            ranking[column] = ""
    atomic_dataframe_csv(ranking[keep], paths["ranking"], allowed_root=allowed_root, index=False)
    seconds = float(time.perf_counter() - started)
    positive_ranks = (
        ranking.loc[ranking.is_custom_disease_train_positive, "rank"].astype(int).to_numpy()
    )
    summary = {
        "status": "succeeded",
        "seed": seed,
        "seed_configuration_hash": paths["seed_configuration_hash"],
        "selection_metric": settings.best_model_metric,
        "best_epoch": int(best_epoch),
        "best_valid_metric": None if not math.isfinite(best_metric) else best_metric,
        "best_valid_metrics": best_valid,
        "test": test,
        "training_seconds": seconds,
        "epoch_seconds": [item["elapsed_seconds"] for item in history],
        "environment": environment,
        "custom_disease_train_positive_rank": {
            "candidate_source_tag": tag,
            "num_ranked_candidates": int(len(ranking)),
            "num_ranked_train_positives": int(len(positive_ranks)),
            "mean_positive_rank": None
            if not len(positive_ranks)
            else float(numpy.mean(positive_ranks)),
            "median_positive_rank": None
            if not len(positive_ranks)
            else float(numpy.median(positive_ranks)),
            "best_positive_rank": None
            if not len(positive_ranks)
            else int(numpy.min(positive_ranks)),
            "worst_positive_rank": None
            if not len(positive_ranks)
            else int(numpy.max(positive_ranks)),
        },
    }
    atomic_write_json(paths["summary"], summary, allowed_root=allowed_root)
    if settings.save_per_seed_models:
        atomic_torch_save(torch, best_state, paths["model"], allowed_root=allowed_root)
    if settings.save_per_seed_embeddings:
        model.eval()
        with torch.no_grad():
            embeddings = model(graph, graph, return_h=True)
        atomic_torch_save(
            torch,
            {key: value.detach().cpu() for key, value in embeddings.items()},
            paths["embeddings"],
            allowed_root=allowed_root,
        )
    manifest = {
        "schema_version": "1.0",
        "seed": seed,
        "seed_configuration_hash": paths["seed_configuration_hash"],
        "cache_key": cache_key,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "file_sha256": {
            name: sha256_file(paths[name]) for name in ("ranking", "history", "summary")
        },
    }
    atomic_write_json(paths["seed_manifest"], manifest, allowed_root=allowed_root)
    return summary


def _operation(
    execution: NodeExecution,
    *,
    context,
    settings,
    seed,
    nodes_path,
    edges_path,
    training_manifest_path,
    checkpoint_path,
    pretrain_config_path,
    nodes_artifact_manifest,
    edges_artifact_manifest,
    training_artifact_manifest,
    checkpoint_artifact_manifest,
    pretrain_config_artifact_manifest,
    allow_cpu_small_graph,
    code_version,
):
    configured = check_unique_seeds(settings.seeds)
    if seed not in configured:
        raise InputError(
            "--seed is not present in kg.finetune.seeds",
            details={"seed": seed, "configured_seeds": list(configured)},
        )
    upstream = []
    specs = [
        ("nodes_path", nodes_artifact_manifest, "kg_training_nodes", "kg_prepare_training_data"),
        ("edges_path", edges_artifact_manifest, "kg_training_edges", "kg_prepare_training_data"),
        (
            "training_manifest_path",
            training_artifact_manifest,
            "kg_training_manifest",
            "kg_prepare_training_data",
        ),
        ("checkpoint_path", checkpoint_artifact_manifest, "kg_pretrain_checkpoint", "kg_pretrain"),
        (
            "pretrain_config_path",
            pretrain_config_artifact_manifest,
            "kg_pretrain_config",
            "kg_pretrain",
        ),
    ]
    values = {}
    if nodes_artifact_manifest is not None:
        for name, manifest_path, artifact_type, producer in specs:
            values[name], item = resolve_committed_artifact(
                context, manifest_path, artifact_type=artifact_type, producer_node_id=producer
            )
            upstream.append(item)
        execution.input_artifact_ids = tuple(
            dict.fromkeys((*execution.input_artifact_ids, *(item.artifact_id for item in upstream)))
        )
    else:
        for name, path, label in (
            ("nodes_path", nodes_path, "kg_training_nodes"),
            ("edges_path", edges_path, "kg_training_edges"),
            ("training_manifest_path", training_manifest_path, "kg_training_manifest"),
            ("checkpoint_path", checkpoint_path, "kg_pretrain_checkpoint"),
            ("pretrain_config_path", pretrain_config_path, "kg_pretrain_config"),
        ):
            values[name] = resolve_run_file(context, path, label=label)
    pretrain_config = json.loads(values["pretrain_config_path"].read_text(encoding="utf-8"))
    if pretrain_config.get("checkpoint_format") != CHECKPOINT_FORMAT:
        raise InputError("pretrain checkpoint format is incompatible")
    training_manifest = json.loads(values["training_manifest_path"].read_text(encoding="utf-8"))
    config_dict = asdict(settings)
    seed_hash = seed_configuration_hash(
        seeds=configured,
        finetune_config=config_dict,
        checkpoint_sha256=sha256_file(values["checkpoint_path"]),
        training_manifest_sha256=sha256_file(values["training_manifest_path"]),
    )
    base = f"{seed_hash}/seed_{seed:03d}"
    paths = {
        "ranking": output_path(context, f"{base}/custom_disease_drug_ranking.csv"),
        "history": output_path(context, f"{base}/history.json"),
        "summary": output_path(context, f"{base}/summary.json"),
        "seed_manifest": output_path(context, f"{base}/manifest.json"),
        "model": output_path(context, f"{base}/model.pt"),
        "embeddings": output_path(context, f"{base}/node_embeddings.pt"),
        "seed_configuration_hash": seed_hash,
    }
    cache_key = json.dumps(
        {"seed": seed, "seed_configuration_hash": seed_hash, "code_version": code_version},
        sort_keys=True,
        separators=(",", ":"),
    )
    if settings.resume_completed_seeds and _cache_matches(paths, cache_key):
        for artifact_type, name in (
            ("kg_seed_ranking", "ranking"),
            ("kg_seed_history", "history"),
            ("kg_seed_summary", "summary"),
            ("kg_seed_manifest", "seed_manifest"),
        ):
            execution.add_output(artifact_type, paths[name])
        if settings.save_per_seed_models and paths["model"].is_file():
            execution.add_output("kg_seed_checkpoint", paths["model"])
        if settings.save_per_seed_embeddings and paths["embeddings"].is_file():
            execution.add_output("kg_seed_embeddings", paths["embeddings"])
        execution.metric("seed", seed)
        execution.metric("seed_configuration_hash", seed_hash)
        execution.mark_cached("matching per-seed manifest and file hashes were found")
        return
    deps = load_training_dependencies(require_dgl=True)
    device = resolve_device(deps["torch"], settings.device, allow_cpu=allow_cpu_small_graph)
    environment = environment_snapshot(torch=deps["torch"], dgl=deps["dgl"])
    nodes = deps["pandas"].read_csv(values["nodes_path"], low_memory=False)
    directed = deps["pandas"].read_csv(values["edges_path"], compression="gzip", low_memory=False)
    summary = run_finetune(
        seed=seed,
        nodes=nodes,
        directed=directed,
        training_manifest=training_manifest,
        pretrain_config=pretrain_config,
        checkpoint_path=values["checkpoint_path"],
        settings=settings,
        paths=paths,
        device=device,
        deps=deps,
        environment=environment,
        cache_key=cache_key,
        allowed_root=context.output_dir,
    )
    for artifact_type, name in (
        ("kg_seed_ranking", "ranking"),
        ("kg_seed_history", "history"),
        ("kg_seed_summary", "summary"),
        ("kg_seed_manifest", "seed_manifest"),
    ):
        execution.add_output(artifact_type, paths[name])
    if settings.save_per_seed_models:
        execution.add_output("kg_seed_checkpoint", paths["model"])
    if settings.save_per_seed_embeddings:
        execution.add_output("kg_seed_embeddings", paths["embeddings"])
    execution.update_metrics(
        {
            "seed": seed,
            "seed_configuration_hash": seed_hash,
            "training_seconds": summary["training_seconds"],
            "epochs_completed": len(summary["epoch_seconds"]),
            "device": str(device),
        }
    )


def kg_finetune_seed(
    *,
    context: RunContext,
    settings: KGFinetuneConfig,
    seed: int,
    config_hash: str,
    code_version: str,
    nodes_path=None,
    edges_path=None,
    training_manifest_path=None,
    checkpoint_path=None,
    pretrain_config_path=None,
    nodes_artifact_manifest=None,
    edges_artifact_manifest=None,
    training_artifact_manifest=None,
    checkpoint_artifact_manifest=None,
    pretrain_config_artifact_manifest=None,
    allow_cpu_small_graph=False,
    task_id=None,
    attempt=1,
    input_artifact_ids=(),
):
    return execute_node(
        lambda execution: _operation(
            execution,
            context=context,
            settings=settings,
            seed=int(seed),
            nodes_path=nodes_path,
            edges_path=edges_path,
            training_manifest_path=training_manifest_path,
            checkpoint_path=checkpoint_path,
            pretrain_config_path=pretrain_config_path,
            nodes_artifact_manifest=nodes_artifact_manifest,
            edges_artifact_manifest=edges_artifact_manifest,
            training_artifact_manifest=training_artifact_manifest,
            checkpoint_artifact_manifest=checkpoint_artifact_manifest,
            pretrain_config_artifact_manifest=pretrain_config_artifact_manifest,
            allow_cpu_small_graph=allow_cpu_small_graph,
            code_version=code_version,
        ),
        context=context,
        node_id=NODE_ID,
        task_id=task_id or f"seed-{int(seed):03d}",
        attempt=attempt,
        config_hash=config_hash,
        code_version=code_version,
        input_artifact_ids=input_artifact_ids,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Fine-tune one independent KG seed.")
    add_common_runner_arguments(parser)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--training-nodes-manifest", required=True, type=Path)
    parser.add_argument("--training-edges-manifest", required=True, type=Path)
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument("--pretrain-checkpoint-manifest", required=True, type=Path)
    parser.add_argument("--pretrain-config-manifest", required=True, type=Path)
    parser.add_argument("--allow-cpu-small-graph", action="store_true")
    add_execution_identity_arguments(parser)
    parser.set_defaults(task_id=None)
    return parser


def main(argv=None):
    ns = build_parser().parse_args(argv)
    env = load_common_runner_environment(
        CommonRunnerArguments.from_namespace(ns), project_root=Path(__file__).resolve().parents[4]
    )
    task_id, attempt, artifact_ids = execution_identity(ns)
    result = kg_finetune_seed(
        context=env.context,
        settings=env.config.kg.finetune,
        seed=ns.seed,
        config_hash=env.config_hash,
        code_version=__version__,
        nodes_artifact_manifest=ns.training_nodes_manifest,
        edges_artifact_manifest=ns.training_edges_manifest,
        training_artifact_manifest=ns.training_manifest,
        checkpoint_artifact_manifest=ns.pretrain_checkpoint_manifest,
        pretrain_config_artifact_manifest=ns.pretrain_config_manifest,
        allow_cpu_small_graph=ns.allow_cpu_small_graph,
        task_id=task_id,
        attempt=attempt,
        input_artifact_ids=artifact_ids,
    )
    sys.stdout.write(result.to_json() + "\n")
    return 0 if result.status in {NodeStatus.SUCCEEDED, NodeStatus.CACHED} else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "kg_finetune_seed", "main", "run_finetune"]
