"""All-relation TxGNN-compatible RGCN pretraining runner."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

from lipid_screening_agent import __version__
from lipid_screening_agent.artifacts import NodeResult, NodeStatus
from lipid_screening_agent.cli import (
    CommonRunnerArguments,
    add_common_runner_arguments,
    load_common_runner_environment,
)
from lipid_screening_agent.config.models import KGPretrainConfig
from lipid_screening_agent.runtime import InputError, RunContext, atomic_write_json, sha256_file
from lipid_screening_agent.runtime.execution import NodeExecution, execute_node

from ..input_prepare._common import add_execution_identity_arguments, execution_identity
from ._training_common import (
    atomic_torch_save,
    environment_snapshot,
    load_training_dependencies,
    output_path,
    resolve_committed_artifact,
    resolve_device,
    resolve_run_file,
    set_random_seed,
)

NODE_ID = "kg_pretrain"
CHECKPOINT_FORMAT = "txgnn_minimal.HeteroRGCN.state_dict.v1"
OUTPUT_FILES = {
    "kg_pretrain_checkpoint": "best_pretrain_model.pt",
    "kg_pretrain_config": "config.json",
    "kg_pretrain_history": "history.json",
    "kg_pretrain_summary": "summary.json",
    "kg_pretrain_embeddings": "node_embeddings.pt",
}


def _build_model(model_class, graph, settings, device, *, data_folder: str):
    model = settings.model
    return model_class(
        graph,
        in_size=model.input_dimension,
        hidden_size=model.hidden_dimension,
        out_size=model.output_dimension,
        attention=model.attention,
        proto=model.prototype_learning,
        proto_num=model.prototype_count,
        sim_measure=model.similarity_measure,
        bert_measure=model.bert_measure,
        agg_measure=model.aggregation_measure,
        num_walks=model.random_walks,
        walk_mode=model.walk_mode,
        path_length=model.path_length,
        split="custom_pretrain_agent_kg",
        data_folder=data_folder,
        exp_lambda=model.exponential_lambda,
        device=device,
    ).to(device)


def _edge_dataloader(dgl, graph, eids, sampler, negative_sampler, batch_size):
    if hasattr(dgl.dataloading, "EdgeDataLoader"):
        return dgl.dataloading.EdgeDataLoader(
            graph,
            eids,
            sampler,
            negative_sampler=negative_sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
    edge_sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler, negative_sampler=negative_sampler
    )
    return dgl.dataloading.DataLoader(
        graph,
        eids,
        edge_sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )


def _finite(value, *, fallback=None):
    value = float(value)
    return value if math.isfinite(value) else fallback


def _evaluate(model, message_graph, pos_graph, neg_graph, device, torch, F, metrics_fn, mode):
    model.eval()
    with torch.no_grad():
        pred_pos, pred_neg, pos_score, neg_score = model(
            message_graph.to(device),
            neg_graph.to(device),
            pos_graph.to(device),
            pretrain_mode=True,
            mode=mode,
        )
    scores = torch.cat((pos_score, neg_score)).reshape(-1)
    labels = torch.cat((torch.ones_like(pos_score), torch.zeros_like(neg_score))).float()
    if not scores.numel():
        raise InputError(f"{mode} split contains no evaluable edges")
    loss = F.binary_cross_entropy(scores, labels)
    values = metrics_fn(
        pred_pos,
        pred_neg,
        scores.detach().cpu().numpy(),
        labels.detach().cpu().numpy(),
        message_graph,
        True,
    )
    auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc = values
    return {
        "loss": float(loss.item()),
        "micro_auroc": _finite(micro_auroc),
        "micro_auprc": _finite(micro_auprc),
        "macro_auroc": _finite(macro_auroc),
        "macro_auprc": _finite(macro_auprc),
        "auroc_rel": {str(k): float(v) for k, v in auroc_rel.items()},
        "auprc_rel": {str(k): float(v) for k, v in auprc_rel.items()},
    }


def run_pretraining(
    *,
    nodes,
    directed,
    training_manifest,
    settings,
    output_paths,
    device,
    dependencies,
    environment,
    input_hashes,
    allowed_root,
):
    numpy, torch, dgl = (
        dependencies["numpy"],
        dependencies["torch"],
        dependencies["dgl"],
    )
    F = torch.nn.functional
    from .model import (
        HeteroRGCN,
        Minibatch_NegSampler,
        create_dgl_graph,
        evaluate_graph_construct,
        get_all_metrics_fb,
        initialize_node_embedding,
    )

    seed = int(training_manifest.get("configuration", {}).get("seed", 42))
    set_random_seed(seed, numpy=numpy, torch=torch, dgl=dgl)
    directed[["x_idx", "y_idx"]] = directed[["x_idx", "y_idx"]].astype(int)
    train = directed[directed.split == "train"].copy()
    valid = directed[directed.split == "valid"].copy()
    test = directed[directed.split == "test"].copy()
    if any(frame.empty for frame in (train, valid, test)):
        raise InputError(
            "pretraining requires non-empty train, valid, and test splits",
            details={"train": len(train), "valid": len(valid), "test": len(test)},
        )
    columns = ["x_type", "relation", "y_type", "x_idx", "y_idx"]
    graph = create_dgl_graph(train[columns], directed[columns])
    graph = initialize_node_embedding(graph, settings.model.input_dimension)
    valid_pos, valid_neg = evaluate_graph_construct(
        valid[["x_idx", "relation", "y_idx"]],
        graph,
        settings.negative_method,
        settings.negative_samples_per_positive,
        device,
    )
    test_pos, test_neg = evaluate_graph_construct(
        test[["x_idx", "relation", "y_idx"]],
        graph,
        settings.negative_method,
        settings.negative_samples_per_positive,
        device,
    )
    model = _build_model(HeteroRGCN, graph, settings, device, data_folder="")
    eids = {etype: graph.edges(form="eid", etype=etype) for etype in graph.canonical_etypes}
    dataloader = _edge_dataloader(
        dgl,
        graph,
        eids,
        dgl.dataloading.MultiLayerFullNeighborSampler(2),
        Minibatch_NegSampler(
            graph, settings.negative_samples_per_positive, settings.negative_method
        ),
        settings.batch_size,
    )
    if settings.optimizer.casefold() != "adamw":
        raise InputError("checkpoint-compatible pretraining supports optimizer=adamw")
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
    config_payload = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "variant": "agent_kg",
        "seed": seed,
        "device": str(device),
        "pretrain": asdict(settings),
        "model_constructor": {
            "n_inp": settings.model.input_dimension,
            "n_hid": settings.model.hidden_dimension,
            "n_out": settings.model.output_dimension,
            "attention": settings.model.attention,
            "proto": settings.model.prototype_learning,
            "proto_num": settings.model.prototype_count,
            "sim_measure": settings.model.similarity_measure,
            "bert_measure": settings.model.bert_measure,
            "agg_measure": settings.model.aggregation_measure,
            "exp_lambda": settings.model.exponential_lambda,
            "num_walks": settings.model.random_walks,
            "walk_mode": settings.model.walk_mode,
            "path_length": settings.model.path_length,
        },
        "canonical_etypes": [list(value) for value in graph.canonical_etypes],
        "node_counts": {kind: int(graph.num_nodes(kind)) for kind in graph.ntypes},
        "edge_counts": {"train": len(train), "valid": len(valid), "test": len(test)},
        "input_sha256": input_hashes,
        "environment": environment,
    }
    atomic_write_json(output_paths["kg_pretrain_config"], config_payload, allowed_root=allowed_root)
    history = []
    best_state = None
    best_metric = float("-inf")
    best_epoch = 0
    patience = 0
    training_started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, settings.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        losses = []
        for step, (_, pos_graph, neg_graph, blocks) in enumerate(dataloader, 1):
            blocks = [block.to(device) for block in blocks]
            pred_pos, pred_neg, pos_score, neg_score = model.forward_minibatch(
                pos_graph.to(device),
                neg_graph.to(device),
                blocks,
                graph,
                mode="train",
                pretrain_mode=True,
            )
            scores = torch.cat((pos_score, neg_score)).reshape(-1)
            labels = torch.cat((torch.ones_like(pos_score), torch.zeros_like(neg_score))).float()
            loss = F.binary_cross_entropy(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        if not losses:
            raise InputError("pretraining dataloader produced no batches")
        record = {
            "epoch": epoch,
            "train_loss_mean": float(numpy.mean(losses)),
            "train_loss_last": losses[-1],
        }
        if epoch % settings.validate_every_epochs == 0:
            valid_metrics = _evaluate(
                model,
                graph,
                valid_pos,
                valid_neg,
                device,
                torch,
                F,
                get_all_metrics_fb,
                "valid",
            )
            record["valid"] = valid_metrics
            current = valid_metrics.get(settings.best_model_metric)
            if current is None:
                raise InputError(
                    "configured pretrain best_model_metric is unavailable",
                    details={"metric": settings.best_model_metric},
                )
            if current > best_metric + settings.early_stopping_min_delta:
                best_metric = float(current)
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                atomic_torch_save(
                    torch,
                    best_state,
                    output_paths["kg_pretrain_checkpoint"],
                    allowed_root=allowed_root,
                )
                patience = 0
            else:
                patience += 1
        record["elapsed_seconds"] = float(time.perf_counter() - epoch_started)
        if device.type == "cuda":
            record["gpu_max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        history.append(record)
        atomic_write_json(output_paths["kg_pretrain_history"], history, allowed_root=allowed_root)
        if patience >= settings.early_stopping_patience:
            break
    if best_state is None:
        best_state = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
        best_epoch = len(history)
        atomic_torch_save(
            torch,
            best_state,
            output_paths["kg_pretrain_checkpoint"],
            allowed_root=allowed_root,
        )
    model.load_state_dict(best_state, strict=True)
    test_metrics = _evaluate(
        model, graph, test_pos, test_neg, device, torch, F, get_all_metrics_fb, "test"
    )
    training_seconds = float(time.perf_counter() - training_started)
    summary = {
        "status": "succeeded",
        "checkpoint_format": CHECKPOINT_FORMAT,
        "best_epoch": int(best_epoch),
        "best_valid_metric_name": settings.best_model_metric,
        "best_valid_metric": None if not math.isfinite(best_metric) else best_metric,
        "epochs_completed": len(history),
        "training_seconds": training_seconds,
        "epoch_seconds": [row["elapsed_seconds"] for row in history],
        "test": test_metrics,
        "environment": environment,
        "checkpoint_sha256": sha256_file(output_paths["kg_pretrain_checkpoint"]),
    }
    atomic_write_json(output_paths["kg_pretrain_summary"], summary, allowed_root=allowed_root)
    if settings.save_embeddings:
        model.eval()
        graph_device = graph.to(device)
        with torch.no_grad():
            embeddings = model(graph_device, graph_device, return_h=True)
        atomic_torch_save(
            torch,
            {kind: value.detach().cpu() for kind, value in embeddings.items()},
            output_paths["kg_pretrain_embeddings"],
            allowed_root=allowed_root,
        )
    return summary


def _operation(
    execution: NodeExecution,
    *,
    context: RunContext,
    settings: KGPretrainConfig,
    nodes_path,
    edges_path,
    training_manifest_path,
    nodes_artifact_manifest,
    edges_artifact_manifest,
    training_artifact_manifest,
    allow_cpu_small_graph: bool,
):
    started = time.perf_counter()
    upstream = []
    if nodes_artifact_manifest is not None:
        nodes_path, item = resolve_committed_artifact(
            context,
            nodes_artifact_manifest,
            artifact_type="kg_training_nodes",
            producer_node_id="kg_prepare_training_data",
        )
        upstream.append(item)
        edges_path, item = resolve_committed_artifact(
            context,
            edges_artifact_manifest,
            artifact_type="kg_training_edges",
            producer_node_id="kg_prepare_training_data",
        )
        upstream.append(item)
        training_manifest_path, item = resolve_committed_artifact(
            context,
            training_artifact_manifest,
            artifact_type="kg_training_manifest",
            producer_node_id="kg_prepare_training_data",
        )
        upstream.append(item)
        execution.input_artifact_ids = tuple(
            dict.fromkeys((*execution.input_artifact_ids, *(item.artifact_id for item in upstream)))
        )
    else:
        nodes_path = resolve_run_file(context, nodes_path, label="kg_training_nodes")
        edges_path = resolve_run_file(context, edges_path, label="kg_training_edges")
        training_manifest_path = resolve_run_file(
            context, training_manifest_path, label="kg_training_manifest"
        )
    deps = load_training_dependencies(require_dgl=True)
    device = resolve_device(deps["torch"], settings.device, allow_cpu=allow_cpu_small_graph)
    environment = environment_snapshot(torch=deps["torch"], dgl=deps["dgl"])
    nodes = deps["pandas"].read_csv(nodes_path, low_memory=False)
    directed = deps["pandas"].read_csv(edges_path, compression="gzip", low_memory=False)
    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    paths = {key: output_path(context, name) for key, name in OUTPUT_FILES.items()}
    summary = run_pretraining(
        nodes=nodes,
        directed=directed,
        training_manifest=training_manifest,
        settings=settings,
        output_paths=paths,
        device=device,
        dependencies=deps,
        environment=environment,
        input_hashes={
            "nodes": sha256_file(nodes_path),
            "edges": sha256_file(edges_path),
            "manifest": sha256_file(training_manifest_path),
        },
        allowed_root=context.output_dir,
    )
    for artifact_type in (
        "kg_pretrain_checkpoint",
        "kg_pretrain_config",
        "kg_pretrain_history",
        "kg_pretrain_summary",
    ):
        execution.add_output(artifact_type, paths[artifact_type])
    if settings.save_embeddings:
        execution.add_output("kg_pretrain_embeddings", paths["kg_pretrain_embeddings"])
    execution.update_metrics(
        {
            "epochs_completed": summary["epochs_completed"],
            "training_seconds": summary["training_seconds"],
            "elapsed_seconds": time.perf_counter() - started,
            "device": str(device),
        }
    )


def kg_pretrain(
    *,
    context: RunContext,
    settings: KGPretrainConfig,
    config_hash: str,
    code_version: str,
    nodes_path=None,
    edges_path=None,
    training_manifest_path=None,
    nodes_artifact_manifest=None,
    edges_artifact_manifest=None,
    training_artifact_manifest=None,
    allow_cpu_small_graph=False,
    task_id="main",
    attempt=1,
    input_artifact_ids=(),
) -> NodeResult:
    return execute_node(
        lambda execution: _operation(
            execution,
            context=context,
            settings=settings,
            nodes_path=nodes_path,
            edges_path=edges_path,
            training_manifest_path=training_manifest_path,
            nodes_artifact_manifest=nodes_artifact_manifest,
            edges_artifact_manifest=edges_artifact_manifest,
            training_artifact_manifest=training_artifact_manifest,
            allow_cpu_small_graph=allow_cpu_small_graph,
        ),
        context=context,
        node_id=NODE_ID,
        task_id=task_id,
        attempt=attempt,
        config_hash=config_hash,
        code_version=code_version,
        input_artifact_ids=input_artifact_ids,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Pretrain the all-relation KG RGCN.")
    add_common_runner_arguments(parser)
    parser.add_argument("--training-nodes-manifest", required=True, type=Path)
    parser.add_argument("--training-edges-manifest", required=True, type=Path)
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument(
        "--allow-cpu-small-graph",
        action="store_true",
        help="Explicitly allow CPU only for tiny tests; not suitable for the full KG.",
    )
    add_execution_identity_arguments(parser)
    return parser


def main(argv=None):
    ns = build_parser().parse_args(argv)
    env = load_common_runner_environment(
        CommonRunnerArguments.from_namespace(ns), project_root=Path(__file__).resolve().parents[4]
    )
    task_id, attempt, artifact_ids = execution_identity(ns)
    result = kg_pretrain(
        context=env.context,
        settings=env.config.kg.pretrain,
        config_hash=env.config_hash,
        code_version=__version__,
        nodes_artifact_manifest=ns.training_nodes_manifest,
        edges_artifact_manifest=ns.training_edges_manifest,
        training_artifact_manifest=ns.training_manifest,
        allow_cpu_small_graph=ns.allow_cpu_small_graph,
        task_id=task_id,
        attempt=attempt,
        input_artifact_ids=artifact_ids,
    )
    sys.stdout.write(result.to_json() + "\n")
    return 0 if result.status in {NodeStatus.SUCCEEDED, NodeStatus.CACHED} else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CHECKPOINT_FORMAT", "build_parser", "kg_pretrain", "main", "run_pretraining"]
