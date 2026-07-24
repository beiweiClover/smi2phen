import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lipid_screening_agent.artifacts import NodeStatus, load_artifact_manifest
from lipid_screening_agent.config import hash_workflow_config, load_workflow_config
from lipid_screening_agent.runners.gps import predict_drug_profiles as runner
from lipid_screening_agent.runners.gps._dependencies import PredictionDependencies
from lipid_screening_agent.runtime import EnvironmentError, RunContext, sha256_file

Chem = pytest.importorskip(
    "rdkit.Chem", reason="GPS profile unit tests require the optional RDKit extra"
)
rdBase = pytest.importorskip("rdkit.rdBase")
AllChem = pytest.importorskip("rdkit.Chem.AllChem")

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONFIG = load_workflow_config(PROJECT_ROOT / "configs/workflow.yaml")


class _Cuda:
    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def manual_seed_all(seed: int) -> None:
        return None


class _Torch:
    __version__ = "mock-torch"
    cuda = _Cuda()
    backends = SimpleNamespace(cudnn=SimpleNamespace(deterministic=False, benchmark=True))

    @staticmethod
    def device(value: str) -> str:
        return value

    @staticmethod
    def manual_seed(seed: int) -> None:
        return None


def _context(tmp_path: Path) -> tuple[RunContext, Path, Path]:
    project = tmp_path / "project"
    resources = tmp_path / "gps-resources"
    prepared = project / "runs/gps-run/inputs/prepared"
    output = project / "runs/gps-run/artifacts/gps"
    project.mkdir()
    resources.mkdir()
    context = RunContext.create(
        runs_root=project / "runs",
        run_id="gps-run",
        project_root=project,
        resource_dir=resources,
        input_dir=prepared,
        output_dir=output,
    )
    return context, prepared, resources


def _mock_resources(resources: Path) -> None:
    code = resources / "GPS4Drugs/code"
    data = resources / "GPS4Drugs/data/input_gene_features"
    code.mkdir(parents=True)
    data.mkdir(parents=True)
    (code / "model.py").write_text("# mock resource model definition\n", encoding="utf-8")
    for cell_line in runner.LEGACY_CELL_LINES:
        checkpoint = code / "results" / cell_line / "multi" / "model.pkl"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"mock-checkpoint")
        pd.DataFrame(
            {"feature-1": [0.1, 0.3], "feature-2": [0.2, 0.4]},
            index=pd.Index(["GENEA", "GENEB"], name="GeneSymbol"),
        ).to_csv(data / f"go_fingerprints_2k_{cell_line}.csv")


def _dependencies() -> PredictionDependencies:
    return PredictionDependencies(
        np=np,
        pd=pd,
        torch=_Torch,
        functional=SimpleNamespace(),
        chem=Chem,
        all_chem=AllChem,
        rdkit_version=rdBase.rdkitVersion,
    )


def test_prediction_runner_uses_mock_models_and_writes_hepg2_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, prepared, resources = _context(tmp_path)
    _mock_resources(resources)
    compounds = prepared / "compounds.normalized.csv"
    compounds.parent.mkdir(parents=True, exist_ok=True)
    compounds.write_text("ID,SMILES\ncmp-a,CCO\ncmp-b,CCN\n", encoding="utf-8")
    loaded_cell_lines: list[str] = []

    monkeypatch.setattr(runner, "load_prediction_dependencies", _dependencies)

    def load_model(**kwargs):
        cell_line = kwargs["model_file"].parents[1].name
        loaded_cell_lines.append(cell_line)
        return cell_line

    monkeypatch.setattr(runner, "_load_gps_model", load_model)

    def predict(model, fingerprints, gene_feature, **kwargs):
        return np.asarray([[0.01, 0.01, 0.98], [0.98, 0.01, 0.01]], dtype=np.float32)

    monkeypatch.setattr(runner, "_predict_gene_probabilities", predict)
    result = runner.gps_predict_drug_profiles(
        context=context,
        compounds_path=compounds,
        model_code_path=CONFIG.resources.gps.model_code.raw,
        model_data_path=CONFIG.resources.gps.model_data.raw,
        settings=CONFIG.gps.drug_profiles,
        config_hash=hash_workflow_config(CONFIG),
        code_version="test-version",
        input_artifact_ids=("compounds-artifact",),
    )

    assert result.status is NodeStatus.SUCCEEDED
    assert loaded_cell_lines == list(runner.LEGACY_CELL_LINES)
    assert result.metrics["device_actual"] == "cpu"
    assert result.metrics["torch_version"] == "mock-torch"
    assert result.metrics["input_compound_count"] == 2
    output = context.resolve_run_relative(runner.OUTPUT_RELATIVE_PATH, must_exist=True)
    profile = pd.read_csv(output, compression="gzip", index_col=0)
    assert profile.index.name == "GeneSymbol"
    assert profile.index.tolist() == ["GENEA", "GENEB"]
    assert profile.columns.tolist() == ["cmp-a", "cmp-b"]
    assert set(np.unique(profile.to_numpy())) == {-1, 1}

    manifest_path = (
        context.run_dir
        / "artifacts/manifests/gps_predict_drug_profiles/main"
        / f"{result.outputs[0]}.json"
    )
    manifest = load_artifact_manifest(manifest_path, run_root=context.run_dir)
    assert manifest.artifact_type == "gps_drug_profiles"
    assert manifest.relative_path == runner.OUTPUT_RELATIVE_PATH
    assert manifest.sha256 == sha256_file(output)
    assert len(manifest.resource_hashes) == 9

    events = [
        json.loads(line)["event"]
        for line in (context.run_dir / "logs/gps_predict_drug_profiles/main.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert "fingerprint_progress" in events
    assert "gene_prediction_progress" in events


def test_checkpoint_loading_uses_resource_model_and_legacy_whole_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_definition = tmp_path / "model.py"
    checkpoint = tmp_path / "model.pkl"
    model_definition.write_text("# fixture\n", encoding="utf-8")
    checkpoint.write_bytes(b"fixture")

    class Model:
        def __init__(self):
            self.device = None
            self.evaluated = False

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.evaluated = True

    model = Model()
    calls = {}

    class Torch:
        __version__ = "fixture-torch"

        @staticmethod
        def load(path, *, map_location, weights_only=True):
            calls.update(
                path=path,
                map_location=map_location,
                weights_only=weights_only,
            )
            return {"model0": model}

    monkeypatch.setattr(runner, "_load_model_module", lambda path: object())
    loaded = runner._load_gps_model(
        torch=Torch,
        model_file=checkpoint,
        model_definition=model_definition,
        device="cpu",
    )

    assert loaded is model
    assert model.device == "cpu"
    assert model.evaluated is True
    assert calls["weights_only"] is False


def test_dependency_failure_is_structured_environment_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, prepared, resources = _context(tmp_path)
    _mock_resources(resources)
    compounds = prepared / "compounds.normalized.csv"
    compounds.parent.mkdir(parents=True, exist_ok=True)
    compounds.write_text("ID,SMILES\ncmp-a,CCO\n", encoding="utf-8")

    def unavailable():
        raise EnvironmentError("heavy GPS dependencies unavailable", retryable=False)

    monkeypatch.setattr(runner, "load_prediction_dependencies", unavailable)
    result = runner.gps_predict_drug_profiles(
        context=context,
        compounds_path=compounds,
        model_code_path=CONFIG.resources.gps.model_code.raw,
        model_data_path=CONFIG.resources.gps.model_data.raw,
        settings=CONFIG.gps.drug_profiles,
        config_hash=hash_workflow_config(CONFIG),
        code_version="test-version",
    )

    assert result.status is NodeStatus.FAILED
    assert result.error is not None
    assert result.error.category.value == "environment"
    assert result.metrics["elapsed_seconds"] >= 0
