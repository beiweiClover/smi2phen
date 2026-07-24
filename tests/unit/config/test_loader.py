from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from lipid_screening_agent.config import (
    PathTemplate,
    canonical_config_dict,
    hash_workflow_config,
    load_workflow_config,
    parse_workflow_config,
    resolve_configured_paths,
    resolve_resource_path,
    resolve_resource_paths,
    resolve_run_workspace_parent,
)
from lipid_screening_agent.runtime.errors import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_CONFIG = PROJECT_ROOT / "configs" / "workflow.yaml"


def raw_config() -> dict:
    with WORKFLOW_CONFIG.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    assert isinstance(value, dict)
    return value


def test_loads_frozen_workflow_into_typed_immutable_models() -> None:
    config = load_workflow_config(WORKFLOW_CONFIG)

    assert config.schema_version == "1.0"
    assert config.workflow.default_mode == "core"
    assert config.workflow.modes["enhanced"].gps_branch == "required"
    assert config.disease.custom_node_id == "disease:custom:hepatic_steatosis"
    assert config.resources.gps.root == PathTemplate("${LIPID_AGENT_GPS_RESOURCE_DIR}")
    assert config.kg.finetune.seeds == (5, 6, 7, 8, 9)
    assert config.ranking.kg.top_n == 200
    assert canonical_config_dict(config) == raw_config()

    with pytest.raises(TypeError):
        config.workflow.modes["other"] = config.workflow.modes["core"]  # type: ignore[index]


def test_loading_does_not_require_or_expand_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "LIPID_AGENT_RUNS_DIR",
        "LIPID_AGENT_GPS_RESOURCE_DIR",
        "LIPID_AGENT_NETINFER_RESOURCE_DIR",
        "LIPID_AGENT_PPI_RESOURCE_FILE",
        "LIPID_AGENT_KG_BASE_RESOURCE_DIR",
    ):
        monkeypatch.delenv(name, raising=False)

    config = load_workflow_config(WORKFLOW_CONFIG)

    assert config.run_workspace.parent.raw == "${LIPID_AGENT_RUNS_DIR}"
    assert config.resource_reference_semantics.environment_placeholder == "${VARIABLE_NAME}"
    with pytest.raises(ConfigurationError, match="LIPID_AGENT_RUNS_DIR"):
        resolve_configured_paths(config, environ={})


def test_resolves_only_modeled_paths_and_joins_resource_children(
    tmp_path: Path,
) -> None:
    config = load_workflow_config(WORKFLOW_CONFIG)
    environment = {
        "LIPID_AGENT_RUNS_DIR": str(tmp_path / "runs"),
        "LIPID_AGENT_GPS_RESOURCE_DIR": str(tmp_path / "gps"),
        "LIPID_AGENT_NETINFER_RESOURCE_DIR": str(tmp_path / "netinfer"),
        "LIPID_AGENT_PPI_RESOURCE_FILE": str(tmp_path / "ppi" / "interactome.tsv"),
        "LIPID_AGENT_KG_BASE_RESOURCE_DIR": str(tmp_path / "kg"),
    }

    resolved = resolve_configured_paths(config, environ=environment)

    assert resolved.run_workspace_parent == tmp_path / "runs"
    assert resolved.resources.gps.model_code == tmp_path / "gps" / "GPS4Drugs" / "code"
    assert resolved.resources.netinfer.drug_target_network == tmp_path / "netinfer" / "DT.tsv"
    assert resolved.resources.proximity.interactome == tmp_path / "ppi" / "interactome.tsv"
    assert resolved.resources.kg.node_table == tmp_path / "kg" / "node.csv"
    assert config.resource_reference_semantics.environment_placeholder == "${VARIABLE_NAME}"


def test_granular_resolution_does_not_require_unrelated_environment(
    tmp_path: Path,
) -> None:
    config = load_workflow_config(WORKFLOW_CONFIG)
    gps_root = tmp_path / "gps"

    human_gene_info = resolve_resource_path(
        config,
        "resources.gps.human_gene_info",
        environ={"LIPID_AGENT_GPS_RESOURCE_DIR": str(gps_root)},
    )

    assert human_gene_info == gps_root / "Homo_sapiens.gene_info.gz"
    selected = resolve_resource_paths(
        config,
        ["resources.gps.root", "resources.gps.model_data"],
        environ={"LIPID_AGENT_GPS_RESOURCE_DIR": str(gps_root)},
    )
    assert selected == {
        "resources.gps.root": gps_root,
        "resources.gps.model_data": gps_root / "GPS4Drugs" / "data",
    }
    with pytest.raises(TypeError):
        selected["resources.gps.root"] = tmp_path  # type: ignore[index]


def test_granular_resolution_validates_reference_and_workspace_variable(
    tmp_path: Path,
) -> None:
    config = load_workflow_config(WORKFLOW_CONFIG)

    assert (
        resolve_run_workspace_parent(
            config,
            environ={"LIPID_AGENT_RUNS_DIR": str(tmp_path / "runs")},
        )
        == tmp_path / "runs"
    )
    with pytest.raises(ConfigurationError, match="Unknown resource reference"):
        resolve_resource_path(config, "resources.unknown", environ={})


def test_resolved_root_paths_must_be_host_native_and_absolute() -> None:
    config = load_workflow_config(WORKFLOW_CONFIG)

    with pytest.raises(ConfigurationError, match="absolute"):
        resolve_run_workspace_parent(
            config,
            environ={"LIPID_AGENT_RUNS_DIR": "./runs"},
        )
    with pytest.raises(ConfigurationError, match="absolute"):
        resolve_resource_path(
            config,
            "resources.gps.root",
            environ={"LIPID_AGENT_GPS_RESOURCE_DIR": "resources/gps"},
        )


def test_hash_is_stable_across_yaml_key_order_and_changes_with_values() -> None:
    original_document = raw_config()
    reordered_document = dict(reversed(list(original_document.items())))

    original = parse_workflow_config(original_document)
    reordered = parse_workflow_config(reordered_document)
    assert hash_workflow_config(original) == hash_workflow_config(reordered)
    assert len(hash_workflow_config(original)) == 64

    changed_document = deepcopy(original_document)
    changed_document["ranking"]["kg"]["top_n"] = 201
    changed = parse_workflow_config(changed_document)
    assert hash_workflow_config(original) != hash_workflow_config(changed)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda doc: doc.update(schema_version="2.0"), "schema_version"),
        (lambda doc: doc.update(unexpected=True), "unexpected keys"),
        (
            lambda doc: doc["kg"]["finetune"].update(seeds=[5, 5]),
            "values must be unique",
        ),
        (
            lambda doc: doc["netinfer"].update(batch_size=True),
            "expected an integer",
        ),
        (
            lambda doc: doc["resources"]["gps"].update(model_code="../outside"),
            "parent traversal",
        ),
        (
            lambda doc: doc["resources"]["kg"].update(node_table=r"C:\outside.csv"),
            "must be relative",
        ),
    ],
)
def test_rejects_schema_type_and_path_errors(mutate, message: str) -> None:
    document = raw_config()
    mutate(document)

    with pytest.raises(ConfigurationError, match=message):
        parse_workflow_config(document)


def test_reports_invalid_yaml_as_configuration_error(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("workflow: [unterminated", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Cannot load workflow config"):
        load_workflow_config(invalid)
