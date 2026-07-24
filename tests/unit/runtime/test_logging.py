import json
import logging
from pathlib import Path

from lipid_screening_agent.runtime.logging import create_node_logger


def test_node_logger_writes_jsonl_and_human_log_without_duplicates(
    tmp_path: Path,
) -> None:
    jsonl = tmp_path / "logs" / "node.jsonl"
    human = tmp_path / "logs" / "node.log"
    logger = create_node_logger(
        run_id="run-01",
        node_id="prepare_compound_library",
        task_id="main",
        jsonl_path=jsonl,
        human_path=human,
        allowed_root=tmp_path,
    )
    logger.info("started", "开始", count=2)
    logger.warning("quality", "one warning", values=[1, 2])
    logger.close()

    records = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert records[0]["message"] == "开始"
    assert records[0]["fields"] == {"count": 2}
    assert records[0]["timestamp"].endswith("Z")
    assert records[1]["level"] == "warning"

    human_text = human.read_text(encoding="utf-8")
    assert human_text.count("开始") == 1
    assert "run-01/prepare_compound_library/main" in human_text


def test_node_logger_honors_requested_level(tmp_path: Path) -> None:
    logger = create_node_logger(
        run_id="run-02",
        node_id="node",
        task_id="main",
        jsonl_path=tmp_path / "node.jsonl",
        human_path=tmp_path / "node.log",
        allowed_root=tmp_path,
        level=logging.WARNING,
    )
    logger.info("ignored", "not written")
    logger.warning("written", "visible")
    logger.close()

    assert len((tmp_path / "node.jsonl").read_text(encoding="utf-8").splitlines()) == 1
