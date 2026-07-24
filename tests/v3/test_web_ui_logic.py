from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from lipid_screening_agent.backend.web_ui_logic import WEB_UI_LOGIC

NODE = shutil.which("node")


def run_javascript(body: str) -> dict:
    if NODE is None:
        pytest.skip("Node.js is unavailable; browser logic execution requires Node")
    script = WEB_UI_LOGIC + "\n" + body
    completed = subprocess.run(
        [NODE],
        input=script.encode("utf-8"),
        capture_output=True,
        check=False,
        timeout=15,
    )
    stderr = completed.stderr.decode("utf-8", errors="replace")
    assert completed.returncode == 0, stderr
    return json.loads(completed.stdout.decode("utf-8"))


def test_status_mapping_grouping_and_dynamic_nodes() -> None:
    result = run_javascript(
        r"""
const logic = globalThis.LipidUiLogic;
const initial = [
  {node_id: "register_inputs", task_id: "main", stage: "preparation", status: "succeeded"},
  {node_id: "netinfer_prepare_inputs", task_id: "main", stage: "netinfer", status: "running"},
  {node_id: "kg_pretrain", task_id: "main", stage: "kg", status: "pending"}
];
const dynamic = initial.concat([
  {node_id: "netinfer_predict_batch", task_id: "batch-003", stage: "netinfer", status: "queued"},
  {node_id: "kg_finetune_seed", task_id: "seed-17", stage: "kg", status: "ready"}
]);
const grouped = logic.groupNodes(dynamic);
const mapped = ["queued", "ready", "pending", "running", "succeeded", "skipped",
  "cancelled", "failed", "blocked"].map((status) => {
    const meta = logic.statusMeta(status);
    return {status, label: meta.label, icon: meta.icon, tone: meta.tone};
  });
process.stdout.write(JSON.stringify({
  mapped,
  initialNetinfer: logic.groupNodes(initial).netinfer.length,
  dynamicNetinfer: grouped.netinfer.length,
  dynamicKg: grouped.kg.length,
  stableKeys: dynamic.map(logic.stableNodeKey),
  cancelling: logic.effectiveRunStatus({status: "running", cancel_requested: true})
}));
"""
    )

    assert {item["status"] for item in result["mapped"]} == {
        "queued",
        "ready",
        "pending",
        "running",
        "succeeded",
        "skipped",
        "cancelled",
        "failed",
        "blocked",
    }
    assert all(item["label"] and item["icon"] and item["tone"] for item in result["mapped"])
    assert result["initialNetinfer"] == 1
    assert result["dynamicNetinfer"] == 2
    assert result["dynamicKg"] == 2
    assert len(result["stableKeys"]) == len(set(result["stableKeys"]))
    assert "netinfer_predict_batch::batch-003" in result["stableKeys"]
    assert result["cancelling"] == "cancelling"


def test_progress_null_is_indeterminate_and_module_summary_uses_real_values() -> None:
    result = run_javascript(
        r"""
const logic = globalThis.LipidUiLogic;
const now = Date.parse("2026-07-23T00:01:00Z");
const unknown = {
  node_id: "kg_pretrain", task_id: "main", stage: "kg", status: "running",
  progress: null, started_at: "2026-07-23T00:00:00Z"
};
const zero = {
  node_id: "gps_score_compounds", task_id: "main", stage: "gps", status: "running",
  progress: 0, started_at: "2026-07-23T00:00:30Z"
};
const real = {
  node_id: "netinfer_predict_batch", task_id: "batch-1", stage: "netinfer",
  status: "running", progress: 0.42, started_at: "2026-07-23T00:00:40Z"
};
const complete = {
  node_id: "netinfer_predict_known", task_id: "main", stage: "netinfer",
  status: "succeeded", progress: null, started_at: "2026-07-23T00:00:00Z",
  finished_at: "2026-07-23T00:00:20Z"
};
process.stdout.write(JSON.stringify({
  unknown: logic.progressLabel(unknown, now),
  zero: logic.progressLabel(zero, now),
  real: logic.progressLabel(real, now),
  moduleKnown: logic.moduleSummary([real, complete], now),
  moduleUnknown: logic.moduleSummary([unknown, complete], now)
}));
"""
    )

    assert result["unknown"] == "不确定进度 · 已运行 1 分 0 秒"
    assert result["zero"] == "真实进度 0% · 已运行 30 秒"
    assert result["real"] == "真实进度 42%"
    assert "42%" in result["moduleKnown"]["progress"]
    assert "不确定进度" in result["moduleUnknown"]["progress"]
    assert result["moduleUnknown"]["eta"] == "暂不可估算"


def test_run_progress_summary_uses_confirmed_progress_without_faking_unknowns() -> None:
    result = run_javascript(
        r"""
const logic = globalThis.LipidUiLogic;
const now = Date.parse("2026-07-23T00:05:00Z");
const snapshot = {
  status: "running",
  eta: {status: "unknown"},
  nodes: [
    {
      node_id: "register_inputs", task_id: "main", status: "succeeded",
      started_at: "2026-07-23T00:00:00Z", finished_at: "2026-07-23T00:00:30Z"
    },
    {
      node_id: "netinfer_predict_batch", task_id: "batch-1", status: "running",
      progress: 0.5, started_at: "2026-07-23T00:01:00Z"
    },
    {
      node_id: "kg_pretrain", task_id: "main", status: "running",
      progress: null, started_at: "2026-07-23T00:02:00Z"
    },
    {
      node_id: "rank_candidates", task_id: "main", status: "pending",
      progress: null
    }
  ]
};
process.stdout.write(JSON.stringify(logic.runProgressSummary(snapshot, now)));
"""
    )

    assert result["percent"] == 38
    assert result["completeCount"] == 1
    assert result["totalCount"] == 4
    assert result["uncertainCount"] == 2
    assert "已确认进度 38%" in result["progress"]
    assert "2 个未返回百分比" in result["progress"]
    assert result["elapsed"] == "5 分 0 秒"
    assert result["eta"] == "暂不可估算"


def test_terminal_snapshot_stops_polling_and_prevents_duplicate_timer() -> None:
    result = run_javascript(
        r"""
(async () => {
  const logic = globalThis.LipidUiLogic;
  let timerCallback = null;
  let timerCreates = 0;
  let timerClears = 0;
  let fetches = 0;
  const seen = [];
  const snapshots = [
    {run_id: "run-a", status: "running", nodes: []},
    {run_id: "run-a", status: "succeeded", nodes: []}
  ];
  const poller = logic.createPollController({
    intervalMs: 4000,
    fetchSnapshot: async () => snapshots[fetches++],
    onSnapshot: (snapshot) => seen.push(snapshot.status),
    setIntervalFn: (callback) => {
      timerCreates += 1;
      timerCallback = callback;
      return timerCreates;
    },
    clearIntervalFn: () => { timerClears += 1; }
  });
  await poller.setRun("run-a");
  await poller.refresh();
  const afterManualRefresh = poller.debugState();
  process.stdout.write(JSON.stringify({
    seen,
    fetches,
    timerCreates,
    timerClears,
    afterManualRefresh,
    final: poller.debugState()
  }));
})().catch((error) => {
  process.stderr.write(error.stack);
  process.exit(1);
});
"""
    )

    assert result["seen"] == ["running", "succeeded"]
    assert result["fetches"] == 2
    assert result["timerCreates"] == 1
    assert result["timerClears"] == 1
    assert result["afterManualRefresh"]["timerActive"] is False
    assert result["final"]["timerActive"] is False


def test_cancel_requires_confirmation_deduplicates_and_ignores_switched_run() -> None:
    result = run_javascript(
        r"""
(async () => {
  const logic = globalThis.LipidUiLogic;
  let context = {runId: "run-a", snapshot: {status: "running", cancel_requested: false}};
  let requests = 0;
  let starts = 0;
  let errors = 0;
  const denied = logic.createCancelController({
    getContext: () => context,
    confirmCancel: async () => false,
    requestCancel: async () => { requests += 1; }
  });
  const deniedResult = await denied.cancel();

  let releaseRequest;
  const requestGate = new Promise((resolve) => { releaseRequest = resolve; });
  const confirmed = logic.createCancelController({
    getContext: () => context,
    confirmCancel: async () => true,
    requestCancel: async () => {
      requests += 1;
      await requestGate;
      return {run_id: "run-a", status: "running", cancel_requested: true};
    },
    onStart: () => { starts += 1; },
    onError: () => { errors += 1; }
  });
  const first = confirmed.cancel();
  const second = confirmed.cancel();
  await Promise.resolve();
  releaseRequest();
  const duplicateResults = await Promise.all([first, second]);

  let releaseConfirm;
  const confirmGate = new Promise((resolve) => { releaseConfirm = resolve; });
  const switched = logic.createCancelController({
    getContext: () => context,
    confirmCancel: async () => confirmGate,
    requestCancel: async () => { requests += 1; }
  });
  const switchedResultPromise = switched.cancel();
  context = {runId: "run-b", snapshot: {status: "running", cancel_requested: false}};
  releaseConfirm(true);
  const switchedResult = await switchedResultPromise;

  const cancellingAllowed = logic.canCancel({
    status: "running", cancel_requested: true
  });
  process.stdout.write(JSON.stringify({
    deniedResult,
    duplicateResults,
    switchedResult,
    requests,
    starts,
    errors,
    cancellingAllowed
  }));
})().catch((error) => {
  process.stderr.write(error.stack);
  process.exit(1);
});
"""
    )

    assert result["deniedResult"] is False
    assert sorted(result["duplicateResults"]) == [False, True]
    assert result["switchedResult"] is False
    assert result["requests"] == 1
    assert result["starts"] == 1
    assert result["errors"] == 0
    assert result["cancellingAllowed"] is False


def test_ndjson_parser_handles_fragmented_events_once_and_reports_invalid_json() -> None:
    result = run_javascript(
        r"""
const logic = globalThis.LipidUiLogic;
const events = [];
const parser = logic.createNdjsonParser((event) => events.push(event));
parser.push('{"type":"stage","stage":"agent","status":"started"}\n{"type":"final","content":"你');
parser.push('好","run_id":"run-a"}\n');
parser.finish();
let invalidError = null;
try {
  const invalid = logic.createNdjsonParser(() => {});
  invalid.push('{"type":oops}\n');
} catch (error) {
  invalidError = error.message;
}
process.stdout.write(JSON.stringify({events, invalidError}));
"""
    )

    assert [event["type"] for event in result["events"]] == ["stage", "final"]
    assert result["events"][1]["content"] == "你好"
    assert "无法解析流式事件" in result["invalidError"]
