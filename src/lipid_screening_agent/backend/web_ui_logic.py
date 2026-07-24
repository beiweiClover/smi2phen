"""Pure browser-side state projection used by the dependency-free Web UI.

The JavaScript deliberately has no DOM dependency so its status, polling, and cancellation
semantics can be executed in Node during automated tests.
"""

WEB_UI_LOGIC = r"""
(() => {
  "use strict";

  const STATUS = Object.freeze({
    none: {icon: "○", label: "暂无任务", tone: "muted"},
    unavailable: {icon: "○", label: "尚未规划", tone: "muted"},
    collecting: {icon: "○", label: "收集输入", tone: "muted"},
    created: {icon: "○", label: "任务尚未启动", tone: "muted"},
    pending: {icon: "○", label: "等待依赖", tone: "muted"},
    ready: {icon: "◉", label: "已就绪", tone: "ready"},
    queued: {icon: "◷", label: "已排队", tone: "queued"},
    running: {icon: "▶", label: "运行中", tone: "running"},
    cancelling: {icon: "◼", label: "正在终止", tone: "queued"},
    succeeded: {icon: "✓", label: "已成功", tone: "success"},
    cached: {icon: "↻", label: "缓存命中", tone: "success"},
    skipped: {icon: "−", label: "已跳过", tone: "muted"},
    cancelled: {icon: "■", label: "已取消", tone: "cancelled"},
    failed: {icon: "✕", label: "失败", tone: "danger"},
    blocked: {icon: "!", label: "已阻塞", tone: "danger"}
  });
  const TERMINAL_RUN = new Set(["succeeded", "cancelled", "failed", "blocked"]);
  const TERMINAL_NODE = new Set([
    "succeeded", "cached", "skipped", "cancelled", "failed", "blocked"
  ]);
  const GROUPS = Object.freeze([
    {key: "preparation", label: "输入准备"},
    {key: "gps", label: "GPS"},
    {key: "netinfer", label: "NetInfer"},
    {key: "proximity", label: "Proximity"},
    {key: "kg", label: "KG"},
    {key: "final", label: "综合排序与报告"}
  ]);

  const numberOrNull = (value) => {
    if (value === null || value === undefined || value === "") return null;
    const decoded = Number(value);
    return Number.isFinite(decoded) ? decoded : null;
  };

  const effectiveRunStatus = (snapshot) => {
    if (!snapshot) return "none";
    const status = String(snapshot.status || "collecting");
    if (snapshot.cancel_requested && !TERMINAL_RUN.has(status)) return "cancelling";
    return status;
  };

  const isTerminalRun = (snapshot) => TERMINAL_RUN.has(effectiveRunStatus(snapshot));

  const statusMeta = (status) => (
    STATUS[String(status || "unavailable")] || {
      icon: "?", label: String(status || "未知状态"), tone: "muted"
    }
  );

  const stableNodeKey = (node) => (
    `${String(node && node.node_id || "unknown")}::${String(node && node.task_id || "main")}`
  );

  const groupKeyForNode = (node) => {
    const nodeId = String(node && node.node_id || "");
    const stage = String(node && node.stage || "");
    if (nodeId.startsWith("gps_")) return "gps";
    if (nodeId.startsWith("netinfer_")) return "netinfer";
    if (nodeId.startsWith("proximity_")) return "proximity";
    if (nodeId.startsWith("kg_")) return "kg";
    if (nodeId === "rank_candidates" || nodeId === "generate_run_report") return "final";
    if (["preparation", "gps", "netinfer", "proximity", "kg", "final"].includes(stage)) {
      return stage;
    }
    return "preparation";
  };

  const scienceModuleKeyForNode = (node) => {
    const nodeId = String(node && node.node_id || "");
    if (nodeId === "prepare_expression_inputs" || nodeId.startsWith("gps_")) return "gps";
    if (nodeId.startsWith("netinfer_")) return "netinfer";
    if (nodeId.startsWith("proximity_")) return "proximity";
    if (nodeId.startsWith("kg_")) return "kg";
    if (nodeId === "rank_candidates" || nodeId === "generate_run_report") return "final";
    return null;
  };

  const groupNodes = (nodes) => {
    const grouped = Object.fromEntries(GROUPS.map((group) => [group.key, []]));
    (Array.isArray(nodes) ? nodes : []).forEach((node) => {
      grouped[groupKeyForNode(node)].push(node);
    });
    return grouped;
  };

  const parseTime = (value) => {
    if (!value) return null;
    const decoded = Date.parse(value);
    return Number.isFinite(decoded) ? decoded : null;
  };

  const elapsedSeconds = (node, nowMs=Date.now()) => {
    const started = parseTime(node && node.started_at);
    if (started === null) return null;
    const finished = parseTime(node && node.finished_at);
    return Math.max(0, ((finished === null ? nowMs : finished) - started) / 1000);
  };

  const formatDuration = (seconds) => {
    const value = numberOrNull(seconds);
    if (value === null) return "尚未开始";
    const total = Math.max(0, Math.floor(value));
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const remainder = total % 60;
    if (hours) return `${hours} 小时 ${minutes} 分`;
    if (minutes) return `${minutes} 分 ${remainder} 秒`;
    return `${remainder} 秒`;
  };

  const progressLabel = (node, nowMs=Date.now()) => {
    const progress = numberOrNull(node && node.progress);
    const status = String(node && node.status || "pending");
    const elapsed = elapsedSeconds(node, nowMs);
    if (progress === null) {
      return status === "running"
        ? `不确定进度 · 已运行 ${formatDuration(elapsed)}`
        : "不确定进度";
    }
    const percent = Math.round(Math.min(1, Math.max(0, progress)) * 100);
    if (status === "running" && percent === 0) {
      return `真实进度 0% · 已运行 ${formatDuration(elapsed)}`;
    }
    return `真实进度 ${percent}%`;
  };

  const etaLabel = (eta) => {
    if (!eta || eta.status !== "estimated") return "暂不可估算";
    const lower = numberOrNull(eta.lower_seconds);
    const upper = numberOrNull(eta.upper_seconds);
    if (lower === null || upper === null) return "暂不可估算";
    if (lower === 0 && upper === 0) return "已结束";
    if (Math.round(lower) === Math.round(upper)) return `约 ${formatDuration(lower)}`;
    return `约 ${formatDuration(lower)}–${formatDuration(upper)}`;
  };

  const nodeEtaLabel = (node) => (
    TERMINAL_NODE.has(String(node && node.status || "")) ? "已结束" : "暂不可估算"
  );

  const currentNode = (snapshot) => {
    const nodes = Array.isArray(snapshot && snapshot.nodes) ? snapshot.nodes : [];
    for (const status of ["running", "queued", "ready"]) {
      const match = nodes.find((node) => node.status === status);
      if (match) return match;
    }
    return null;
  };

  const runProgressSummary = (snapshot, nowMs=Date.now()) => {
    const status = effectiveRunStatus(snapshot);
    if (!snapshot) {
      return {
        status,
        percent: null,
        progress: "暂无任务",
        elapsed: "尚未开始",
        eta: "暂不可估算",
        completeCount: 0,
        totalCount: 0,
        uncertainCount: 0
      };
    }
    const nodes = Array.isArray(snapshot.nodes) ? snapshot.nodes : [];
    if (!nodes.length) {
      return {
        status,
        percent: null,
        progress: status === "collecting" ? "任务尚未启动" : "尚未生成 DAG 节点",
        elapsed: "尚未开始",
        eta: etaLabel(snapshot.eta),
        completeCount: 0,
        totalCount: 0,
        uncertainCount: 0
      };
    }

    let knownUnits = 0;
    let uncertainCount = 0;
    const totalCount = nodes.length;
    const completeCount = nodes.filter((node) => (
      TERMINAL_NODE.has(String(node.status || ""))
    )).length;
    nodes.forEach((node) => {
      const nodeStatus = String(node.status || "");
      if (TERMINAL_NODE.has(nodeStatus)) {
        knownUnits += 1;
        return;
      }
      const progress = numberOrNull(node.progress);
      if (progress === null) {
        uncertainCount += 1;
      } else {
        knownUnits += Math.max(0, Math.min(1, progress));
      }
    });
    const percent = Math.round((knownUnits / totalCount) * 100);
    const progress = uncertainCount
      ? `已确认进度 ${percent}% · 节点 ${completeCount}/${totalCount} · ${uncertainCount} 个未返回百分比`
      : `真实进度 ${percent}% · 节点 ${completeCount}/${totalCount}`;

    const starts = nodes.map((node) => parseTime(node.started_at)).filter((value) => value !== null);
    const finishes = nodes.map((node) => parseTime(node.finished_at)).filter((value) => value !== null);
    const active = !isTerminalRun(snapshot);
    const elapsed = starts.length
      ? formatDuration(((active ? nowMs : Math.max(...finishes, ...starts)) - Math.min(...starts)) / 1000)
      : "尚未开始";

    return {
      status,
      percent,
      progress,
      elapsed,
      eta: etaLabel(snapshot.eta),
      completeCount,
      totalCount,
      uncertainCount
    };
  };

  const moduleStatus = (nodes) => {
    if (!nodes.length) return "unavailable";
    const statuses = nodes.map((node) => String(node.status || "pending"));
    if (statuses.includes("failed")) return "failed";
    if (statuses.includes("running")) return "running";
    if (statuses.includes("queued")) return "queued";
    if (statuses.includes("ready")) return "ready";
    if (statuses.includes("pending")) return "pending";
    if (statuses.includes("blocked")) return "blocked";
    if (statuses.includes("cancelled")) return "cancelled";
    if (statuses.every((status) => status === "skipped")) return "skipped";
    if (statuses.every((status) => ["succeeded", "cached", "skipped"].includes(status))) {
      return statuses.every((status) => status === "skipped") ? "skipped" : "succeeded";
    }
    return statuses[0] || "unavailable";
  };

  const moduleSummary = (nodes, nowMs=Date.now()) => {
    const items = Array.isArray(nodes) ? nodes : [];
    if (!items.length) {
      return {
        status: "unavailable",
        progress: "等待真实节点",
        elapsed: "尚未开始",
        eta: "暂不可估算",
        error: null,
        completeCount: 0,
        totalCount: 0
      };
    }
    const terminal = items.filter((node) => TERMINAL_NODE.has(String(node.status || "")));
    const running = items.filter((node) => node.status === "running");
    const realProgress = running
      .map((node) => numberOrNull(node.progress))
      .filter((value) => value !== null);
    let progress;
    if (terminal.length === items.length) {
      progress = `节点已结束 ${terminal.length}/${items.length}`;
    } else if (running.length && realProgress.length === running.length) {
      const values = realProgress.map((value) => Math.round(Math.max(0, Math.min(1, value)) * 100));
      const range = Math.min(...values) === Math.max(...values)
        ? `${values[0]}%`
        : `${Math.min(...values)}%–${Math.max(...values)}%`;
      progress = `运行节点真实进度 ${range} · 已结束 ${terminal.length}/${items.length}`;
    } else {
      progress = `不确定进度 · 已结束 ${terminal.length}/${items.length}`;
    }
    const starts = items.map((node) => parseTime(node.started_at)).filter((value) => value !== null);
    const active = items.some((node) => !TERMINAL_NODE.has(String(node.status || "")));
    const finishes = items
      .map((node) => parseTime(node.finished_at))
      .filter((value) => value !== null);
    const elapsed = starts.length
      ? formatDuration(((active ? nowMs : Math.max(...finishes, ...starts)) - Math.min(...starts)) / 1000)
      : "尚未开始";
    const failed = items.find((node) => ["failed", "blocked"].includes(node.status) && node.error);
    const error = failed && (
      failed.error.message || failed.error.code || failed.error.exception_type
    );
    return {
      status: moduleStatus(items),
      progress,
      elapsed,
      eta: active ? "暂不可估算" : "已结束",
      error: error ? String(error) : null,
      completeCount: terminal.length,
      totalCount: items.length
    };
  };

  const canCancel = (snapshot) => {
    const status = effectiveRunStatus(snapshot);
    return status === "queued" || status === "running";
  };

  const safeErrorMessage = (error) => {
    if (!error) return "未知错误";
    if (typeof error === "string") return error;
    if (typeof error.message === "string" && error.message) return error.message;
    return String(error);
  };

  const createNdjsonParser = (onEvent) => {
    let buffer = "";
    let ended = false;
    const parseLine = (line) => {
      const trimmed = line.trim();
      if (!trimmed) return;
      let event;
      try {
        event = JSON.parse(trimmed);
      } catch (error) {
        throw new Error(`无法解析流式事件：${safeErrorMessage(error)}`);
      }
      onEvent(event);
    };
    const push = (chunk) => {
      if (ended) throw new Error("流式解析器已经结束");
      buffer += String(chunk || "");
      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop() || "";
      lines.forEach(parseLine);
    };
    const finish = () => {
      if (ended) return;
      ended = true;
      if (buffer.trim()) parseLine(buffer);
      buffer = "";
    };
    return {push, finish};
  };

  const createPollController = ({
    fetchSnapshot,
    onSnapshot,
    onError=() => {},
    setIntervalFn=(callback, delay) => globalThis.setInterval(callback, delay),
    clearIntervalFn=(timer) => globalThis.clearInterval(timer),
    intervalMs=4000
  }) => {
    let runId = null;
    let generation = 0;
    let requestSequence = 0;
    let activeRequest = null;
    let timer = null;

    const clearTimer = () => {
      if (timer !== null) clearIntervalFn(timer);
      timer = null;
    };

    const ensureTimer = () => {
      if (timer === null && runId) timer = setIntervalFn(refresh, intervalMs);
    };

    const refresh = async () => {
      if (!runId || activeRequest !== null) return null;
      const requestedRun = runId;
      const requestedGeneration = generation;
      const requestToken = `${requestedGeneration}:${++requestSequence}`;
      activeRequest = requestToken;
      try {
        const snapshot = await fetchSnapshot(requestedRun);
        if (requestedGeneration !== generation || requestedRun !== runId) return null;
        onSnapshot(snapshot, requestedRun);
        if (isTerminalRun(snapshot)) clearTimer();
        else ensureTimer();
        return snapshot;
      } catch (error) {
        if (requestedGeneration === generation && requestedRun === runId) {
          onError(error, requestedRun);
          ensureTimer();
        }
        return null;
      } finally {
        if (activeRequest === requestToken) activeRequest = null;
      }
    };

    const setRun = (nextRunId) => {
      generation += 1;
      runId = nextRunId ? String(nextRunId) : null;
      activeRequest = null;
      clearTimer();
      return runId ? refresh() : Promise.resolve(null);
    };

    const dispose = () => {
      generation += 1;
      runId = null;
      activeRequest = null;
      clearTimer();
    };

    const debugState = () => ({
      runId,
      generation,
      requestActive: activeRequest !== null,
      timerActive: timer !== null
    });

    return {setRun, refresh, dispose, debugState};
  };

  const createCancelController = ({
    getContext,
    confirmCancel,
    requestCancel,
    onStart=() => {},
    onSuccess=() => {},
    onError=() => {}
  }) => {
    let busyRun = null;

    const cancel = async () => {
      const initial = getContext();
      const targetRun = initial && initial.runId ? String(initial.runId) : null;
      if (!targetRun || busyRun !== null || !canCancel(initial.snapshot)) return false;
      busyRun = targetRun;
      try {
        const confirmed = await confirmCancel(initial);
        const current = getContext();
        if (
          !confirmed ||
          !current ||
          String(current.runId || "") !== targetRun ||
          !canCancel(current.snapshot)
        ) {
          return false;
        }
        onStart(targetRun);
        try {
          const snapshot = await requestCancel(targetRun);
          const after = getContext();
          if (after && String(after.runId || "") === targetRun) {
            onSuccess(snapshot, targetRun);
          }
          return true;
        } catch (error) {
          const after = getContext();
          if (after && String(after.runId || "") === targetRun) {
            onError(error, targetRun);
          }
          return false;
        }
      } finally {
        if (busyRun === targetRun) busyRun = null;
      }
    };

    return {
      cancel,
      isBusy: () => busyRun !== null,
      busyRun: () => busyRun
    };
  };

  const api = Object.freeze({
    GROUPS,
    STATUS,
    TERMINAL_NODE,
    canCancel,
    createCancelController,
    createNdjsonParser,
    createPollController,
    currentNode,
    effectiveRunStatus,
    elapsedSeconds,
    etaLabel,
    formatDuration,
    groupKeyForNode,
    groupNodes,
    isTerminalRun,
    moduleStatus,
    moduleSummary,
    nodeEtaLabel,
    progressLabel,
    runProgressSummary,
    safeErrorMessage,
    scienceModuleKeyForNode,
    stableNodeKey,
    statusMeta
  });
  globalThis.LipidUiLogic = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})();
"""

__all__ = ["WEB_UI_LOGIC"]
