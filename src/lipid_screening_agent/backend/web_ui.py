"""Dependency-free browser UI for the guided V3 screening workflow."""

from .web_ui_logic import WEB_UI_LOGIC

_WEB_UI_HEAD = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>smi2phen 多角度候选药物筛选</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, "Microsoft YaHei", sans-serif;
      --ink: #17221c;
      --muted: #637168;
      --line: #dce7df;
      --green: #17623a;
      --green-soft: #eaf3ed;
      --blue: #255d7a;
      --blue-soft: #eaf3f8;
      --amber: #8a5708;
      --amber-soft: #fff6df;
      --red: #9a2e28;
      --red-soft: #fff0ef;
      --slate-soft: #f1f4f2;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: linear-gradient(145deg, #edf4ef 0, #f8faf9 45%, #eef3f0 100%);
      color: var(--ink);
    }
    main { width: min(1280px, calc(100% - 28px)); margin: 24px auto 36px; }
    header, .panel {
      background: rgba(255,255,255,.96);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: 0 10px 30px rgba(31, 66, 44, .05);
    }
    header { padding: 22px 24px; margin-bottom: 14px; }
    h1 { margin: 0 0 7px; font-size: clamp(21px, 3vw, 28px); }
    h2 { margin: 0; font-size: 17px; }
    h3 { margin: 0; font-size: 14px; }
    p { margin: 5px 0; color: var(--muted); font-size: 13px; line-height: 1.55; }
    .eyebrow {
      margin-bottom: 8px; color: var(--green); font-size: 11px; font-weight: 800;
      letter-spacing: .13em; text-transform: uppercase;
    }
    .key-row, .send-row, .panel-head, .meta, .quick-actions, .file-actions,
    .task-actions, .node-head, .node-facts {
      display: flex; gap: 9px; align-items: center;
    }
    .key-row { margin-top: 16px; }
    input, textarea, button { font: inherit; }
    input, textarea {
      width: 100%; border: 1px solid #c8d6cd; border-radius: 10px; padding: 11px 12px;
      background: #fff;
    }
    input:focus, textarea:focus { outline: 2px solid #9bcbb0; border-color: #438660; }
    button {
      border: 0; border-radius: 9px; padding: 10px 14px; cursor: pointer;
      background: var(--green); color: #fff; white-space: nowrap;
    }
    button.secondary { background: #e5eee8; color: #254832; }
    button.ghost { background: transparent; color: var(--green); border: 1px solid #b9d0c0; }
    button.danger { background: var(--red); }
    button:disabled { opacity: .5; cursor: not-allowed; }
    dialog {
      width: min(760px, calc(100% - 28px)); max-height: min(760px, calc(100vh - 36px));
      border: 1px solid var(--line); border-radius: 16px; padding: 0;
      box-shadow: 0 24px 70px rgba(23, 49, 33, .22);
    }
    dialog::backdrop { background: rgba(18, 32, 24, .42); }
    .dialog-body { padding: 18px; }
    .history-list { display: grid; gap: 9px; margin: 12px 0; }
    .history-item {
      display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 10px;
      padding: 11px; border: 1px solid var(--line); border-radius: 11px; background: #fbfdfb;
    }
    .history-title { font-weight: 800; }
    .history-meta { margin-top: 4px; color: var(--muted); font-size: 11px; overflow-wrap: anywhere; }
    .history-actions { display: flex; gap: 7px; align-items: center; }
    .history-actions button, .history-actions a {
      padding: 7px 9px; border-radius: 8px; font-size: 12px;
    }
    .history-actions a { border: 1px solid var(--green); font-weight: 700; }
    .panel { padding: 16px; min-width: 0; }
    .panel-head { justify-content: space-between; margin-bottom: 10px; }
    .badge, .status-pill {
      display: inline-flex; gap: 5px; align-items: center; padding: 4px 8px;
      border-radius: 999px; font-size: 11px; font-weight: 750;
    }
    .badge { background: var(--green-soft); color: var(--green); }
    .badge.optional, .status-muted { background: #f0f2f1; color: #66736b; }
    .badge.warn, .status-ready, .status-queued { background: var(--amber-soft); color: var(--amber); }
    .status-running { background: var(--blue-soft); color: var(--blue); }
    .status-success { background: var(--green-soft); color: var(--green); }
    .status-cancelled { background: #eceff1; color: #4e5961; }
    .status-danger { background: var(--red-soft); color: var(--red); }
    .meta { flex-wrap: wrap; margin: 9px 2px 5px; font-size: 12px; color: var(--muted); }
    .meta span { overflow-wrap: anywhere; }
    .task-panel { margin-bottom: 14px; display: grid; grid-template-columns: 1fr auto; gap: 14px; }
    .task-title { display: flex; align-items: center; gap: 9px; flex-wrap: wrap; }
    .task-grid {
      display: grid; grid-template-columns: repeat(5, minmax(120px, 1fr)); gap: 9px;
      margin-top: 12px;
    }
    .task-fact { padding: 9px 10px; background: var(--slate-soft); border-radius: 9px; }
    .task-fact span { display: block; color: var(--muted); font-size: 10px; margin-bottom: 3px; }
    .task-fact strong { display: block; font-size: 12px; overflow-wrap: anywhere; }
    .task-actions { align-self: center; flex-direction: column; min-width: 150px; }
    #cancelError { color: var(--red); font-size: 11px; max-width: 250px; text-align: right; }
    .module-panel { margin-bottom: 14px; }
    .module-note { max-width: 780px; }
    #moduleCards {
      display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px;
    }
    .module-card {
      border: 1px solid var(--line); border-radius: 12px; padding: 12px;
      background: #fbfdfb; min-width: 0;
    }
    .module-card h3 { font-size: 15px; margin-bottom: 7px; }
    .module-card .description { min-height: 80px; }
    .io-line { font-size: 11px; line-height: 1.45; margin-top: 5px; color: var(--muted); }
    .io-line strong { color: var(--ink); }
    .module-state { margin-top: 10px; padding-top: 9px; border-top: 1px solid var(--line); }
    .module-state p { font-size: 11px; margin: 4px 0; }
    .module-error { color: var(--red) !important; overflow-wrap: anywhere; }
    .workspace {
      display: grid; grid-template-columns: minmax(0, 1.65fr) minmax(330px, 1fr);
      gap: 14px; align-items: start;
    }
    #messages { height: min(52vh, 500px); min-height: 330px; overflow-y: auto; padding: 8px 2px; }
    .chat-progress {
      margin: 3px 0 10px; padding: 10px 11px; border: 1px solid var(--line);
      border-radius: 10px; background: #fbfdfb;
    }
    .chat-progress-main, .chat-progress-meta {
      display: flex; gap: 9px; align-items: center; flex-wrap: wrap;
    }
    .chat-progress-main { justify-content: space-between; }
    .chat-progress-text { color: var(--ink); font-size: 12px; font-weight: 700; }
    .chat-progress-meta {
      margin-top: 7px; color: var(--muted); font-size: 11px;
    }
    .chat-progress-meta span { overflow-wrap: anywhere; }
    .chat-progress-track { margin-top: 9px; }
    .message {
      max-width: 86%; margin: 9px 0; padding: 11px 13px; border-radius: 13px;
      line-height: 1.6; white-space: pre-wrap; overflow-wrap: anywhere; font-size: 14px;
    }
    .user { margin-left: auto; background: var(--green); color: #fff; }
    .assistant { background: #edf3ef; }
    .error { background: var(--red-soft); color: var(--red); }
    .stream-stage-list {
      display: grid; gap: 3px; margin-bottom: 7px; color: var(--muted); font-size: 11px;
      white-space: normal;
    }
    .stream-stage { display: flex; gap: 5px; align-items: center; }
    .stream-stage::before { content: "•"; color: var(--green); }
    .stream-content:empty { display: none; }
    textarea { min-height: 84px; resize: vertical; }
    .send-row { align-items: stretch; }
    .chat-upload-guide {
      margin: 9px 0; padding: 11px; border: 1px solid #cfe0d4; border-radius: 11px;
      background: #f6faf7;
    }
    .chat-upload-guide h3 { margin-bottom: 5px; }
    .chat-upload-guide .source { color: var(--ink); }
    .chat-upload-actions { display: flex; gap: 8px; align-items: stretch; margin-top: 8px; }
    .chat-upload-actions input { min-width: 0; padding: 7px; font-size: 12px; }
    .quick-actions { flex-wrap: wrap; padding: 6px 0 12px; }
    .quick-actions button { padding: 6px 9px; font-size: 12px; }
    .hint { font-size: 12px; margin-top: 7px; }
    #inputCards { display: grid; gap: 9px; }
    .input-card { border: 1px solid var(--line); border-radius: 12px; padding: 11px; background: #fbfdfb; }
    .input-title { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
    .format { min-height: 39px; }
    .examples { display: flex; gap: 12px; margin: 7px 0; font-size: 12px; }
    a { color: var(--green); text-decoration: none; }
    a:hover { text-decoration: underline; }
    .file-actions { align-items: stretch; }
    .file-actions input { min-width: 0; padding: 7px; font-size: 12px; }
    .upload-status {
      margin-top: 7px; padding: 7px 8px; border-radius: 8px; background: #f1f4f2;
      color: var(--muted); font-size: 11px; overflow-wrap: anywhere;
    }
    .upload-status.ok { background: var(--green-soft); color: var(--green); }
    .upload-status.bad { background: var(--red-soft); color: var(--red); }
    .expression-pairs { display: grid; gap: 9px; margin-bottom: 9px; }
    .expression-pair-row {
      display: grid; grid-template-columns: 1fr 1fr auto; gap: 8px; align-items: end;
      margin-top: 8px;
    }
    .expression-file label {
      display: block; margin-bottom: 4px; font-size: 11px; color: var(--muted);
    }
    .expression-file input { padding: 7px; font-size: 12px; }
    .expression-toolbar { display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0 2px; }
    .section-note {
      margin: 9px 0 12px; padding: 9px 10px; border-radius: 10px;
      background: var(--amber-soft); color: #74501a; font-size: 12px; line-height: 1.5;
    }
    .results-panel { margin-top: 14px; }
    #artifactList { display: grid; gap: 8px; margin-top: 10px; }
    .artifact {
      display: flex; justify-content: space-between; align-items: center; gap: 10px;
      padding: 9px 10px; border-radius: 10px; background: #f1f5f2;
      border: 1px solid #dce9df; font-size: 12px;
    }
    .artifact-main { min-width: 0; }
    .artifact-title { font-weight: 800; color: var(--green); }
    .artifact-path { margin-top: 3px; color: var(--muted); overflow-wrap: anywhere; }
    .artifact-download {
      white-space: nowrap; padding: 6px 9px; border-radius: 999px;
      background: #fff; border: 1px solid var(--green); font-weight: 800;
    }
    .candidate-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 10px;
      margin-top: 10px;
    }
    .candidate-card {
      display: grid; grid-template-columns: 122px minmax(0, 1fr); gap: 10px;
      min-width: 0; padding: 10px; border: 1px solid var(--line); border-radius: 12px;
      background: #fbfdfb;
    }
    .candidate-structure {
      width: 122px; height: 104px; object-fit: contain; border-radius: 9px;
      background: #fff; border: 1px solid #e4ece6;
    }
    .candidate-structure-fallback {
      display: grid; place-items: center; width: 122px; height: 104px; padding: 8px;
      border-radius: 9px; background: var(--slate-soft); color: var(--muted);
      text-align: center; font-size: 11px;
    }
    .candidate-name { font-weight: 800; overflow-wrap: anywhere; }
    .candidate-id { color: var(--muted); font-size: 10px; overflow-wrap: anywhere; }
    .candidate-chips { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 7px; }
    .candidate-chip {
      padding: 3px 6px; border-radius: 999px; background: var(--green-soft);
      color: var(--green); font-size: 10px;
    }
    .candidate-properties {
      display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 4px 8px;
      margin-top: 7px; font-size: 10px; color: var(--muted);
    }
    .candidate-smiles {
      margin-top: 7px; color: #34463b; font: 10px/1.4 Consolas, monospace;
      overflow-wrap: anywhere;
    }
    .candidate-more { display: flex; justify-content: center; margin-top: 11px; }
    .candidate-more button { min-width: 180px; }
    .empty { color: var(--muted); font-size: 12px; }
    .dag-panel { margin-top: 14px; }
    #dagGroups { display: grid; gap: 14px; }
    .dag-group { position: relative; padding-left: 22px; }
    .dag-group::before {
      content: ""; position: absolute; left: 7px; top: 26px; bottom: 2px;
      width: 2px; background: var(--line);
    }
    .group-title { display: flex; gap: 8px; align-items: center; margin-bottom: 7px; }
    .group-count { font-size: 11px; color: var(--muted); }
    .node-card {
      position: relative; border: 1px solid var(--line); border-radius: 11px;
      background: #fbfdfb; padding: 10px 11px; margin: 8px 0;
    }
    .node-card::before {
      content: attr(data-icon); position: absolute; left: -22px; top: 11px; width: 16px;
      height: 16px; display: grid; place-items: center; border-radius: 50%;
      background: #fff; border: 1px solid var(--line); font-size: 10px; font-weight: 800;
    }
    .node-card.tone-running { border-left: 4px solid var(--blue); }
    .node-card.tone-success { border-left: 4px solid var(--green); }
    .node-card.tone-danger { border-left: 4px solid var(--red); }
    .node-card.tone-ready, .node-card.tone-queued { border-left: 4px solid var(--amber); }
    .node-card.tone-cancelled, .node-card.tone-muted { border-left: 4px solid #8a9890; }
    .node-head { justify-content: space-between; align-items: flex-start; }
    .node-name { font-weight: 750; font-size: 13px; }
    .node-key { color: var(--muted); font-size: 10px; overflow-wrap: anywhere; margin-top: 2px; }
    .node-facts { align-items: stretch; margin-top: 8px; flex-wrap: wrap; }
    .node-fact { flex: 1 1 145px; background: var(--slate-soft); border-radius: 8px; padding: 7px 8px; }
    .node-fact span { display: block; color: var(--muted); font-size: 9px; margin-bottom: 2px; }
    .node-fact strong { display: block; font-size: 11px; font-weight: 650; }
    .progress-track {
      height: 5px; margin-top: 8px; overflow: hidden; border-radius: 99px; background: #e4e9e6;
    }
    .progress-value { height: 100%; background: var(--green); border-radius: inherit; }
    .progress-track.indeterminate {
      background: repeating-linear-gradient(135deg, #e4e9e6 0 7px, #f7f9f8 7px 14px);
    }
    .node-error {
      margin-top: 8px; padding: 7px 8px; border-radius: 8px;
      background: var(--red-soft); color: var(--red); font-size: 11px; overflow-wrap: anywhere;
    }
    details { margin-top: 8px; }
    summary { cursor: pointer; color: var(--green); font-size: 11px; }
    pre {
      margin: 7px 0 0; padding: 8px; border-radius: 8px; background: #f0f3f1;
      white-space: pre-wrap; overflow-wrap: anywhere; font: 10px/1.45 Consolas, monospace;
    }
    @media (max-width: 1120px) {
      #moduleCards { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .task-grid { grid-template-columns: repeat(3, minmax(120px, 1fr)); }
    }
    @media (max-width: 900px) {
      .workspace { grid-template-columns: 1fr; }
      #messages { height: 430px; }
      #moduleCards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .task-panel { grid-template-columns: 1fr; }
      .task-actions { align-items: stretch; }
      #cancelError { text-align: left; max-width: none; }
      .candidate-grid { grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }
    }
    @media (max-width: 640px) {
      .key-row, .send-row, .file-actions, .chat-upload-actions { flex-direction: column; }
      .expression-pair-row { grid-template-columns: 1fr; }
      .chat-progress-main { align-items: flex-start; }
      .message { max-width: 95%; }
      main { width: min(100% - 18px, 1280px); margin-top: 10px; }
      #moduleCards, .task-grid { grid-template-columns: 1fr; }
      .module-card .description { min-height: auto; }
      .history-item { grid-template-columns: 1fr; }
      .candidate-card { grid-template-columns: 96px minmax(0, 1fr); }
      .candidate-structure, .candidate-structure-fallback { width: 96px; height: 88px; }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { scroll-behavior: auto !important; transition: none !important; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">Deterministic screening workflow</div>
    <h1>smi2phen 多角度候选药物筛选</h1>
    <p>Agent 负责引导输入、解释固定 DAG 和获得启动确认；GPS、NetInfer、Proximity 与 KG 由后端确定性执行。</p>
    <div class="key-row">
      <input id="apiKey" type="password" autocomplete="off" spellcheck="false" placeholder="DeepSeek API Key（仅随当前聊天请求发送）">
      <button id="toggleKey" class="secondary" type="button">显示</button>
      <button id="history" class="secondary" type="button">历史记录</button>
      <button id="newSession" class="secondary" type="button">新会话</button>
    </div>
    <p class="hint">Key 不写入浏览器存储或服务端数据库；远程部署时应使用 HTTPS。</p>
    <div class="meta">
      <span id="health">正在检查服务…</span>
      <span id="session">正在创建会话…</span>
      <span id="run"></span>
      <span id="runStatus"></span>
    </div>
  </header>

  <dialog id="historyDialog" aria-labelledby="historyHeading">
    <div class="dialog-body">
      <div class="panel-head">
        <div>
          <h2 id="historyHeading">历史任务与对话</h2>
          <p>打开后可继续查看本地任务；下载包包含对话、任务状态、结果清单和最终 artifacts。</p>
        </div>
        <button id="closeHistory" class="secondary" type="button">关闭</button>
      </div>
      <div id="historyList" class="history-list">
        <div class="empty">正在读取本地历史…</div>
      </div>
    </div>
  </dialog>

  <section class="panel task-panel" aria-labelledby="taskHeading">
    <div>
      <div class="task-title">
        <h2 id="taskHeading">当前任务</h2>
        <span id="taskStatus" class="status-pill status-muted">○ 暂无任务</span>
      </div>
      <div class="task-grid">
        <div class="task-fact"><span>Run ID</span><strong id="taskRunId">暂无任务</strong></div>
        <div class="task-fact"><span>运行模式</span><strong id="taskMode">—</strong></div>
        <div class="task-fact"><span>证据模式</span><strong id="taskEvidence">—</strong></div>
        <div class="task-fact"><span>当前节点</span><strong id="taskNode">—</strong></div>
        <div class="task-fact"><span>Run ETA</span><strong id="taskEta">暂不可估算</strong></div>
      </div>
    </div>
    <div class="task-actions">
      <button id="cancelRun" class="danger" type="button" disabled>终止当前任务</button>
      <div id="cancelError" role="alert"></div>
    </div>
  </section>

  <section class="panel module-panel" aria-labelledby="moduleHeading">
    <div class="panel-head">
      <div>
        <h2 id="moduleHeading">筛选模块与真实状态</h2>
        <p class="module-note">综合排序不是第五个筛选模块；它只汇总满足固定阈值且位于可用证据交集中的候选，不自动放宽阈值。</p>
      </div>
      <button id="refresh" class="ghost" type="button">刷新任务</button>
    </div>
    <div id="moduleCards"></div>
  </section>

  <div class="workspace">
    <section class="panel">
      <div class="panel-head"><h2>Agent 对话</h2></div>
      <div id="chatProgress" class="chat-progress" aria-live="polite">
        <div class="chat-progress-main">
          <span id="chatProgressStatus" class="status-pill status-muted">○ 暂无任务</span>
          <span id="chatProgressText" class="chat-progress-text">暂无任务</span>
        </div>
        <div class="chat-progress-meta">
          <span>已运行：<strong id="chatRunElapsed">尚未开始</strong></span>
          <span>当前节点：<strong id="chatRunNode">—</strong></span>
          <span>ETA：<strong id="chatRunEta">暂不可估算</strong></span>
        </div>
        <div id="chatProgressTrack" class="progress-track" aria-label="当前任务整体进度">
          <div id="chatProgressValue" class="progress-value" style="width:0%"></div>
        </div>
      </div>
      <div class="quick-actions">
        <button class="secondary quick" data-message="请为肝脂肪变性创建一个筛选任务。">创建示例任务</button>
        <button class="secondary quick" data-message="请检查当前输入并预览执行计划。">检查并预览</button>
        <button class="secondary quick" data-message="我已检查计划，确认启动工作流。">确认启动</button>
        <button class="secondary quick" data-message="请查询当前运行进度。">查询进度</button>
        <button class="secondary quick" data-message="请查看结果并解释候选分子的计算排名理由。">解释结果</button>
      </div>
      <div id="messages" aria-live="polite"></div>
      <div id="chatUploadGuide" class="chat-upload-guide">
        <p>创建任务后，Agent 会在这里提示并接收当前缺少的文件。</p>
      </div>
      <div class="send-row">
        <textarea id="message" placeholder="先告诉 Agent 要研究的疾病，例如：为肝脂肪变性创建筛选任务。Ctrl+Enter 发送。"></textarea>
        <button id="send" type="button">发送</button>
      </div>
    </section>

    <aside class="panel">
      <div class="panel-head">
        <h2>输入文件</h2>
        <span id="inputReadiness" class="badge warn">等待任务</span>
      </div>
      <div id="inputNote" class="section-note">正在加载格式说明…</div>
      <div id="inputCards"></div>
      <div id="expressionPairSection" class="expression-pairs"></div>
    </aside>
  </div>

  <section id="results" class="panel results-panel" aria-labelledby="resultsHeading">
    <div class="panel-head">
      <h2 id="resultsHeading">结果与下载</h2>
      <span id="resultStatus" class="badge optional">尚未完成</span>
    </div>
    <div id="candidatePreview" class="empty">工作流成功后显示候选预览。</div>
    <div id="artifactHint" class="empty">成功后可在这里下载完整候选排名表、排名汇总和运行报告。</div>
    <div id="artifactList"></div>
  </section>

  <section class="panel dag-panel" aria-labelledby="dagHeading">
    <div class="panel-head">
      <div>
        <h2 id="dagHeading">真实 DAG 节点</h2>
        <p>节点完全来自 GET /runs/{run_id}；动态 batch 与 seed 节点会在下一次快照中加入。</p>
      </div>
      <span id="dagSummary" class="badge optional">暂无节点</span>
    </div>
    <div id="dagEmpty" class="empty">创建并预览任务后显示真实节点。</div>
    <div id="dagGroups"></div>
  </section>
</main>
<script>
"""

_WEB_UI_APP = r"""
</script>
<script>
(() => {
  "use strict";
  const logic = globalThis.LipidUiLogic;
  let threadId = null;
  let runId = null;
  let runSnapshot = null;
  let modelConfigured = false;
  let inputSpecs = [];
  let expressionSpec = null;
  let expressionPairRows = ["1"];
  let uploadedByKey = {};
  let sessionEpoch = 0;
  let chatController = null;
  let resultsLoadedFor = null;
  let resultsLoadingFor = null;
  let cancelPendingRun = null;
  let displayedCandidates = [];
  let displayedCandidateTotal = 0;
  let candidateVisibleCount = 12;

  const MODULES = [
    {
      key: "gps",
      name: "GPS",
      description: "根据 TPM 和 metadata 枚举疾病差异表达特征，与预测的化合物扰动表达特征计算表达逆转分数；当前配置中分数越低越优。",
      input: "TPM、metadata、标准化化合物",
      output: "GPS 表达逆转分数"
    },
    {
      key: "netinfer",
      name: "NetInfer",
      description: "基于药物—子结构—靶点网络进行加权网络推断，生成已知和预测靶点；它不是深度学习模型。",
      input: "标准化化合物、药物—子结构—靶点网络",
      output: "已知与预测靶点及证据"
    },
    {
      key: "proximity",
      name: "Proximity",
      description: "在人类 PPI 网络中计算化合物靶点集合与疾病基因集合的网络邻近性；当前配置中 Z 值越低越优，无统计检验时不称为显著。",
      input: "化合物靶点、疾病基因、人类 PPI",
      output: "网络邻近性 Z 值"
    },
    {
      key: "kg",
      name: "KG",
      description: "完成知识图谱数据准备、预训练、种子微调和种子聚合，生成 KG 排名。",
      input: "化合物、疾病基因、靶点与 KG 基础资源",
      output: "多种子聚合的 KG 排名"
    },
    {
      key: "final",
      name: "综合排序",
      description: "对满足阈值且位于可用证据交集中的候选计算证据内百分位均值；不自动放宽阈值。",
      input: "可用的 KG、Proximity 与 GPS 证据交集",
      output: "综合候选排序与运行报告"
    }
  ];
  const NODE_NAMES = {
    create_run_workspace: "创建 Run 工作区",
    register_inputs: "登记输入文件",
    prepare_compound_library: "准备化合物库",
    prepare_disease_genes: "准备疾病基因",
    prepare_expression_inputs: "准备表达数据",
    import_drug_targets: "导入外部靶点",
    gps_predict_drug_profiles: "预测化合物扰动表达特征",
    gps_build_disease_signature: "构建疾病差异表达特征",
    gps_score_compounds: "计算 GPS 表达逆转分数",
    netinfer_prepare_inputs: "准备 NetInfer 输入",
    netinfer_predict_known: "推断已知药物靶点",
    netinfer_predict_batch: "推断化合物批次靶点",
    netinfer_merge_targets: "合并已知与预测靶点",
    proximity_prepare_network: "准备人类 PPI 邻近性网络",
    proximity_score_compounds: "计算网络邻近性",
    kg_construct_graph: "准备知识图谱数据",
    kg_prepare_training_data: "准备 KG 训练数据",
    kg_pretrain: "KG 预训练",
    kg_finetune_seed: "KG 种子微调",
    kg_aggregate_seeds: "聚合 KG 种子排名",
    rank_candidates: "计算综合候选排序",
    generate_run_report: "生成运行报告"
  };
  const $ = (id) => document.getElementById(id);
  const nodeName = (node) => {
    const base = NODE_NAMES[node.node_id] || node.node_id;
    return node.task_id && node.task_id !== "main" ? `${base} · ${node.task_id}` : base;
  };
  const plainAssistantText = (value) => {
    const lines = String(value || "").split(/\r?\n/);
    const output = [];
    lines.forEach((line) => {
      if (/^\s*[-|: ]{3,}\s*$/.test(line)) return;
      let text = line
        .replace(/^\s{0,3}#{1,6}\s+/, "")
        .replace(/\*\*/g, "")
        .replace(/`/g, "")
        .trimEnd();
      if (/^\s*\|.*\|\s*$/.test(text)) {
        const cells = text.split("|").slice(1, -1).map((cell) => cell.trim());
        if (cells.every((cell) => /^:?-{3,}:?$/.test(cell))) return;
        text = cells.length === 2 ? `${cells[0]}：${cells[1]}` : cells.join(" · ");
      }
      output.push(text);
    });
    return output.join("\n").replace(/\n{3,}/g, "\n\n").trim();
  };
  const addMessage = (text, role) => {
    const node = document.createElement("div");
    node.className = `message ${role}`;
    node.textContent = role === "assistant" ? plainAssistantText(text) : text;
    $("messages").appendChild(node);
    node.scrollIntoView({block: "end"});
  };
  const addStreamingAssistant = () => {
    const node = document.createElement("div");
    node.className = "message assistant";
    node.setAttribute("aria-busy", "true");
    const stages = document.createElement("div");
    stages.className = "stream-stage-list";
    const content = document.createElement("div");
    content.className = "stream-content";
    node.append(stages, content);
    $("messages").appendChild(node);
    node.scrollIntoView({block: "end"});
    return {node, stages, content, finalSeen: false, errorSeen: false};
  };
  const appendStreamStage = (view, text) => {
    const line = document.createElement("div");
    line.className = "stream-stage";
    line.textContent = text;
    view.stages.appendChild(line);
    view.node.scrollIntoView({block: "end"});
  };
  const applyStreamEvent = (view, event) => {
    if (event.type === "stage") {
      if (event.stage === "agent" && event.status === "started") {
        appendStreamStage(view, "Agent 已接收请求");
      } else if (event.stage === "model") {
        const action = event.status === "started" ? "处理中" : "已完成";
        appendStreamStage(view, `模型第 ${event.round || 1} 轮${action}`);
      }
      return;
    }
    if (event.type === "tool") {
      const labels = {started: "开始", succeeded: "完成", failed: "失败"};
      appendStreamStage(view, `工具 ${event.name || "unknown"} ${labels[event.status] || event.status}`);
      return;
    }
    if (event.type === "final") {
      view.content.textContent = plainAssistantText(event.content);
      view.finalSeen = true;
      view.node.setAttribute("aria-busy", "false");
      view.node.scrollIntoView({block: "end"});
      return;
    }
    if (event.type === "error") {
      view.content.textContent = `${event.message || "流式请求失败"}\n请重试。`;
      view.content.classList.add("error");
      view.errorSeen = true;
      view.node.setAttribute("aria-busy", "false");
      view.node.scrollIntoView({block: "end"});
    }
  };
  const errorText = async (response) => {
    try {
      const body = await response.json();
      return typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail || body);
    } catch (_) {
      return `${response.status} ${response.statusText}`;
    }
  };
  const formatBytes = (value) => {
    const bytes = Number(value || 0);
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  };
  const applyStatus = (element, status) => {
    const meta = logic.statusMeta(status);
    element.className = `status-pill status-${meta.tone}`;
    element.textContent = `${meta.icon} ${meta.label}`;
  };

  const renderChatRunProgress = () => {
    const summary = logic.runProgressSummary(runSnapshot);
    applyStatus($("chatProgressStatus"), summary.status);
    $("chatProgressText").textContent = summary.progress;
    $("chatRunElapsed").textContent = summary.elapsed;
    const activeNode = logic.currentNode(runSnapshot);
    $("chatRunNode").textContent = activeNode ? nodeName(activeNode) : "—";
    $("chatRunEta").textContent = summary.eta;
    const track = $("chatProgressTrack");
    const value = $("chatProgressValue");
    if (summary.percent === null) {
      track.className = runSnapshot ? "progress-track indeterminate chat-progress-track" : "progress-track chat-progress-track";
      value.style.width = "0%";
      track.setAttribute("aria-label", summary.progress);
      return;
    }
    track.className = summary.uncertainCount
      ? "progress-track indeterminate chat-progress-track"
      : "progress-track chat-progress-track";
    value.style.width = `${Math.max(0, Math.min(100, summary.percent))}%`;
    track.setAttribute("aria-label", summary.progress);
  };

  const renderTaskControl = () => {
    const status = logic.effectiveRunStatus(runSnapshot);
    applyStatus($("taskStatus"), status);
    $("taskRunId").textContent = runId || "暂无任务";
    $("taskMode").textContent = runSnapshot && runSnapshot.mode || "—";
    $("taskEvidence").textContent = runSnapshot && runSnapshot.evidence_mode || "—";
    const activeNode = logic.currentNode(runSnapshot);
    $("taskNode").textContent = activeNode ? nodeName(activeNode) : "—";
    $("taskEta").textContent = logic.etaLabel(runSnapshot && runSnapshot.eta);
    $("run").textContent = runId ? `任务：${runId}` : "";
    $("runStatus").textContent = runSnapshot ? `状态：${logic.statusMeta(status).label}` : "";
    const cancelButton = $("cancelRun");
    const pending = cancelPendingRun === runId || status === "cancelling";
    cancelButton.disabled = pending || !logic.canCancel(runSnapshot);
    cancelButton.textContent = pending ? "正在终止" : "终止当前任务";
    renderChatRunProgress();
  };

  const renderModuleCards = () => {
    const root = $("moduleCards");
    root.replaceChildren();
    const nodes = Array.isArray(runSnapshot && runSnapshot.nodes) ? runSnapshot.nodes : [];
    MODULES.forEach((definition) => {
      const moduleNodes = nodes.filter(
        (node) => logic.scienceModuleKeyForNode(node) === definition.key
      );
      const summary = logic.moduleSummary(moduleNodes);
      const meta = logic.statusMeta(summary.status);
      const card = document.createElement("article");
      card.className = "module-card";
      card.dataset.module = definition.key;
      const title = document.createElement("div");
      title.className = "panel-head";
      const heading = document.createElement("h3");
      heading.textContent = definition.name;
      const status = document.createElement("span");
      status.className = `status-pill status-${meta.tone}`;
      status.textContent = `${meta.icon} ${meta.label}`;
      title.append(heading, status);
      const description = document.createElement("p");
      description.className = "description";
      description.textContent = definition.description;
      const input = document.createElement("div");
      input.className = "io-line";
      const inputStrong = document.createElement("strong");
      inputStrong.textContent = "主要输入：";
      input.append(inputStrong, document.createTextNode(definition.input));
      const output = document.createElement("div");
      output.className = "io-line";
      const outputStrong = document.createElement("strong");
      outputStrong.textContent = "主要输出：";
      output.append(outputStrong, document.createTextNode(definition.output));
      const state = document.createElement("div");
      state.className = "module-state";
      [
        `进度：${summary.progress}`,
        `已用时间：${summary.elapsed}`,
        `ETA：${summary.eta}`
      ].forEach((value) => {
        const row = document.createElement("p");
        row.textContent = value;
        state.appendChild(row);
      });
      if (summary.error) {
        const error = document.createElement("p");
        error.className = "module-error";
        error.textContent = `错误：${summary.error}`;
        state.appendChild(error);
      }
      card.append(title, description, input, output, state);
      root.appendChild(card);
    });
  };

  const phaseLabel = (node) => {
    if (node.status === "running") return `正在执行：${nodeName(node)}`;
    if (node.status === "queued") return `等待 ${node.queue || node.resource_class || "Worker"} 队列`;
    if (node.status === "ready") return "依赖已满足，等待入队";
    if (node.status === "pending") return "等待上游依赖";
    if (node.status === "skipped") return "该路径未启用";
    return `节点已进入${logic.statusMeta(node.status).label}状态`;
  };

  const detailPayload = (node) => ({
    dependencies: node.dependencies || [],
    queue: node.queue || node.resource_class || null,
    attempt: node.attempt,
    heartbeat_at: node.heartbeat_at,
    parameters: node.parameters || {},
    artifacts: (node.artifacts || []).map((artifact) => ({
      artifact_id: artifact.artifact_id,
      artifact_type: artifact.artifact_type,
      relative_path: artifact.relative_path,
      sha256: artifact.sha256
    })),
    metrics: node.metrics || {},
    warnings: node.warnings || [],
    error: node.error || null
  });

  const renderDag = () => {
    const root = $("dagGroups");
    root.replaceChildren();
    const nodes = Array.isArray(runSnapshot && runSnapshot.nodes) ? runSnapshot.nodes : [];
    $("dagEmpty").hidden = nodes.length > 0;
    $("dagSummary").textContent = nodes.length ? `${nodes.length} 个真实节点` : "暂无节点";
    $("dagSummary").className = nodes.length ? "badge" : "badge optional";
    if (!nodes.length) return;
    const grouped = logic.groupNodes(nodes);
    logic.GROUPS.forEach((group) => {
      const items = grouped[group.key] || [];
      if (!items.length) return;
      const section = document.createElement("section");
      section.className = "dag-group";
      section.dataset.group = group.key;
      const title = document.createElement("div");
      title.className = "group-title";
      const heading = document.createElement("h3");
      heading.textContent = group.label;
      const count = document.createElement("span");
      count.className = "group-count";
      count.textContent = `${items.length} 个节点`;
      title.append(heading, count);
      section.appendChild(title);
      items.forEach((node) => {
        const meta = logic.statusMeta(node.status);
        const card = document.createElement("article");
        card.className = `node-card tone-${meta.tone}`;
        card.id = `node-${encodeURIComponent(logic.stableNodeKey(node))}`;
        card.dataset.nodeKey = logic.stableNodeKey(node);
        card.dataset.status = node.status;
        card.dataset.icon = meta.icon;
        const head = document.createElement("div");
        head.className = "node-head";
        const identity = document.createElement("div");
        const name = document.createElement("div");
        name.className = "node-name";
        name.textContent = nodeName(node);
        const key = document.createElement("div");
        key.className = "node-key";
        key.textContent = `${node.node_id} / ${node.task_id || "main"}`;
        identity.append(name, key);
        const status = document.createElement("span");
        status.className = `status-pill status-${meta.tone}`;
        status.textContent = `${meta.icon} ${meta.label}`;
        head.append(identity, status);
        const facts = document.createElement("div");
        facts.className = "node-facts";
        [
          ["进度", logic.progressLabel(node)],
          ["已用时间", logic.formatDuration(logic.elapsedSeconds(node))],
          ["ETA", logic.nodeEtaLabel(node)],
          ["当前阶段", phaseLabel(node)]
        ].forEach(([label, value]) => {
          const fact = document.createElement("div");
          fact.className = "node-fact";
          const caption = document.createElement("span");
          caption.textContent = label;
          const strong = document.createElement("strong");
          strong.textContent = value;
          fact.append(caption, strong);
          facts.appendChild(fact);
        });
        const progress = document.createElement("div");
        const numeric = node.progress === null || node.progress === undefined
          ? null
          : Math.max(0, Math.min(1, Number(node.progress)));
        progress.className = numeric === null ? "progress-track indeterminate" : "progress-track";
        progress.setAttribute("aria-label", logic.progressLabel(node));
        if (numeric !== null && Number.isFinite(numeric)) {
          const value = document.createElement("div");
          value.className = "progress-value";
          value.style.width = `${numeric * 100}%`;
          progress.appendChild(value);
        }
        card.append(head, facts, progress);
        if (node.error) {
          const error = document.createElement("div");
          error.className = "node-error";
          error.textContent = `错误：${node.error.message || node.error.code || "节点执行失败"}`;
          card.appendChild(error);
        }
        const details = document.createElement("details");
        const summary = document.createElement("summary");
        summary.textContent = "查看依赖、输入输出与错误详情";
        const pre = document.createElement("pre");
        pre.textContent = JSON.stringify(detailPayload(node), null, 2);
        details.append(summary, pre);
        card.appendChild(details);
        section.appendChild(card);
      });
      root.appendChild(section);
    });
  };

  const renderRun = (snapshot) => {
    if (!snapshot || String(snapshot.run_id || "") !== String(runId || "")) return;
    runSnapshot = snapshot;
    renderTaskControl();
    renderModuleCards();
    renderDag();
    if (snapshot.requirements) syncRequirements(snapshot.requirements);
    const status = logic.effectiveRunStatus(snapshot);
    if (status === "succeeded") {
      $("resultStatus").textContent = "已完成";
      $("resultStatus").className = "badge";
      fetchResults(runId);
    } else if (["failed", "blocked", "cancelled"].includes(status)) {
      $("resultStatus").textContent = logic.statusMeta(status).label;
      $("resultStatus").className = "badge warn";
    } else if (["queued", "running", "cancelling"].includes(status)) {
      $("resultStatus").textContent = status === "cancelling" ? "正在终止" : "运行中";
      $("resultStatus").className = "badge warn";
    }
  };

  const fetchRunSnapshot = async (targetRun) => {
    const response = await fetch(`/runs/${encodeURIComponent(targetRun)}`);
    if (!response.ok) throw new Error(await errorText(response));
    return response.json();
  };
  const poller = logic.createPollController({
    intervalMs: 4000,
    fetchSnapshot: fetchRunSnapshot,
    onSnapshot: (snapshot) => renderRun(snapshot),
    onError: (error, targetRun) => {
      if (targetRun === runId) $("runStatus").textContent = `状态刷新失败：${error.message}`;
    }
  });

  const resetRunUi = () => {
    runId = null;
    runSnapshot = null;
    uploadedByKey = {};
    resultsLoadedFor = null;
    resultsLoadingFor = null;
    cancelPendingRun = null;
    displayedCandidates = [];
    displayedCandidateTotal = 0;
    candidateVisibleCount = 12;
    $("cancelError").textContent = "";
    $("inputReadiness").textContent = "等待任务";
    $("inputReadiness").className = "badge warn";
    $("resultStatus").textContent = "尚未完成";
    $("resultStatus").className = "badge optional";
    $("candidatePreview").textContent = "工作流成功后显示候选预览。";
    $("artifactHint").textContent = "成功后可在这里下载完整候选排名表、排名汇总和运行报告。";
    $("artifactList").replaceChildren();
    $("chatUploadGuide").replaceChildren();
    const chatHint = document.createElement("p");
    chatHint.textContent = "创建任务后，Agent 会在这里提示并接收当前缺少的文件。";
    $("chatUploadGuide").appendChild(chatHint);
    renderInputStatuses();
    renderTaskControl();
    renderModuleCards();
    renderDag();
    poller.dispose();
  };

  const setRun = async (value) => {
    if (!value) return;
    const next = String(value);
    if (next === runId) {
      await poller.refresh();
      return;
    }
    runId = next;
    runSnapshot = null;
    resultsLoadedFor = null;
    resultsLoadingFor = null;
    cancelPendingRun = null;
    $("cancelError").textContent = "";
    renderTaskControl();
    renderModuleCards();
    renderDag();
    await poller.setRun(next);
  };

  const createSession = async () => {
    const epoch = ++sessionEpoch;
    if (chatController) chatController.abort();
    $("session").textContent = "正在创建会话…";
    const response = await fetch("/sessions", {
      method: "POST", headers: {"Content-Type": "application/json"}, body: "{}"
    });
    if (!response.ok) throw new Error(await errorText(response));
    const body = await response.json();
    if (epoch !== sessionEpoch) return;
    threadId = body.thread_id;
    $("session").textContent = `会话：${threadId}`;
    $("messages").replaceChildren();
    resetRunUi();
    await refreshHistory();
  };

  const loadSession = async (targetThread) => {
    const epoch = ++sessionEpoch;
    if (chatController) chatController.abort();
    poller.dispose();
    $("session").textContent = "正在恢复会话…";
    const response = await fetch(`/sessions/${encodeURIComponent(targetThread)}`);
    if (!response.ok) throw new Error(await errorText(response));
    const body = await response.json();
    if (epoch !== sessionEpoch) return;
    threadId = String(body.session.thread_id);
    $("session").textContent = `会话：${threadId}`;
    $("messages").replaceChildren();
    resetRunUi();
    (body.messages || []).forEach((message) => addMessage(message.content, message.role));
    if (!(body.messages || []).length) {
      addMessage("请先告诉我本次筛选关注的疾病。任务创建后可直接在对话区上传所需文件。", "assistant");
    }
    if (body.session.run_id) await setRun(body.session.run_id);
    if ($("historyDialog").open) $("historyDialog").close();
  };

  const historyStatusLabel = (status) => ({
    new: "未创建任务",
    collecting: "收集输入",
    previewed: "计划已预览",
    created: "等待确认",
    ready: "等待确认",
    queued: "已排队",
    running: "运行中",
    succeeded: "已完成",
    failed: "失败",
    cancelled: "已取消",
    blocked: "阻塞",
    unknown: "状态未知"
  }[status] || status || "状态未知");

  async function refreshHistory() {
    const list = $("historyList");
    const response = await fetch("/sessions?limit=100");
    if (!response.ok) throw new Error(await errorText(response));
    const body = await response.json();
    list.replaceChildren();
    const sessions = body.sessions || [];
    if (!sessions.length) {
      const empty = document.createElement("div");
      empty.className = "empty";
      empty.textContent = "尚无本地历史记录。";
      list.appendChild(empty);
      return sessions;
    }
    sessions.forEach((item) => {
      const row = document.createElement("article");
      row.className = "history-item";
      const main = document.createElement("div");
      const title = document.createElement("div");
      title.className = "history-title";
      title.textContent = item.disease && item.disease.name
        ? item.disease.name
        : "尚未创建筛选任务";
      const meta = document.createElement("div");
      meta.className = "history-meta";
      const updated = item.updated_at ? new Date(item.updated_at).toLocaleString() : "时间未知";
      meta.textContent = `${historyStatusLabel(item.status)} · ${item.message_count || 0} 条消息 · ${updated}`;
      const ids = document.createElement("div");
      ids.className = "history-meta";
      ids.textContent = item.run_id || item.thread_id;
      main.append(title, meta, ids);
      const actions = document.createElement("div");
      actions.className = "history-actions";
      const open = document.createElement("button");
      open.type = "button";
      open.textContent = item.thread_id === threadId ? "当前" : "打开";
      open.disabled = item.thread_id === threadId;
      open.addEventListener("click", () => loadSession(item.thread_id).catch(
        (error) => addMessage(`历史会话恢复失败：${error.message}`, "error")
      ));
      const download = document.createElement("a");
      download.href = `/sessions/${encodeURIComponent(item.thread_id)}/export`;
      download.textContent = "下载包";
      download.setAttribute("download", "");
      actions.append(open, download);
      row.append(main, actions);
      list.appendChild(row);
    });
    return sessions;
  }

  const bootstrapSession = async () => {
    const sessions = await refreshHistory();
    const selected = sessions[0];
    if (selected) {
      await loadSession(selected.thread_id);
      return;
    }
    await createSession();
    addMessage("请先告诉我本次筛选关注的疾病。任务创建后可直接在对话区上传所需文件。", "assistant");
  };

  const checkHealth = async () => {
    const response = await fetch("/healthz");
    if (!response.ok) throw new Error(await errorText(response));
    const body = await response.json();
    modelConfigured = Boolean(body.model_configured);
    $("health").textContent = `服务正常 · 模型：${body.model}`;
  };

  const makeLink = (label, url) => {
    const link = document.createElement("a");
    link.textContent = label;
    link.href = url;
    link.setAttribute("download", "");
    return link;
  };

  const artifactLabels = {
    final_candidates: "完整候选排名表",
    ranking_summary: "排名汇总 JSON",
    run_report_json: "完整运行报告 JSON",
    run_report_markdown: "完整运行报告 Markdown"
  };

  const fileNameFromPath = (path) => {
    if (!path) return "";
    const parts = String(path).split(/[\\/]/);
    return parts[parts.length - 1] || "";
  };

  const artifactLabel = (artifact) => {
    if (artifact.download_label) return String(artifact.download_label);
    const fileName = artifact.file_name || fileNameFromPath(artifact.relative_path);
    const base = artifactLabels[String(artifact.artifact_type || "")] || "结果文件";
    return fileName ? `${base}（${fileName}）` : base;
  };

  const artifactDownloadUrl = (artifact, targetRun) => {
    if (artifact.download_url) return String(artifact.download_url);
    return `/runs/${encodeURIComponent(targetRun)}/artifacts/${encodeURIComponent(artifact.artifact_id)}`;
  };

  const expressionKey = (role, pairId) => {
    const base = role === "tpm" ? "expression_tpm" : "expression_metadata";
    return `${base}:${pairId}`;
  };

  const expressionUpload = (role, pairId) => {
    const base = role === "tpm" ? "expression_tpm" : "expression_metadata";
    return uploadedByKey[expressionKey(role, pairId)] ||
      (pairId === "1" ? uploadedByKey[base] : null);
  };

  const knownExpressionPairIds = () => {
    const ids = new Set(expressionPairRows);
    Object.keys(uploadedByKey).forEach((key) => {
      if (key === "expression_tpm" || key === "expression_metadata") ids.add("1");
      if (key.startsWith("expression_tpm:") || key.startsWith("expression_metadata:")) {
        ids.add(key.split(":", 2)[1]);
      }
    });
    return [...ids].sort((left, right) => Number(left) - Number(right) || left.localeCompare(right));
  };

  const addExpressionPairRow = () => {
    const numeric = knownExpressionPairIds()
      .map((value) => Number(value))
      .filter((value) => Number.isInteger(value) && value > 0);
    const next = String((numeric.length ? Math.max(...numeric) : 0) + 1);
    if (!expressionPairRows.includes(next)) expressionPairRows.push(next);
    renderExpressionPairs();
  };

  const renderExpressionPairs = () => {
    const root = $("expressionPairSection");
    root.replaceChildren();
    if (!expressionSpec) return;
    const header = document.createElement("div");
    header.className = "input-title";
    const heading = document.createElement("h3");
    heading.textContent = expressionSpec.label;
    const badge = document.createElement("span");
    badge.className = "badge optional";
    badge.textContent = "可选多对";
    header.append(heading, badge);
    const note = document.createElement("p");
    note.className = "format";
    note.textContent = expressionSpec.format;
    const examples = document.createElement("div");
    examples.className = "examples";
    examples.append(
      makeLink("TPM 合规样例", expressionSpec.tpm_valid_example_url),
      makeLink("metadata 合规样例", expressionSpec.metadata_valid_example_url)
    );
    const toolbar = document.createElement("div");
    toolbar.className = "expression-toolbar";
    const add = document.createElement("button");
    add.type = "button";
    add.className = "secondary";
    add.textContent = "+ 增加表达对";
    add.addEventListener("click", addExpressionPairRow);
    toolbar.appendChild(add);
    const wrapper = document.createElement("article");
    wrapper.className = "input-card";
    wrapper.append(header, note, examples, toolbar);

    knownExpressionPairIds().forEach((pairId) => {
      const tpmUploaded = expressionUpload("tpm", pairId);
      const metadataUploaded = expressionUpload("metadata", pairId);
      const pair = document.createElement("div");
      pair.className = "expression-pair-row";
      pair.dataset.pairId = pairId;

      const tpmBox = document.createElement("div");
      tpmBox.className = "expression-file";
      const tpmLabel = document.createElement("label");
      tpmLabel.htmlFor = `file-expression-tpm-${pairId}`;
      tpmLabel.textContent = `TPM_matrix_${pairId}.tsv`;
      const tpmInput = document.createElement("input");
      tpmInput.type = "file";
      tpmInput.accept = expressionSpec.tpm_accept;
      tpmInput.id = `file-expression-tpm-${pairId}`;
      tpmBox.append(tpmLabel, tpmInput);

      const metadataBox = document.createElement("div");
      metadataBox.className = "expression-file";
      const metadataLabel = document.createElement("label");
      metadataLabel.htmlFor = `file-expression-metadata-${pairId}`;
      metadataLabel.textContent = `metadata_${pairId}.tsv`;
      const metadataInput = document.createElement("input");
      metadataInput.type = "file";
      metadataInput.accept = expressionSpec.metadata_accept;
      metadataInput.id = `file-expression-metadata-${pairId}`;
      metadataBox.append(metadataLabel, metadataInput);

      const upload = document.createElement("button");
      upload.type = "button";
      upload.textContent = tpmUploaded || metadataUploaded ? "替换该对" : "上传该对";
      upload.id = `upload-expression-pair-${pairId}`;
      upload.addEventListener("click", () => uploadExpressionPair(pairId));
      pair.append(tpmBox, metadataBox, upload);

      const status = document.createElement("div");
      status.className = "upload-status";
      status.id = `status-expression-pair-${pairId}`;
      wrapper.append(pair, status);
    });
    root.appendChild(wrapper);
    renderExpressionPairStatuses();
  };

  const renderExpressionPairStatuses = () => {
    knownExpressionPairIds().forEach((pairId) => {
      const status = $(`status-expression-pair-${pairId}`);
      if (!status) return;
      const tpmUploaded = expressionUpload("tpm", pairId);
      const metadataUploaded = expressionUpload("metadata", pairId);
      const upload = $(`upload-expression-pair-${pairId}`);
      if (upload) upload.textContent = tpmUploaded || metadataUploaded ? "替换该对" : "上传该对";
      if (tpmUploaded && metadataUploaded) {
        status.className = "upload-status ok";
        status.textContent = `表达对 ${pairId} 已上传：${tpmUploaded.original_name} + ${metadataUploaded.original_name}`;
      } else if (tpmUploaded || metadataUploaded) {
        status.className = "upload-status bad";
        status.textContent = `表达对 ${pairId} 不完整，请同时重新上传 TPM 和 metadata。`;
      } else {
        status.className = "upload-status";
        status.textContent = runId ? "尚未上传" : "创建任务后可上传";
      }
    });
  };

  const renderInputCards = () => {
    const root = $("inputCards");
    root.replaceChildren();
    inputSpecs.forEach((spec) => {
      const card = document.createElement("article");
      card.className = "input-card";
      card.dataset.kind = spec.kind;
      const title = document.createElement("div");
      title.className = "input-title";
      const heading = document.createElement("h3");
      heading.textContent = spec.label;
      const badge = document.createElement("span");
      badge.className = spec.required ? "badge" : "badge optional";
      badge.textContent = spec.required ? "必需" : "可选";
      title.append(heading, badge);
      const format = document.createElement("p");
      format.className = "format";
      format.textContent = spec.format;
      const source = document.createElement("p");
      source.textContent = `${spec.source || ""} ${spec.collection_note || ""}`.trim();
      const examples = document.createElement("div");
      examples.className = "examples";
      examples.append(makeLink("合规样例", spec.valid_example_url));
      const actions = document.createElement("div");
      actions.className = "file-actions";
      const input = document.createElement("input");
      input.type = "file";
      input.accept = spec.accept;
      input.id = `file-${spec.kind}`;
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = "上传";
      button.addEventListener("click", () => uploadFile(spec));
      actions.append(input, button);
      const status = document.createElement("div");
      status.className = "upload-status";
      status.id = `status-${spec.kind}`;
      status.textContent = "尚未上传";
      card.append(title, format, source, examples, actions, status);
      root.appendChild(card);
    });
    renderInputStatuses();
    renderExpressionPairs();
  };

  const renderInputStatuses = () => {
    inputSpecs.forEach((spec) => {
      const status = $(`status-${spec.kind}`);
      if (!status) return;
      const uploaded = uploadedByKey[spec.input_key];
      if (!uploaded) {
        status.className = "upload-status";
        status.textContent = runId ? "尚未上传" : "创建任务后可上传";
        return;
      }
      status.className = "upload-status ok";
      status.textContent = `${uploaded.original_name} · ${formatBytes(uploaded.size_bytes)} · SHA-256 ${uploaded.sha256.slice(0, 12)}…`;
    });
    renderExpressionPairStatuses();
  };

  const renderChatUpload = (requirements) => {
    const root = $("chatUploadGuide");
    root.replaceChildren();
    if (!runId) {
      const note = document.createElement("p");
      note.textContent = "创建任务后，Agent 会在这里提示并接收当前缺少的文件。";
      root.appendChild(note);
      return;
    }
    if (requirements.ready) {
      const heading = document.createElement("h3");
      heading.textContent = "必需输入已齐备";
      const note = document.createElement("p");
      note.textContent = "现在可以让 Agent 检查并预览计划；表达数据和 KG 先验仍可在右侧按需补充。";
      root.append(heading, note);
      return;
    }
    const guidance = requirements.next_required_guidance || {};
    const heading = document.createElement("h3");
    heading.textContent = `下一步：${guidance.label || requirements.next_required || "上传文件"}`;
    const format = document.createElement("p");
    format.textContent = guidance.format || "请按右侧格式说明准备文件。";
    const source = document.createElement("p");
    source.className = "source";
    source.textContent = `${guidance.source || ""} ${guidance.collection_note || ""}`.trim();
    root.append(heading, format, source);
    if (requirements.next_required === "expression_pair") {
      const actions = document.createElement("div");
      actions.className = "chat-upload-actions";
      const locate = document.createElement("button");
      locate.type = "button";
      locate.textContent = "前往表达对上传";
      locate.addEventListener("click", () => {
        $("expressionPairSection").scrollIntoView({behavior: "smooth", block: "center"});
      });
      actions.appendChild(locate);
      root.appendChild(actions);
      return;
    }
    const spec = inputSpecs.find((item) => item.kind === requirements.next_required);
    if (!spec) return;
    const examples = document.createElement("div");
    examples.className = "examples";
    examples.append(makeLink("下载合规样例", spec.valid_example_url));
    const actions = document.createElement("div");
    actions.className = "chat-upload-actions";
    const input = document.createElement("input");
    input.type = "file";
    input.accept = spec.accept;
    input.hidden = true;
    const upload = document.createElement("button");
    upload.type = "button";
    upload.textContent = "选择并上传文件";
    const status = document.createElement("div");
    status.className = "upload-status";
    status.textContent = "点击按钮选择文件，选中后将自动上传并检查。";
    upload.addEventListener("click", () => {
      input.value = "";
      input.click();
    });
    input.addEventListener("change", async () => {
      if (!input.files.length) return;
      const file = input.files[0];
      upload.disabled = true;
      upload.textContent = "上传中…";
      status.className = "upload-status";
      status.textContent = `已选择 ${file.name}，正在上传并检查…`;
      await uploadSelectedFile(spec, file, status, input);
      if (upload.isConnected) {
        upload.disabled = false;
        upload.textContent = "重新选择并上传";
      }
    });
    actions.append(upload, input);
    root.append(examples, actions, status);
  };

  const syncRequirements = (requirements) => {
    uploadedByKey = {};
    (requirements.uploaded_files || []).forEach((item) => {
      uploadedByKey[item.input_key] = item;
    });
    let expressionRowsChanged = false;
    (requirements.expression_pairs || []).forEach((item) => {
      if (item.pair_id && !expressionPairRows.includes(String(item.pair_id))) {
        expressionPairRows.push(String(item.pair_id));
        expressionRowsChanged = true;
      }
    });
    renderInputStatuses();
    if (expressionRowsChanged) renderExpressionPairs();
    const ready = Boolean(requirements.ready);
    $("inputReadiness").textContent = ready
      ? "输入已齐备"
      : requirements.next_required === "expression_pair"
      ? "待补全：表达 TPM / metadata 对"
      : `待上传：${requirements.next_required || "文件"}`;
    $("inputReadiness").className = ready ? "badge" : "badge warn";
    const source = requirements.target_source === "provided"
      ? "使用外部靶点"
      : "Python NetInfer 生成靶点";
    const mode = requirements.evidence_mode === "kg_proximity_gps" ? "Enhanced" : "Core";
    const pairCount = requirements.inputs && requirements.inputs.expression_pair_count
      ? `表达对：${requirements.inputs.expression_pair_count} 对。`
      : "";
    const planState = requirements.inputs_locked
      ? "任务已确认启动，输入已锁定。"
      : requirements.plan_previewed
      ? "计划已预览；仍可补充文件，输入变更后需要重新预览。"
      : "";
    $("inputNote").textContent = `${mode} · ${source}。${pairCount}上传只做轻量结构检查，科学完整性由 DAG 节点验证。${planState}`;
    renderChatUpload(requirements);
  };

  const uploadSelectedFile = async (spec, file, status, inputToClear) => {
    const targetRun = runId;
    if (!targetRun) {
      status.className = "upload-status bad";
      status.textContent = "请先通过 Agent 创建任务。";
      return;
    }
    const form = new FormData();
    form.append("upload", file);
    const replace = Boolean(uploadedByKey[spec.input_key]);
    status.className = "upload-status";
    status.textContent = `正在上传 ${file.name} 并检查…`;
    try {
      const response = await fetch(
        `/runs/${encodeURIComponent(targetRun)}/files/${spec.kind}?replace=${replace}`,
        {method: "POST", body: form}
      );
      if (!response.ok) throw new Error(await errorText(response));
      const body = await response.json();
      if (targetRun !== runId) return;
      const validation = body.input.validation || {};
      status.className = "upload-status ok";
      status.textContent = `基础检查通过：${validation.message || "格式已识别"}`;
      syncRequirements(body.requirements);
      addMessage(
        body.receipt && body.receipt.content
          ? body.receipt.content
          : `${spec.label}“${body.input.original_name}”上传成功，基础结构检查通过。`,
        "assistant"
      );
      if (inputToClear) inputToClear.value = "";
      refreshHistory().catch(() => {});
    } catch (error) {
      if (targetRun !== runId) return;
      status.className = "upload-status bad";
      status.textContent = `未接收：${error.message}`;
    }
  };

  const uploadFile = async (spec) => {
    const input = $(`file-${spec.kind}`);
    const status = $(`status-${spec.kind}`);
    if (!input.files.length) {
      status.className = "upload-status bad";
      status.textContent = "请先选择文件。";
      return;
    }
    await uploadSelectedFile(spec, input.files[0], status, input);
  };

  const uploadExpressionPair = async (pairId) => {
    const targetRun = runId;
    const tpmInput = $(`file-expression-tpm-${pairId}`);
    const metadataInput = $(`file-expression-metadata-${pairId}`);
    const status = $(`status-expression-pair-${pairId}`);
    if (!targetRun) {
      status.className = "upload-status bad";
      status.textContent = "请先通过 Agent 创建任务。";
      return;
    }
    if (!tpmInput.files.length || !metadataInput.files.length) {
      status.className = "upload-status bad";
      status.textContent = "每个表达对必须同时选择 TPM 和 metadata 两个文件。";
      return;
    }
    const form = new FormData();
    form.append("tpm", tpmInput.files[0]);
    form.append("metadata", metadataInput.files[0]);
    const replace = Boolean(
      expressionUpload("tpm", pairId) || expressionUpload("metadata", pairId)
    );
    status.className = "upload-status";
    status.textContent = `表达对 ${pairId} 上传并检查中…`;
    try {
      const response = await fetch(
        `/runs/${encodeURIComponent(targetRun)}/expression-pairs/${encodeURIComponent(pairId)}?replace=${replace}`,
        {method: "POST", body: form}
      );
      if (!response.ok) throw new Error(await errorText(response));
      const body = await response.json();
      if (targetRun !== runId) return;
      status.className = "upload-status ok";
      status.textContent = `表达对 ${pairId} 基础检查通过。`;
      syncRequirements(body.requirements);
      addMessage(
        body.receipt && body.receipt.content
          ? body.receipt.content
          : `表达对 ${pairId} 上传成功，TPM 与 metadata 基础结构检查通过。`,
        "assistant"
      );
      tpmInput.value = "";
      metadataInput.value = "";
      refreshHistory().catch(() => {});
    } catch (error) {
      if (targetRun !== runId) return;
      status.className = "upload-status bad";
      status.textContent = `表达对 ${pairId} 未接收：${error.message}`;
    }
  };

  const scoreLabel = (value, digits=3) => {
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(digits) : "—";
  };

  const renderCandidatePreview = () => {
    const preview = $("candidatePreview");
    preview.replaceChildren();
    if (!displayedCandidates.length) {
      preview.className = "empty";
      preview.textContent = "当前没有可预览的候选记录。";
      return;
    }
    preview.className = "";
    const visible = displayedCandidates.slice(0, candidateVisibleCount);
    const caption = document.createElement("div");
    caption.className = "artifact-path";
    caption.textContent = `候选小分子清单：共 ${displayedCandidateTotal} 条；已展示 ${visible.length} 条，界面最多加载 Top100。`;
    const grid = document.createElement("div");
    grid.className = "candidate-grid";
    visible.forEach((candidate) => {
      const card = document.createElement("article");
      card.className = "candidate-card";
      let structure;
      if (candidate.structure_available && candidate.structure_url) {
        structure = document.createElement("img");
        structure.className = "candidate-structure";
        structure.src = candidate.structure_url;
        structure.alt = `${candidate.compound_name || candidate.compound_id} 的二维结构`;
        structure.loading = "lazy";
        structure.addEventListener("error", () => {
          const fallback = document.createElement("div");
          fallback.className = "candidate-structure-fallback";
          fallback.textContent = "二维结构暂不可用";
          structure.replaceWith(fallback);
        });
      } else {
        structure = document.createElement("div");
        structure.className = "candidate-structure-fallback";
        structure.textContent = candidate.smiles ? "SMILES 已记录" : "无结构信息";
      }
      const content = document.createElement("div");
      const name = document.createElement("div");
      name.className = "candidate-name";
      name.textContent = `#${candidate.final_rank || "—"} ${candidate.compound_name || candidate.compound_id || "未命名候选"}`;
      const id = document.createElement("div");
      id.className = "candidate-id";
      id.textContent = candidate.compound_id || "—";
      const chips = document.createElement("div");
      chips.className = "candidate-chips";
      [
        `证据 ${candidate.evidence_count || "—"}`,
        `KG ${scoreLabel(candidate.kg_rank_mean, 1)}`,
        `PPI Z ${scoreLabel(candidate.proximity_z)}`,
        candidate.gps_score ? `GPS ${scoreLabel(candidate.gps_score)}` : null
      ].filter(Boolean).forEach((value) => {
        const chip = document.createElement("span");
        chip.className = "candidate-chip";
        chip.textContent = value;
        chips.appendChild(chip);
      });
      const properties = document.createElement("div");
      properties.className = "candidate-properties";
      const values = candidate.properties || {};
      [
        ["分子式", values.formula],
        ["分子量", values.molecular_weight],
        ["cLogP", values.logp],
        ["TPSA", values.tpsa],
        ["HBD/HBA", values.h_bond_donors === undefined
          ? null
          : `${values.h_bond_donors}/${values.h_bond_acceptors}`],
        ["可旋转键", values.rotatable_bonds]
      ].forEach(([label, value]) => {
        const fact = document.createElement("span");
        fact.textContent = `${label}：${value === null || value === undefined || value === "" ? "—" : value}`;
        properties.appendChild(fact);
      });
      const smiles = document.createElement("div");
      smiles.className = "candidate-smiles";
      smiles.textContent = candidate.smiles || "SMILES unavailable";
      content.append(name, id, chips, properties, smiles);
      card.append(structure, content);
      grid.appendChild(card);
    });
    preview.append(caption, grid);
    if (visible.length < displayedCandidates.length) {
      const more = document.createElement("div");
      more.className = "candidate-more";
      const button = document.createElement("button");
      button.type = "button";
      button.className = "secondary";
      button.textContent = `继续显示（剩余 ${displayedCandidates.length - visible.length}）`;
      button.addEventListener("click", () => {
        candidateVisibleCount += 12;
        renderCandidatePreview();
      });
      more.appendChild(button);
      preview.appendChild(more);
    }
  };

  const renderResults = (body, targetRun) => {
    if (targetRun !== runId) return;
    displayedCandidates = body.candidate_preview || [];
    displayedCandidateTotal = body.candidate_count ??
      (body.ranking_summary && body.ranking_summary.stage_counts
        ? body.ranking_summary.stage_counts.final_candidates
        : displayedCandidates.length);
    candidateVisibleCount = 12;
    renderCandidatePreview();
    const list = $("artifactList");
    list.replaceChildren();
    const artifacts = body.artifacts || [];
    $("artifactHint").textContent = artifacts.length
      ? "以下为后端真实记录的最终产物下载入口。完整排名请下载“完整候选排名表”。"
      : "当前没有可下载的最终产物。";
    artifacts.forEach((artifact) => {
      const item = document.createElement("div");
      item.className = "artifact";
      const main = document.createElement("div");
      main.className = "artifact-main";
      const title = document.createElement("div");
      title.className = "artifact-title";
      title.textContent = artifactLabel(artifact);
      const path = document.createElement("div");
      path.className = "artifact-path";
      path.textContent = artifact.relative_path || artifact.artifact_type || artifact.artifact_id || "结果文件";
      const link = document.createElement("a");
      link.className = "artifact-download";
      link.href = artifactDownloadUrl(artifact, targetRun);
      link.setAttribute("download", "");
      link.textContent = "下载";
      main.append(title, path);
      item.append(main, link);
      list.appendChild(item);
    });
  };

  const fetchResults = async (targetRun) => {
    if (!targetRun || resultsLoadedFor === targetRun || resultsLoadingFor === targetRun) return;
    resultsLoadingFor = targetRun;
    try {
      const response = await fetch(`/runs/${encodeURIComponent(targetRun)}/results`);
      if (!response.ok) throw new Error(await errorText(response));
      const body = await response.json();
      if (targetRun !== runId) return;
      renderResults(body, targetRun);
      resultsLoadedFor = targetRun;
    } catch (error) {
      if (targetRun === runId) {
        $("candidatePreview").textContent = `结果读取失败：${error.message}`;
      }
    } finally {
      if (resultsLoadingFor === targetRun) resultsLoadingFor = null;
    }
  };

  const cancelController = logic.createCancelController({
    getContext: () => ({
      runId,
      snapshot: runSnapshot,
      currentNode: logic.currentNode(runSnapshot)
    }),
    confirmCancel: ({runId: targetRun, currentNode}) => {
      const node = currentNode ? `${nodeName(currentNode)}（${logic.stableNodeKey(currentNode)}）` : "暂无活动节点";
      return window.confirm(`确认终止当前任务？\nRun ID：${targetRun}\n当前节点：${node}`);
    },
    requestCancel: async (targetRun) => {
      const response = await fetch(`/runs/${encodeURIComponent(targetRun)}/cancel`, {
        method: "POST"
      });
      if (!response.ok) throw new Error(await errorText(response));
      return response.json();
    },
    onStart: (targetRun) => {
      cancelPendingRun = targetRun;
      $("cancelError").textContent = "";
      renderTaskControl();
    },
    onSuccess: (snapshot, targetRun) => {
      cancelPendingRun = null;
      if (targetRun === runId) {
        renderRun(snapshot);
        poller.refresh();
      }
    },
    onError: (error, targetRun) => {
      cancelPendingRun = null;
      if (targetRun === runId) {
        $("cancelError").textContent = `终止失败：${logic.safeErrorMessage(error)}`;
        renderTaskControl();
      }
    }
  });

  const send = async () => {
    const message = $("message").value.trim();
    const apiKey = $("apiKey").value.trim();
    if (!message) return;
    if (!apiKey && !modelConfigured) {
      addMessage("请先输入 API Key。", "error");
      return;
    }
    if (!threadId) await createSession();
    const targetThread = threadId;
    const epoch = sessionEpoch;
    addMessage(message, "user");
    const streamView = addStreamingAssistant();
    $("message").value = "";
    $("send").disabled = true;
    chatController = new AbortController();
    try {
      const headers = {"Content-Type": "application/json"};
      if (apiKey) headers["X-Model-API-Key"] = apiKey;
      const response = await fetch(`/sessions/${encodeURIComponent(targetThread)}/chat/stream`, {
        method: "POST",
        headers,
        body: JSON.stringify({message}),
        signal: chatController.signal
      });
      if (!response.ok) throw new Error(await errorText(response));
      if (!response.body) throw new Error("浏览器未提供可读取的响应流");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let finalEvent = null;
      const parser = logic.createNdjsonParser((event) => {
        if (epoch !== sessionEpoch || targetThread !== threadId) return;
        applyStreamEvent(streamView, event);
        if (event.type === "final") finalEvent = event;
      });
      while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        parser.push(decoder.decode(value, {stream: true}));
      }
      parser.push(decoder.decode());
      parser.finish();
      if (epoch !== sessionEpoch || targetThread !== threadId) return;
      if (!streamView.finalSeen && !streamView.errorSeen) {
        streamView.content.textContent = "连接已结束，但没有收到完整回复。请重试。";
        streamView.content.classList.add("error");
        streamView.node.setAttribute("aria-busy", "false");
      }
      if (finalEvent && finalEvent.run_id) await setRun(finalEvent.run_id);
      else if (runId) await poller.refresh();
    } catch (error) {
      if (error.name !== "AbortError" && epoch === sessionEpoch) {
        streamView.content.textContent = `请求失败：${error.message}\n请重试。`;
        streamView.content.classList.add("error");
        streamView.node.setAttribute("aria-busy", "false");
      }
    } finally {
      if (epoch === sessionEpoch) {
        $("send").disabled = false;
        $("message").focus();
        refreshHistory().catch(() => {});
      }
    }
  };

  const loadSpecs = async () => {
    const response = await fetch("/input-specs");
    if (!response.ok) throw new Error(await errorText(response));
    const body = await response.json();
    inputSpecs = body.inputs || [];
    expressionSpec = body.expression_pair || null;
    $("inputNote").textContent = body.note;
    renderInputCards();
  };

  $("send").addEventListener("click", send);
  $("message").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && event.ctrlKey) {
      event.preventDefault();
      send();
    }
  });
  $("toggleKey").addEventListener("click", () => {
    const hidden = $("apiKey").type === "password";
    $("apiKey").type = hidden ? "text" : "password";
    $("toggleKey").textContent = hidden ? "隐藏" : "显示";
  });
  $("newSession").addEventListener("click", async () => {
    try {
      await createSession();
      addMessage("请先告诉我本次筛选关注的疾病。任务创建后可直接在对话区上传所需文件。", "assistant");
    } catch (error) {
      addMessage(error.message, "error");
    }
  });
  $("history").addEventListener("click", async () => {
    $("historyDialog").showModal();
    try {
      await refreshHistory();
    } catch (error) {
      $("historyList").textContent = `历史记录读取失败：${error.message}`;
    }
  });
  $("closeHistory").addEventListener("click", () => $("historyDialog").close());
  $("refresh").addEventListener("click", () => poller.refresh());
  $("cancelRun").addEventListener("click", () => cancelController.cancel());
  document.querySelectorAll(".quick").forEach((button) => {
    button.addEventListener("click", () => {
      $("message").value = button.dataset.message;
      send();
    });
  });
  renderTaskControl();
  renderModuleCards();
  renderDag();
  Promise.all([checkHealth(), loadSpecs()])
    .then(() => bootstrapSession())
    .catch((error) => addMessage(error.message, "error"));
})();
</script>
</body>
</html>
"""

WEB_UI = _WEB_UI_HEAD + WEB_UI_LOGIC + _WEB_UI_APP

__all__ = ["WEB_UI"]
