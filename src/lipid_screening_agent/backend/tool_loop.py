"""Small OpenAI-compatible tool-calling loop; no LangGraph or prompt stack."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from .service import ScreeningService
from .store import V3Store

SYSTEM_PROMPT = """你是降脂药物筛选计算工作流助手，负责把用户对话转换为受控工作流操作。
只通过已提供工具读取或改变运行状态，不执行或模拟科学算法，不编造文件、状态、结果或结论。
每次优先推进一个明确阶段，并使用简洁中文：
1. intake：确认疾病名称；信息明确后创建唯一任务。
2. required_inputs：按 next_required 逐项提示用户在对话区或右侧上传文件，并说明格式、来源和样例入口。
3. plan_review：输入齐备后先预览固定 DAG；预览不锁定输入，补充文件后需要重新预览。
4. confirmation：必须先向用户展示计划并取得“确认启动”等明确授权；不得从模糊表达推断授权。
5. execution：启动后查询真实状态，说明运行中、失败节点或完成状态。
6. results：只依据工具返回的候选与证据字段解释排名理由，并提醒需要实验和专业复核。
文件上传和下载由浏览器或 REST 完成，你不能声称替用户读取本地路径、上传文件或替用户点击下载。
当前版本不自动从外部数据库选取、下载或标准化研究数据，因为数据集选择、授权、样本分组和
标准化方案需要用户确认；应依据下列来源信息引导用户准备文件，不得声称已经联网收集：
- compounds 可来自用户已有化合物库、TargetMol/商业库、实验室或自定义集合。
- disease_genes 可来自文献、差异表达分析、疾病数据库或专家整理，并优先统一为 HGNC symbol。
- expression_pair 可来自 GEO、SRA、ArrayExpress、TCGA 或用户自己的表达数据和样本注释。
- positive_drugs 可来自文献、指南、数据库或实验确认；disease_links 来自当前 KG 基础资源节点。
当 get_results 返回 artifacts 的 download_url 时，必须如实告知用户可在页面“结果与下载”区域下载，
也可以直接给出这些相对下载链接；不得声称系统没有下载入口。
只能使用以下可信格式说明，不得自行增加或改写格式：
- compounds：CSV/TSV 表格，必须有 ID 和 SMILES 两列；这是必需文件。
- disease_genes：TXT/TSV/CSV，每行一个 HGNC gene symbol，或 symbol/entrez_id 表格；这是必需文件。
- positive_drugs：可选 TSV，KG 先验输入，必须含 input_type 和 value；input_type 仅为 library_id、base_drug_name 或 base_drug_id。
- disease_links：可选 TSV，KG 先验输入，必须含 input_type 和 value；input_type 仅为 base_disease_id 或 base_disease_name，可附加 node_name。
- expression_pair：可选多对 TSV；每对必须同时上传 TPM_matrix_<编号>.tsv 和 metadata_<编号>.tsv。
  TPM 首列必须为 GeneID，其余列为样本 ID；metadata 必须含 sample_id 和 group，group 仅为 control/disease。
页面为每种文件提供“合规样例”。不要向用户推荐或展示不合规样例。
不要使用 Markdown 表格，使用短段落或短列表。
只能使用以下可信工作流语义，不得自行发明算法描述：
- Core = Python wSDTNBI/NetInfer + PPI Proximity + KG；证据模式为 kg_proximity。
- Enhanced = Core + GPS 表达逆转分支；证据模式为 kg_proximity_gps。
- NetInfer 是基于药物-子结构-靶点网络的加权网络推断，不得称为深度学习。
- Proximity 在人 PPI 网络上计算化合物靶点集合与疾病基因集合的网络邻近性。
- GPS 从 TPM/metadata 构建疾病差异表达特征，与预测的化合物扰动特征计算表达逆转分数；
  不得称为余弦相似度，配置中分数越低越优。
- KG 构图后执行预训练、种子微调与多种子聚合，为所有模式提供 KG 排名。
- 最终排名对满足配置阈值且处于可用证据交集的候选计算证据内百分位并求均值；
  不自动放宽阈值。
- 当前默认正式配置的 KG 截断为 ranking.kg.top_n=200；若结果 summary 显示其他数值，
  以 get_results 返回的 thresholds.kg.top_n 为准。Enhanced 结果通常先由 KG Top200、
  Proximity z < 0 和 GPS < 0 的证据交集形成候选清单，再排序。NetInfer 不作为
  final_candidates.tsv 中的独立平均分字段；它通过靶点推断进入 Proximity 靶点集合和 KG DTI 边。
- 结果解释只能使用 get_results 返回的字段。不得补充工具未返回的药物类别、已知适应证、
  安全性、肝毒性、临床作用或文献知识。
- 当前工具集中没有文献检索、DeepSeek 深度调研或网页搜索工具；不得把普通模型推理称为
  文献调研、深度调研、真实检索或批量文献证据。
- KG 只按 kg_rank_mean 越低越优解释；kg_score_mean 是未校准原始量，不得把数值大小直接
  解释为关联强度。Proximity Z 和 GPS 分数按配置越低越优，但没有 p 值时不得称为显著、
  高度邻近或接近随机水平。python_netinfer 结果可能同时含已知与预测靶点，不得称为全部预测。
结果仅代表计算优先级，不代表药效、安全性、机制证明、处方或临床建议。"""

STAGE_GUIDANCE = {
    "intake": (
        "当前没有任务。若用户最新消息已给出疾病并要求创建任务，必须立即调用 "
        "create_screening_run，不要重复询问。肝脂肪变性/脂肪肝可规范为 "
        "disease_name=hepatic steatosis、disease_slug=hepatic_steatosis。"
    ),
    "required_inputs": (
        "任务正在收集输入。只提示 context.next_required 对应文件；使用 "
        "context.next_required_guidance 说明来源与准备方法，并提醒对话区和右侧都有上传入口及合规样例。"
    ),
    "plan_review": (
        "必需输入已齐备但尚未锁定。用户要求检查或继续时调用 preview_execution_plan，"
        "然后解释分支并请求明确启动确认。"
    ),
    "confirmation": (
        "计划已预览但输入尚未锁定。用户仍可补充可选文件；只有用户明确确认启动时才调用 "
        "start_workflow，启动时才按最终输入锁定并生成正式计划。"
    ),
    "execution": "工作流已排队或运行。按用户请求调用 get_workflow_status，绝不猜测进度。",
    "results": (
        "工作流已结束。调用 get_results；成功时按候选排名、KG、proximity、GPS 字段解释，"
        "说明 final_candidates 数量、KG top_n、可用证据交集和可下载 artifacts 的 "
        "download_label 与 download_url；失败时引用真实错误和节点。"
    ),
}


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str

    def to_message_value(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass(frozen=True, slots=True)
class ModelReply:
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    def to_message(self) -> dict[str, Any]:
        value: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            value["tool_calls"] = [call.to_message_value() for call in self.tool_calls]
        return value


class ChatModel(Protocol):
    def complete(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelReply: ...


class OpenAICompatibleModel:
    """Thin adapter for DeepSeek or another OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 90.0,
        max_retries: int = 2,
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.model = model

    def complete(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelReply:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
            tool_choice="auto",
            temperature=0.0,
        )
        message = response.choices[0].message
        calls = tuple(
            ToolCall(
                id=call.id,
                name=call.function.name,
                arguments=call.function.arguments,
            )
            for call in (message.tool_calls or ())
        )
        return ModelReply(content=message.content, tool_calls=calls)


def _function(name: str, description: str, properties: dict[str, Any], required=()) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(required),
                "additionalProperties": False,
            },
        },
    }


TOOL_DEFINITIONS = [
    _function(
        "create_screening_run",
        "Create the single screening run for this session after disease identity is clear.",
        {
            "disease_name": {"type": "string", "minLength": 1, "maxLength": 200},
            "disease_slug": {
                "type": "string",
                "pattern": "^[a-z0-9](?:[a-z0-9_]{0,62}[a-z0-9])?$",
            },
        },
        ("disease_name", "disease_slug"),
    ),
    _function("get_requirements", "Return the trusted input checklist.", {}),
    _function("preview_execution_plan", "Build and return the fixed plan without starting it.", {}),
    _function(
        "start_workflow",
        "Start only after the user explicitly confirms execution.",
        {"confirmed": {"type": "boolean"}},
        ("confirmed",),
    ),
    _function("get_workflow_status", "Return current workflow and node status.", {}),
    _function("cancel_workflow", "Cancel the current queued or running workflow.", {}),
    _function("get_results", "Return final computational artifacts when available.", {}),
]


class ToolDispatcher:
    def __init__(self, service: ScreeningService) -> None:
        self.service = service

    def execute(self, thread_id: str, call: ToolCall) -> dict[str, Any]:
        try:
            arguments = json.loads(call.arguments or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError("tool arguments are not valid JSON") from exc
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be an object")

        if call.name == "create_screening_run":
            return self.service.create_run(
                thread_id=thread_id,
                disease_name=str(arguments["disease_name"]),
                disease_slug=str(arguments["disease_slug"]),
            )
        run = self.service.run_for_thread(thread_id)
        if run is None:
            raise ValueError("create a screening run first")
        run_id = str(run["run_id"])
        if call.name == "get_requirements":
            return self.service.requirements(run_id)
        if call.name == "preview_execution_plan":
            return _compact_plan(self.service.preview_plan(run_id))
        if call.name == "start_workflow":
            confirmed = arguments.get("confirmed")
            if not isinstance(confirmed, bool):
                raise ValueError("confirmed must be a boolean")
            return _compact_status(self.service.start(run_id, confirmed=confirmed))
        if call.name == "get_workflow_status":
            return _compact_status(self.service.snapshot(run_id))
        if call.name == "cancel_workflow":
            return _compact_status(self.service.cancel(run_id))
        if call.name == "get_results":
            return _compact_results(self.service.results(run_id))
        raise ValueError(f"unknown tool: {call.name}")


class ToolCallingAgent:
    def __init__(
        self,
        *,
        model: ChatModel,
        store: V3Store,
        dispatcher: ToolDispatcher,
        max_rounds: int = 4,
    ) -> None:
        self.model = model
        self.store = store
        self.dispatcher = dispatcher
        self.max_rounds = max(1, max_rounds)

    def chat(self, *, thread_id: str, user_message: str) -> dict[str, Any]:
        final: dict[str, Any] | None = None
        for event in self.chat_events(thread_id=thread_id, user_message=user_message):
            if event.get("type") == "final":
                final = {
                    "content": event["content"],
                    "tool_events": event["tool_events"],
                    "run_id": event["run_id"],
                }
        if final is None:
            raise RuntimeError("agent stream ended without a final response")
        return final

    def chat_events(
        self, *, thread_id: str, user_message: str
    ) -> Iterator[dict[str, Any]]:
        """Yield true model/tool stages while preserving the original final response contract."""

        text = user_message.strip()
        if not text:
            raise ValueError("message must not be empty")
        if len(text) > 4000:
            raise ValueError("message exceeds 4000 characters")
        self.store.session(thread_id)
        self.store.append_message(thread_id, {"role": "user", "content": text})
        events: list[dict[str, Any]] = []
        yield {"type": "stage", "stage": "agent", "status": "started"}

        for round_index in range(1, self.max_rounds + 1):
            messages = [{"role": "system", "content": self._system_prompt(thread_id)}]
            messages.extend(self.store.messages(thread_id, limit=24))
            yield {
                "type": "stage",
                "stage": "model",
                "status": "started",
                "round": round_index,
            }
            reply = self.model.complete(messages=messages, tools=TOOL_DEFINITIONS)
            yield {
                "type": "stage",
                "stage": "model",
                "status": "completed",
                "round": round_index,
            }
            assistant = reply.to_message()
            self.store.append_message(thread_id, assistant)
            if not reply.tool_calls:
                content = (reply.content or "").strip()
                if not content:
                    raise RuntimeError("model returned neither text nor tool calls")
                run = self.service_run(thread_id)
                yield {
                    "type": "final",
                    "content": content,
                    "tool_events": events,
                    "run_id": run,
                }
                return

            for call in reply.tool_calls:
                yield {
                    "type": "tool",
                    "name": call.name,
                    "status": "started",
                    "round": round_index,
                }
                try:
                    result = self.dispatcher.execute(thread_id, call)
                    payload = {"ok": True, "result": result}
                    events.append({"name": call.name, "status": "succeeded"})
                    streamed_event = {
                        "type": "tool",
                        "name": call.name,
                        "status": "succeeded",
                        "round": round_index,
                    }
                except Exception as exc:
                    payload = {
                        "ok": False,
                        "error": {"type": type(exc).__name__, "message": str(exc)[:500]},
                    }
                    events.append(
                        {"name": call.name, "status": "failed", "error": type(exc).__name__}
                    )
                    streamed_event = {
                        "type": "tool",
                        "name": call.name,
                        "status": "failed",
                        "error": type(exc).__name__,
                        "round": round_index,
                    }
                self.store.append_message(
                    thread_id,
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                    },
                )
                yield streamed_event

        raise RuntimeError("tool-calling loop exceeded its round limit")

    def service_run(self, thread_id: str) -> str | None:
        run = self.dispatcher.service.run_for_thread(thread_id)
        return None if run is None else str(run["run_id"])

    def _system_prompt(self, thread_id: str) -> str:
        run = self.dispatcher.service.run_for_thread(thread_id)
        if run is None:
            stage = "intake"
            context = {"stage": stage, "run_id": None}
        else:
            run_id = str(run["run_id"])
            snapshot = self.dispatcher.service.snapshot(run_id)
            status = str(snapshot.get("status", "collecting"))
            if not run["workflow_created"]:
                requirements = snapshot.get("requirements") or {}
                stage = (
                    "confirmation"
                    if requirements.get("ready") and run.get("plan_previewed")
                    else "plan_review"
                    if requirements.get("ready")
                    else "required_inputs"
                )
                context = {
                    "stage": stage,
                    "run_id": run_id,
                    "status": status,
                    "next_required": requirements.get("next_required"),
                    "next_required_guidance": requirements.get(
                        "next_required_guidance"
                    ),
                    "inputs": requirements.get("inputs"),
                    "evidence_mode": requirements.get("evidence_mode"),
                    "target_source": requirements.get("target_source"),
                }
            elif status in {"created", "ready"}:
                stage = "confirmation"
                context = {"stage": stage, "run_id": run_id, "status": status}
            elif status in {"queued", "running", "cancelling"}:
                stage = "execution"
                context = {"stage": stage, "run_id": run_id, "status": status}
            else:
                stage = "results"
                context = {"stage": stage, "run_id": run_id, "status": status}
        encoded = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        return f"{SYSTEM_PROMPT}\n\n阶段指令：{STAGE_GUIDANCE[stage]}\n可信状态 context={encoded}"


def _compact_plan(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": value.get("run_id"),
        "mode": value.get("mode"),
        "evidence_mode": value.get("evidence_mode"),
        "nodes": [
            {
                "node_id": item.get("node_id"),
                "task_id": item.get("task_id"),
                "stage": item.get("stage"),
                "status": item.get("initial_status"),
                "queue": item.get("resource_class"),
            }
            for item in value.get("nodes", [])
        ],
        "module_readiness": value.get("module_readiness", {}),
    }


def _compact_status(value: dict[str, Any]) -> dict[str, Any]:
    result = {
        "run_id": value.get("run_id"),
        "status": value.get("status"),
        "evidence_mode": value.get("evidence_mode"),
        "nodes": [
            {
                "node_id": item.get("node_id"),
                "task_id": item.get("task_id"),
                "status": item.get("status"),
                "progress": item.get("progress"),
                "error": item.get("error"),
            }
            for item in value.get("nodes", [])
        ],
    }
    if "requirements" in value:
        result["requirements"] = value["requirements"]
    return result


def _compact_results(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": value.get("run_id"),
        "status": value.get("status"),
        "evidence_mode": value.get("evidence_mode"),
        "artifacts": [
            {
                "artifact_id": item.get("artifact_id"),
                "artifact_type": item.get("artifact_type"),
                "relative_path": item.get("relative_path"),
                "file_name": item.get("file_name"),
                "download_label": item.get("download_label"),
                "download_url": item.get("download_url"),
                "sha256": item.get("sha256"),
            }
            for item in value.get("artifacts", [])
        ],
        "candidate_preview": value.get("candidate_preview", [])[:10],
        "candidate_count": value.get("candidate_count"),
        "ranking_summary": value.get("ranking_summary"),
        "scope": value.get("scope", "computational_prioritization_only"),
    }


__all__ = [
    "ChatModel",
    "ModelReply",
    "OpenAICompatibleModel",
    "SYSTEM_PROMPT",
    "STAGE_GUIDANCE",
    "TOOL_DEFINITIONS",
    "ToolCall",
    "ToolCallingAgent",
    "ToolDispatcher",
]
