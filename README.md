# smi2phen

**An Agent-assisted Multi-perspective Drug Screening Workflow**

**以科研工作流为核心、智能体辅助交互的多角度候选药物筛选系统**

smi2phen 是一个面向候选药物优先级研究的开源软件工作流。系统将药物–靶点推断、
网络邻近性、表达逆转和知识图谱学习等计算证据组织在统一的可复现流程中，并通过
Agent 辅助用户准备输入、理解执行计划、确认任务、查看进度和定位结果。

Agent 负责交互与解释，科学计算由固定的 Workflow/DAG 执行。候选排序仅表示后续研究
的优先级，不代表疗效、毒性、安全性或临床有效性，也不构成医疗或用药建议。

## 主要功能

- **Agent 辅助使用**：引导输入、解释计划、确认执行、查询状态和导航结果。
- **药物–靶点推断**：使用 NetInfer 推断候选化合物的潜在靶点，也支持导入已验证的
  药物–靶点关系。
- **网络邻近性分析**：在 PPI 网络上评估候选药物靶点与疾病基因模块的接近程度。
- **表达逆转分析**：在提供 TPM 矩阵和样本分组信息时，使用 GPS 计算可选的表达逆转证据。
- **知识图谱学习**：整合候选化合物、靶点、疾病及先验关系，执行固定种子的图学习流程。
- **多证据共识排序**：聚合各科学模块的证据，生成候选药物优先级列表及运行报告。
- **可追踪运行**：保存输入校验、节点状态、参数、资源哈希和结果 artifact，便于检查与复现。

## Core 与 Enhanced 模式

| 模式 | 主要证据 | 适用输入 |
| --- | --- | --- |
| **Core** | NetInfer、网络邻近性、知识图谱、共识排序 | 化合物库和疾病基因 |
| **Enhanced** | Core 的全部证据，加上 GPS 表达逆转 | 化合物库、疾病基因、TPM 矩阵和样本元数据 |

如果提供 `drug_targets.json` 和 `target_mapping.tsv`，系统可以使用已有靶点关系并跳过
NetInfer。未提供表达数据时，Workflow 会自动跳过 GPS，按 Core 模式执行。

## 输入与输出

### 主要输入

| 输入 | 要求 | 是否必需 |
| --- | --- | --- |
| 候选化合物 | CSV，包含唯一的 `ID` 和非空 `SMILES` | 是 |
| 疾病基因 | 支持基因 Symbol 和/或 Entrez ID 的表格 | 是 |
| 表达矩阵 | TPM 表格，第一列为基因标识，其余列为样本 | Enhanced 必需 |
| 样本元数据 | 包含样本名称和分组信息，与 TPM 配对 | Enhanced 必需 |
| 已有药物–靶点关系 | `drug_targets.json` 与 `target_mapping.tsv` | 可选 |
| KG 先验关系 | 阳性药物和疾病关联表 | 可选 |

仓库提供两类示例：

- [`examples/minimal_inputs/`](examples/minimal_inputs/)：用于熟悉格式的最小输入；
- [`examples/full_inputs/`](examples/full_inputs/)：完整保留的示例输入集。

输入和 artifact 的详细字段约定见 [`contracts/`](contracts/)。

### 主要输出

- `final_candidates.tsv`：候选药物共识排序；
- `ranking_summary.json`：排序摘要和证据概览；
- 各科学模块的中间结果、日志和 manifest；
- 运行级报告及可下载 artifact。

每次运行使用独立工作目录。结果需要结合资源覆盖度、输入质量和方法假设进行解释，并接受
进一步计算或实验验证。

## 快速开始

### 环境要求

- Git；
- Python 3.10 或更高版本，用于资源下载与校验脚本；
- Docker Desktop 或 Docker Engine，并支持 Docker Compose；
- NVIDIA GPU、兼容的 NVIDIA 驱动及 Docker GPU Runtime；
- 足够的磁盘空间用于约 4.24 GB 的容器镜像、约 304 MiB 的解压后科研资源及运行结果。

### 1. 获取项目并创建本地配置

```bash
git clone https://github.com/beiweiClover/smi2phen.git
cd smi2phen
```

Linux 或 macOS：

```bash
cp .env.example .env
```

Windows CMD：

```bat
copy .env.example .env
```

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

如需使用大模型辅助对话，在 `.env` 中填写自己的 API Key。直接使用 Web/API 创建和执行
Workflow 时，模型 API Key 不是必需项。请勿将 `.env` 或任何真实凭据提交到 Git。

如果本机端口 `8000` 已被占用，可在 `.env` 中设置：

```text
SMI2PHEN_API_PORT=18000
```

### 2. 下载完整科研资源

```bash
python scripts/download_resources.py
python scripts/check_resources.py --mode enhanced
```

下载脚本会从 [v0.1.0 Release](https://github.com/beiweiClover/smi2phen/releases/tag/v0.1.0)
获取完整资源包，默认安装到 `.local-resources/`。脚本会校验资源包及其中每个文件的
SHA-256；已存在且校验正确的文件会被跳过。

准备完成时应显示：

```text
core: READY
enhanced: READY
```

### 3. 启动服务

```bash
docker compose pull
docker compose up -d
docker compose ps
```

固定版本镜像为：

```text
ghcr.io/beiweiclover/smi2phen:v0.1.0
```

当 `api`、`redis` 和 `unified-worker` 均显示 `healthy` 后，访问：

```text
http://127.0.0.1:8000/
```

如修改了 `SMI2PHEN_API_PORT`，请使用对应端口。健康检查地址为：

```text
http://127.0.0.1:8000/healthz
```

停止服务：

```bash
docker compose down
```

## Web 使用流程

1. 创建新的研究会话和运行任务；
2. 填写疾病名称并上传化合物、疾病基因及可选输入；
3. 查看系统生成的 Core 或 Enhanced 执行计划；
4. 确认输入和计划后启动 Workflow；
5. 在任务页面查看节点进度、状态与提示；
6. 任务结束后检查并下载排序结果、模块结果和运行报告。

新部署不会携带历史用户、会话、上传文件或运行记录。更详细的 Web 与 API 说明见
[`docs/API_AND_WEB.md`](docs/API_AND_WEB.md)。

## 示例结果

[`examples/demo_result/final_candidates_no_toxicity.tsv`](examples/demo_result/final_candidates_no_toxicity.tsv)
是一个静态输出格式示例，包含 80 条候选记录。

## 科研资源与复现

大型科研资源不写入 Git 历史，也不打包进 Docker 镜像。完整资源包通过 GitHub Release
独立分发，并由 [`resources/manifest.json`](resources/manifest.json) 记录文件名、目标路径、
大小、SHA-256、版本、来源和许可信息。

复现要求和资源说明见：

- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)
- [`docs/RESOURCES.md`](docs/RESOURCES.md)

## 方法学边界

- 各计算模块提供的是具有不同假设和适用范围的研究证据；
- 基因标识映射、资源覆盖度、疾病基因选择和图构建方式都可能影响排序；
- Core 与 Enhanced 使用的证据集合不同，其结果不应直接视为等价；
- 共识排序是证据聚合步骤，不是独立的疗效、毒性或安全性预测模型；
- 所有候选结果都需要进一步的计算、实验和专业评价。

## 测试

开发者可在安装项目测试依赖后运行：

```bash
python -m pytest
python -m compileall -q src scripts tests
python -m ruff check .
docker compose config
```

测试通过或服务健康仅表示对应的软件功能正常，不等同于完整科学流程已经复现。

## 引用

如在研究中使用本项目，请引用项目名称、版本、GitHub 仓库和访问日期：

> smi2phen: An Agent-assisted Multi-perspective Drug Screening Workflow. Version 0.1.0.
> https://github.com/beiweiClover/smi2phen

当前项目没有已确认的 DOI，请勿推断或虚构 DOI。

## 许可证

smi2phen 源代码采用 [MIT License](LICENSE)。第三方科研资源仍受其各自许可证和来源条款
约束；本项目的源代码许可证不会覆盖第三方资源的再分发条件。
