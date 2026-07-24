# 输入文件来源与整理说明

这个文件夹用于放置整个筛选流程的用户输入文件。用户更换研究疾病或化合物库时，通常只需要替换这里的文件，再按模块顺序运行后续 notebook。

## 文件总览

| 文件                                      | 是否必需 | 主要用途                                                            |
| ----------------------------------------- | -------- | ------------------------------------------------------------------- |
| `smiles.csv`                            | 必需     | 输入待筛选化合物库，用于 GPS 药物扰动谱、NetInfer 靶点预测、KG 构图 |
| `steatosis_gene.txt`                    | 必需     | 输入目标疾病相关基因，用于网络邻近度和 KG 疾病节点构建              |
| `TPM_matrix.tsv` / `TPM_matrix_*.tsv` | GPS 必需 | 输入疾病转录组 TPM 矩阵，用于构建疾病 GPS 基准谱                    |
| `metadata.tsv` / `metadata_*.tsv`     | GPS 必需 | 输入 TPM 样本分组信息，用于差异表达分析                             |
| `positive_drugs.tsv`                    | 可选     | 输入已知阳性药物，用于 KG 中连接目标疾病节点                        |
| `disease_links.tsv`                     | 可选     | 输入基础图谱中与目标疾病相关的疾病节点，用于 KG 中增加疾病关联      |
| `sample_metadata.tsv`                   | 可选     | 当前示例的样本来源记录，不直接作为核心流程输入                      |

## `smiles.csv`

### 来源

这个文件来自用户需要筛选的化合物库。例如比赛提供的 TargetMol 化合物库、实验室自有化合物库、商业化合物库或用户整理的候选药物列表。

### 格式要求

必须包含两列：

| 列名       | 含义                              |
| ---------- | --------------------------------- |
| `ID`     | 化合物唯一编号，例如 TargetMol ID |
| `SMILES` | 化合物 SMILES 结构                |

可以额外保留其他列，例如：

| 列名           | 含义                  |
| -------------- | --------------------- |
| `CAS`        | CAS 号                |
| `Formula`    | 分子式                |
| `MolWt`      | 分子量                |
| `parse_mode` | SMILES 整理或解析状态 |

### 整理要求

- `ID` 必须唯一，不能重复。
- `SMILES` 应尽量使用标准化后的 canonical SMILES。
- 如果有盐型、溶剂、水合物，建议保留主体结构；流程中也会用 RDKit 做基础标准化和最大片段选择。
- 无法被 RDKit 解析的 SMILES 会被跳过，后续不会产生可靠预测结果。
- 如果用户化合物和基础图谱中的药物结构一致，KG 构图时会通过标准化 SMILES 匹配，避免重复建立 drug 节点。

## `steatosis_gene.txt`

### 来源

这个文件是目标疾病相关基因集合。用户可以从以下来源整理：

- 文献报道的疾病相关蛋白或基因。
- 差异表达分析得到的疾病上调/下调基因。
- 疾病数据库或靶点数据库中收录的疾病相关基因。
- 专家人工筛选后的机制相关基因。

### 格式要求

这是 TSV 文件，必须包含两列：

| 列名          | 含义           |
| ------------- | -------------- |
| `symbol`    | Gene symbol    |
| `entrez_id` | Entrez Gene ID |

示例：

```text
symbol	entrez_id
IGFBP1	3484
FOS	2353
EGR1	1958
```

### 整理要求

- `symbol` 建议使用 HGNC 标准人类基因名。
- `entrez_id` 必须能和 PPI、GPS、KG 中使用的基因 ID 对应。
- 如果只有 symbol，建议先用 NCBI gene info、mygene、biomaRt 或 org.Hs.eg.db 做 ID 转换。
- 建议去除无法映射到人类 Entrez ID 的条目。

## `TPM_matrix.tsv` / `TPM_matrix_*.tsv`

### 来源

这个文件来自目标疾病相关转录组数据。常见来源包括 GEO、SRA、ArrayExpress、TCGA 或用户自己的 RNA-seq 表达矩阵。

当前示例使用了编号形式：

```text
TPM_matrix_1.tsv
TPM_matrix_2.tsv
```

表示存在两组独立差异分析。流程会分别分析每一组，然后对结果取交集，得到更稳健的疾病变化基准谱。

### 格式要求

这是 TSV 文件：

- 第一列必须是 `GeneID`。
- `GeneID` 当前要求为 Entrez Gene ID。
- 后续每一列是一个样本 ID。
- 表格中的值是 TPM 表达量。

示例结构：

```text
GeneID	GSM000001	GSM000002	GSM000003
3484	12.3	8.4	30.1
2353	5.1	7.6	2.2
```

### 命名规则

如果只有一组差异分析：

```text
TPM_matrix.tsv
metadata.tsv
```

如果有多组差异分析：

```text
TPM_matrix_1.tsv
metadata_1.tsv
TPM_matrix_2.tsv
metadata_2.tsv
```

编号必须一一对应。流程会自动寻找 `TPM_matrix*.tsv`，并匹配对应的 `metadata*.tsv`。

### 整理要求

- TPM 矩阵中的样本列名必须和 metadata 的 `sample_id` 完全一致。
- 每个 gene 建议只保留一行；如果存在重复 Entrez ID，应先合并或保留代表值。
- 矩阵应只包含数值表达量，不要混入注释列。
- 如果原始数据是 count 矩阵，需要先转换为 TPM，或根据需要先完成标准化后再输入。

## `metadata.tsv` / `metadata_*.tsv`

### 来源

这个文件来自转录组数据的样本注释信息。通常从 GEO 样本信息、SRA metadata、论文补充表或用户自己的实验设计表中整理。

### 格式要求

这是 TSV 文件，必须包含两列：

| 列名          | 含义                                   |
| ------------- | -------------------------------------- |
| `sample_id` | 样本 ID，必须和 TPM 矩阵列名一致       |
| `group`     | 分组，只能是`control` 或 `disease` |

示例：

```text
sample_id	group
GSM000001	control
GSM000002	disease
GSM000003	disease
```

### 整理要求

- `group` 只能出现两类：`control` 和 `disease`。
- 同一组差异分析中必须同时包含 control 和 disease。
- 如果一个项目中有多个疾病分期或多个亚型，应先由用户决定哪些样本归为 control，哪些样本归为 disease。
- 如果要做多组差异分析，需要按编号分别保存 metadata。

## `positive_drugs.tsv`

### 来源

这个文件用于输入用户已知的目标疾病阳性药物。来源可以是文献报道、临床指南、数据库证据或用户已有实验结果。

### 格式要求

这是 TSV 文件，必须包含两列：

| 列名           | 含义           |
| -------------- | -------------- |
| `input_type` | 阳性药输入类型 |
| `value`      | 对应的药物标识 |

支持的 `input_type`：

| input_type         | value 写法                         |
| ------------------ | ---------------------------------- |
| `library_id`     | 写`smiles.csv` 里的化合物 `ID` |
| `base_drug_name` | 写基础图谱中已有 drug 节点的名称   |

示例：

```text
input_type	value
library_id	T1558
base_drug_name	Resveratrol
```

### 整理要求

- 如果阳性药在用户化合物库中，优先使用 `library_id`。
- 如果阳性药不在用户化合物库中，但基础图谱已有该药物，可以使用 `base_drug_name`。
- `base_drug_name` 必须能在基础图谱 drug 节点中唯一匹配，否则构图脚本会报错。
- 这个文件是可选文件；如果没有可靠阳性药，可以不提供。

## `disease_links.tsv`

### 来源

这个文件用于输入基础图谱中与目标疾病相关的已有疾病节点。例如目标疾病的上位概念、同义疾病、疾病分期或强相关疾病。

### 格式要求

这是 TSV 文件，必须包含两列：

| 列名           | 含义               |
| -------------- | ------------------ |
| `input_type` | 疾病节点输入类型   |
| `value`      | 对应的疾病节点标识 |

可以额外包含：

| 列名          | 含义                     |
| ------------- | ------------------------ |
| `node_name` | 疾病名称，仅用于人工查看 |

支持的 `input_type`：

| input_type            | value 写法                     |
| --------------------- | ------------------------------ |
| `base_disease_id`   | 基础图谱中的 disease node_id   |
| `base_disease_name` | 基础图谱中的 disease node_name |

示例：

```text
input_type	value	node_name
base_disease_id	disease:mondo_grouped:43693_13209_4790_21104	fatty liver disease
base_disease_name	non-alcoholic steatohepatitis	non-alcoholic steatohepatitis
```

### 整理要求

- 推荐优先使用 `base_disease_id`，因为 ID 比名称更稳定。
- `base_disease_name` 必须能在基础图谱 disease 节点中唯一匹配。
- `node_name` 不参与计算，只是方便人工检查。
- 这个文件是可选文件；如果不希望加入疾病-疾病关联，可以不提供。

## 最小输入组合

如果只运行 KG、NetInfer 和网络邻近度模块，至少需要：

```text
smiles.csv
steatosis_gene.txt
```

如果还要运行 GPS 疾病扰动打分模块，还需要：

```text
TPM_matrix.tsv
metadata.tsv
```

或多组编号文件：

```text
TPM_matrix_1.tsv
metadata_1.tsv
TPM_matrix_2.tsv
metadata_2.tsv
```

如果希望在 KG 中加入已知阳性药或疾病关联，可以额外提供：

```text
positive_drugs.tsv
disease_links.tsv
```

## 通用检查清单

- `smiles.csv` 中 `ID` 唯一，`SMILES` 可被 RDKit 解析。
- `steatosis_gene.txt` 中 `symbol` 和 `entrez_id` 都不为空。
- TPM 矩阵第一列是 Entrez Gene ID，样本列全是表达量。
- metadata 中 `sample_id` 与 TPM 样本列完全一致。
- metadata 中 `group` 只包含 `control` 和 `disease`。
- 多组 TPM/metadata 文件编号一一对应。
- 阳性药和疾病关联文件中的 `input_type` 只使用流程支持的取值。
