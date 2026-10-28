# CausalQuery 主模型核心代码

这是依据论文《CausalQuery: Uncovering Evidence Dependence Across Web Providers through Coordinated Interventions》第 3 节及附录 A.1 实现的独立 Python 工程。代码包含共享随机证据干预、支持规则执行、成对响应特征提取、逻辑回归依赖估计、功能依赖真值计算、缺失响应处理与去冗余证据排序。

本精简版本**不包含测试、合成数据、演示样例、预训练演示权重和演示结果**。它也不是论文作者公开仓库的逐文件副本。需要使用实际数据训练后才能正常推理；不能用缺失的合成演示权重直接调用 `infer`。

## 安装

要求 Python >= 3.10。

```bash
python -m pip install -e .
```

## 核心结构

| 路径 | 功能 |
| --- | --- |
| `configs/default.yaml` | 干预、统计模型、超参数和重排序配置 |
| `src/causalquery/cli.py` | 训练、推理、评估及公共响应命令行 |
| `src/causalquery/config.py` | YAML 解析和参数校验 |
| `src/causalquery/hashing.py` | 根据公开种子和规范化来源 ID 生成确定性共享掩码 |
| `src/causalquery/provider.py` | 带私有来源集合的可插拔提供方、支持逻辑和状态响应 |
| `src/causalquery/probe.py` | 多提供方同批次协调探测 |
| `src/causalquery/features.py` | 有效配对、分歧率、零响应概率、协方差及 OR-aware 特征 |
| `src/causalquery/estimator.py` | 模型训练、开发集参数选择及预测 |
| `src/causalquery/oracle.py` | 支持规则的来源影响力估计及监督标签 |
| `src/causalquery/ranking.py` | 依赖惩罚的贪心证据重排序 |
| `src/causalquery/external.py` | 公共探针响应导出、消费者独立评分 |
| `src/causalquery/metrics.py` | 依赖预测指标 |
| `src/causalquery/io.py` | JSON/YAML 数据读写 |

## 输入数据

可使用 YAML 或 JSON。工具化提供方输入顶层为 `claims` 数组；每条 claim 提供 `id`、`split`、`question`、`claim`、`snapshot` 和 `providers`；每个 provider 提供 `id`、`relevance`、`referenced_origins` 和 `support`。`split` 至少应包含 `train`、`dev`，若使用评估命令还应包含 `test`。训练集需要同时包含正负功能依赖样本。

`support` 是确定性的单调布尔规则，支持 `any`、`all`、`at_least`、`any_of`、`all_of`；其中 `at_least` 使用 `count`，嵌套规则使用 `children`。同一原始证据在不同提供方应具有一致的规范化来源 ID。

消费侧也可以输入 `format: causalquery-public-responses-v1` 的公开响应数据：其中包含 claim 基本信息、provider 标识、`probes` 对象及其 `responses`。用于 `fit` 或 `evaluate` 时还必须提供相应的 `pair_labels`（成对 `functional`、`structural` 标签）。不含标注的公开响应可以使用已训练模型直接评分。

## 训练

```bash
python -m causalquery fit --config configs/default.yaml --dataset /path/to/your_dataset.yaml --model outputs/model.json --report outputs/training_report.json
```

## 推理与评估

```bash
python -m causalquery infer --config configs/default.yaml --dataset /path/to/your_dataset.yaml --model outputs/model.json --output outputs/inference.json --split test
python -m causalquery evaluate --config configs/default.yaml --dataset /path/to/labeled_dataset.yaml --model outputs/model.json --output outputs/evaluation.json --split test
```

## 提供方和消费者解耦

```bash
python -m causalquery export-responses --config configs/default.yaml --dataset /path/to/your_dataset.yaml --output outputs/public_responses.json
python -m causalquery score-responses --config configs/default.yaml --dataset outputs/public_responses.json --model outputs/model.json --output outputs/scored.json --split test
```

`export-responses` 不输出提供方私有来源集合和支持函数。消费者只依赖公开响应和已训练模型，使用相同有效配对样本估计证据依赖。`status=unavailable` 不被误认为 `status=ok, bit=0`。

## 公式与边界

协议以 `M_z(r)=1[h(z,r)<p]` 对所有提供方的相同来源施加一致保留掩码。成对特征包含协方差、响应签名分歧率与带可用性门控的 OR 重叠反演，并由开发集选择超参数拟合逻辑回归概率。证据排序的边际效用为 `relevance_i - lambda * max(previous_dependence)`。

模型判定的是支持函数的**功能性来源重叠**，不是事实真假，也不等价于对任意布尔规则的结构来源交集识别。有限探针、过低的零响应频率、来源 ID 不一致和相关缺失均可能造成不可识别或偏差。输出中对不可定义的反演保留未定义状态，并不擅自置零。

论文使用的 AVeriTeC 数据、作者特定的提供方服务和固定验证 backbone 不在本精简代码库中，故论文表格中的结果并不随此项目自动复现。
