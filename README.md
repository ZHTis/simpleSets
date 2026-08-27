# sEEG 回归通道验证器

这个小工具直接读取 `readGripData` 的 `FeaturePool.save()` 输出，对每个指定通道分别运行
OLS、Ridge、Lasso、Elastic Net、广义线性、概率、自回归或非线性模型，预测握力或其它连续标签，
并用**整段 trial 留出**的交叉验证评价通道。

它回答的是：在当前特征、时间段和 trial 分布下，这个通道是否含有可跨 trial 泛化的预测信息。
它不直接证明因果关系或临床意义。

## 为什么这样切分

sEEG 和握力都是强时间自相关信号。随机拆分相邻窗口会让训练集和测试集近乎重复，产生过高分数。
这里使用 `GroupKFold(trial_key)`。显著性零假设采用每个 trial 内循环移位标签，尽量保留标签的时间自相关。

## 安装

```bash
cd ~/Documents/simpleSets
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
pytest
```

## 输入

特征目录应包含：

```text
features.npz
labels.parquet
windows.parquet
feature_names.json
manifest.json              # 推荐保留，解码器本身不强制读取
```

坐标 CSV 以 `channel` 为必需键。只要提供了坐标表，所有待测通道都必须匹配，避免通道顺序静默错位：

```csv
channel,x,y,z,hemisphere
A1,-32.1,-18.4,42.0,L
```

时间标签 CSV 使用半开区间 `[start_s, end_s)`；时间对应 `windows.parquet` 的
`label_time_s`，并按 `trial_key` 匹配：

```csv
trial_key,start_s,end_s,include,label
R11_trial-001,2.0,8.0,true,active_flight
```

## 模型分支

所有分支共享相同的数据筛选、训练折内预处理、`GroupKFold` 和置换检验，只改变回归器：

| 顺序 | `--model` | 目的 | 起始参数 |
|---:|---|---|---|
| 1 | `ols` | 无正则化基线 | 不使用 `alpha` |
| 2 | `ridge` | 稳定相关特征的系数 | `--alpha 1.0` |
| 3 | `lasso` | 稀疏选择特征 | `--alpha 0.01` |
| 4 | `elasticnet` | 稀疏选择并保留相关特征组 | `--alpha 0.01 --l1-ratio 0.5` |
| 5 | `glm` | 按目标分布建立广义线性模型 | `--glm-family normal --alpha 1.0` |
| 6 | `spline` | 可解释的逐特征平滑非线性（加性模型） | `--spline-knots 5 --spline-degree 3` |
| 7 | `tree` | 可读阈值规则与特征交互 | `--tree-max-depth 3 --tree-min-samples-leaf 20` |
| 8 | `bayesian` | 概率回归；输出预测均值和不确定性 | 默认 Bayesian Ridge |
| 9 | `autoregressive` | EEG特征加同一Trial内历史力值的一步预测 | `--ar-lags 3 --alpha 1.0` |
| 10 | `mlp` | 简单全连接神经网络回归 | `--mlp-hidden-layers 32` |

`alpha` 的起始值只用于首次跑通，不是最终答案。Ridge 和 Lasso/Elastic Net 在 scikit-learn
中的损失缩放不同，因此不能把相同 `alpha` 当作完全公平的强度比较。正式实验应在训练折内部选择参数。

### GLM 的分布和目标值域

`glm` 使用 `--glm-family` 选择响应分布：

| family | Tweedie power | 目标要求 | 常见用途 |
|---|---:|---|---|
| `normal` | 0 | 任意实数 | 连续、近似对称的目标 |
| `poisson` | 1 | 非负 | 计数或非负偏态目标 |
| `gamma` | 2 | 严格正 | 正值、右偏连续目标 |
| `inverse_gaussian` | 3 | 严格正 | 正值且强右偏的连续目标 |
| `tweedie` | `--glm-power` | 依 power 而定 | 自定义均值-方差关系 |

默认 `--glm-link auto`：Normal 使用 identity，其余上述 family 使用 log。Poisson 并不会自动把连续握力
变成计数数据；应根据目标分布而不是模型名称选择 family。若 `force_normalized` 含负值，先使用
`normal`，或采用有科学含义且仅在训练流程内拟合的目标变换。

### 可解释非线性模型

`spline` 对每个输入特征分别建立 B-spline 基函数，再用 Ridge 拟合；没有自动加入特征间交互，
因此可将每个特征的平滑效应单独画出。`tree` 会学习阈值规则，可直接导出规则，但深树容易过拟合，
建议保持较小的 `--tree-max-depth` 并设置足够大的 `--tree-min-samples-leaf`。

### 概率、自回归与神经网络模型

`bayesian` 使用 Bayesian Ridge。`oof_predictions.parquet` 除预测均值外还包含
`y_pred_std`；`channel_summary.csv` 包含预测标准差汇总和95%预测区间覆盖率。它是线性概率模型，
适合先做可扩展的不确定性基线，不等同于计算量更大的高斯过程。

`autoregressive` 是一步 ARX：在当前窗口的EEG特征后追加同一 Trial 内过去 `--ar-lags`
个真实目标值。Trial开头缺少的滞后值只用训练折统计量填充，绝不跨 Trial。该评估属于
`teacher_forced_one_step`，因此回答“已知近期真实力时能否预测下一窗口”，不是完全自由滚动预测。

`mlp` 使用小型 `MLPRegressor`。隐藏层由 `--mlp-hidden-layers 32,16` 指定，默认最多迭代
`--mlp-max-iter 500`轮；所有缺失值填充、
标准化和网络拟合都发生在训练折内。神经网络容量更高，仍应使用Trial留出验证。

## 分别运行

先把公共参数保存成便于复制的命令形式，然后依次使用不同输出目录：

### 1. OLS 基线

```bash
seeg-validate-channels /path/to/feature_pool /path/to/results/01_ols \
  --model ols \
  --coordinates examples/channel_coordinates.csv \
  --time-labels examples/time_labels.csv \
  --channels A1,A2 --target force_normalized --folds 5 --permutations 200
```

### 2. Ridge

```bash
seeg-validate-channels /path/to/feature_pool /path/to/results/02_ridge \
  --model ridge --alpha 1.0 \
  --coordinates examples/channel_coordinates.csv \
  --time-labels examples/time_labels.csv \
  --channels A1,A2 --target force_normalized --folds 5 --permutations 200
```

### 3. Lasso

```bash
seeg-validate-channels /path/to/feature_pool /path/to/results/03_lasso \
  --model lasso --alpha 0.01 \
  --coordinates examples/channel_coordinates.csv \
  --time-labels examples/time_labels.csv \
  --channels A1,A2 --target force_normalized --folds 5 --permutations 200
```

### 4. Elastic Net

```bash
seeg-validate-channels /path/to/feature_pool /path/to/results/04_elasticnet \
  --model elasticnet --alpha 0.01 --l1-ratio 0.5 \
  --coordinates examples/channel_coordinates.csv \
  --time-labels examples/time_labels.csv \
  --channels A1,A2 --target force_normalized --folds 5 --permutations 200
```

### 5. GLM（示例：Gamma）

```bash
seeg-validate-channels /path/to/feature_pool /path/to/results/05_glm_gamma \
  --model glm --glm-family gamma --glm-link auto --alpha 1.0 \
  --coordinates examples/channel_coordinates.csv \
  --time-labels examples/time_labels.csv \
  --channels A1,A2 --target force_normalized --folds 5 --permutations 200
```

### 6. 样条加性回归

```bash
seeg-validate-channels /path/to/feature_pool /path/to/results/06_spline \
  --model spline --spline-knots 5 --spline-degree 3 --alpha 1.0 \
  --coordinates examples/channel_coordinates.csv \
  --time-labels examples/time_labels.csv \
  --channels A1,A2 --target force_normalized --folds 5 --permutations 200
```

### 7. 浅层回归树

```bash
seeg-validate-channels /path/to/feature_pool /path/to/results/07_tree \
  --model tree --tree-max-depth 3 --tree-min-samples-leaf 20 \
  --coordinates examples/channel_coordinates.csv \
  --time-labels examples/time_labels.csv \
  --channels A1,A2 --target force_normalized --folds 5 --permutations 200
```

### 8. Bayesian Ridge概率回归

```bash
seeg-validate-channels /path/to/feature_pool /path/to/results/08_bayesian \
  --model bayesian --folds 5 --permutations 200
```

### 9. 一步自回归（ARX）

```bash
seeg-validate-channels /path/to/feature_pool /path/to/results/09_autoregressive \
  --model autoregressive --ar-lags 3 --alpha 1.0 --folds 5 --permutations 200
```

### 10. 简单神经网络回归

```bash
seeg-validate-channels /path/to/feature_pool /path/to/results/10_mlp \
  --model mlp --mlp-hidden-layers 32 --mlp-activation relu \
  --mlp-alpha 0.0001 --mlp-learning-rate 0.001 --mlp-max-iter 500 \
  --folds 5 --permutations 200
```

## 通用运行示例

```bash
seeg-validate-channels \
  /path/to/feature_pool \
  /path/to/decoder_results \
  --coordinates examples/channel_coordinates.csv \
  --time-labels examples/time_labels.csv \
  --channels A1,A2 \
  --features lmp,bandpower_13_30Hz,bandpower_60_150Hz \
  --target force_normalized \
  --model ridge \
  --mask mask_flight \
  --folds 5 \
  --alpha 1.0 \
  --permutations 200
```

若时间 CSV 已经精确限定范围且不希望再叠加 `mask_flight`，使用 `--mask none`。

输出包括：

- `channel_summary.csv`：每通道 Pearson r、R²、MAE、循环移位置换 p 值、跨通道 BH-FDR q 值及坐标；
- `oof_predictions.parquet`：所有留出预测，便于画真实值与预测值；
- `run_config.json`：运行参数，便于复现。

建议同时关注 `pearson_r`、`r2` 与 `permutation_p`。相关系数高但 R² 为负，通常表示趋势相关、
但幅值或偏置的泛化仍然很差。`permutation_q_fdr_bh` 已对本次所有待测通道做多重比较校正；
可将 `q < 0.05`、正的 Pearson r 以及可接受的 R² 共同作为候选标准，而不要只看一个阈值。
