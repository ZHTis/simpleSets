# sEEG 线性回归通道验证器

这个小工具直接读取 `readGripData` 的 `FeaturePool.save()` 输出，对每个指定通道分别运行
OLS、Ridge、Lasso 或 Elastic Net，预测握力或其它连续标签，并用**整段 trial 留出**的交叉验证评价通道。

它回答的是：在当前特征、时间段和 trial 分布下，这个通道是否含有可跨 trial 泛化的线性信息。
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

## 四个实验分支

四个分支共享完全相同的数据筛选、训练折内标准化、`GroupKFold` 和置换检验，只改变回归器：

| 顺序 | `--model` | 目的 | 起始参数 |
|---:|---|---|---|
| 1 | `ols` | 无正则化基线 | 不使用 `alpha` |
| 2 | `ridge` | 稳定相关特征的系数 | `--alpha 1.0` |
| 3 | `lasso` | 稀疏选择特征 | `--alpha 0.01` |
| 4 | `elasticnet` | 稀疏选择并保留相关特征组 | `--alpha 0.01 --l1-ratio 0.5` |

`alpha` 的起始值只用于首次跑通，不是最终答案。Ridge 和 Lasso/Elastic Net 在 scikit-learn
中的损失缩放不同，因此不能把相同 `alpha` 当作完全公平的强度比较。正式实验应在训练折内部选择参数。

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
