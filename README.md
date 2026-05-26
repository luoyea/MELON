# MELON-MQ: 面向多模态推荐的模态质量感知实验项目

本项目基于 MELON: Learning Multi-Aspect Modality Preferences for Accurate Multimedia Recommendation 进行复现与扩展，主要用于 MenClothing 数据集上的多模态推荐实验。

在原始 MELON 模型基础上，本项目加入了模态质量感知与多方面动态融合模块，并配套完成了消融实验、噪声鲁棒性实验、日志记录和结果汇总。

## 项目改动

- 在 `codes/modules/modality_quality.py` 中新增：
  - `ModalityQualityScorer`：根据用户模态表示、物品模态表示和原始模态特征估计图像/文本质量分数。
  - `AspectDynamicGate`：按 aspect 学习图像与文本的动态融合权重。
  - `AspectFusion`：基于 aspect 级权重融合图像/文本匹配得分。
- 在 `codes/Models.py` 中接入 MQ 分支，将 `s_mq` 加入原 BPR 打分，并加入门控熵正则。
- 在 `codes/main.py` 中加入：
  - 特征噪声注入，用于鲁棒性实验。
  - epoch 级指标、summary、gate 权重统计和全局实验汇总输出。
  - best validation checkpoint 保存和最终 test 评估。
- 在 `codes/data/create_tiny_dataset.py` 中加入小规模数据集构造脚本，便于快速实验。

## 目录结构

```text
MELON/
├── environment.yml              # GPU 环境
├── environment-cpu.yml          # CPU 环境
├── README.md
└── codes/
    ├── main.py                  # 训练、验证、测试入口
    ├── Models.py                # MELON 主模型与 MQ 扩展
    ├── modules/
    │   └── modality_quality.py  # MQM / AQG / AspectFusion
    ├── utility/                 # 参数、数据加载、评估指标
    ├── data/
    │   ├── build_data.py
    │   ├── create_tiny_dataset.py
    │   ├── MenClothing/
    │   └── MenClothing_tiny/
    ├── logs/                    # 实验日志与 CSV 结果
    └── models/                  # best validation checkpoint
```

## 环境配置

GPU 版本：

```bash
conda env create -f environment.yml
conda activate melon
```

CPU 版本：

```bash
conda env create -f environment-cpu.yml
conda activate melon
```

主要依赖包括 Python 3.8、PyTorch 1.10.2、PyG 2.0.3、gensim 3.8.3、sentence-transformers 2.2.0、numpy、pandas、scipy 等。

## 数据说明

当前项目主要使用：

- `codes/data/MenClothing/`：原始 MenClothing 实验数据。
- `codes/data/MenClothing_tiny/`：从 MenClothing 切分出的快速实验数据，包含重新映射后的交互、图像特征、文本特征和 `item_id_map.json`。

如需重新生成 tiny 数据集，可在 `codes` 目录下运行：

```bash
python data/create_tiny_dataset.py --src MenClothing --dst MenClothing_tiny --n_users 500
```

## 运行方式

所有命令建议在 `codes` 目录下执行。

原始 MELON baseline：

```bash
python main.py --dataset MenClothing_tiny --model_name MELON_base_tiny_e30 --epoch 30 --use_mqm 0 --use_aqg 0 --eta_mq 0
```

加入 MQM + AQG 的完整模型：

```bash
python main.py --dataset MenClothing_tiny --model_name MELON_MQ_A4_tiny_e30 --epoch 30 --use_mqm 1 --use_aqg 1 --fixed_weight 0 --n_aspects 4 --eta_mq 0.2 --lambda_ent 1e-3 --q_lambda 1.0
```

固定权重消融：

```bash
python main.py --dataset MenClothing_tiny --model_name ab_fixed_weight_tiny_e30 --epoch 30 --use_mqm 1 --use_aqg 1 --fixed_weight 1 --n_aspects 4
```

关闭 MQM：

```bash
python main.py --dataset MenClothing_tiny --model_name ab_wo_mqm_tiny_e30 --epoch 30 --use_mqm 0 --use_aqg 1 --fixed_weight 0 --n_aspects 4
```

关闭 AQG：

```bash
python main.py --dataset MenClothing_tiny --model_name ab_wo_aqg_tiny_e30 --epoch 30 --use_mqm 1 --use_aqg 0 --fixed_weight 0 --n_aspects 4
```

方面数消融：

```bash
python main.py --dataset MenClothing_tiny --model_name ab_A1_tiny_e30 --epoch 30 --n_aspects 1
python main.py --dataset MenClothing_tiny --model_name ab_A2_tiny_e30 --epoch 30 --n_aspects 2
python main.py --dataset MenClothing_tiny --model_name ab_A8_tiny_e30 --epoch 30 --n_aspects 8
```

噪声鲁棒性实验：

```bash
python main.py --dataset MenClothing_tiny --model_name noise01_tiny_e30 --epoch 30 --noise_ratio 0.1 --noise_mode gaussian
python main.py --dataset MenClothing_tiny --model_name noise03_tiny_e30 --epoch 30 --noise_ratio 0.3 --noise_mode gaussian
python main.py --dataset MenClothing_tiny --model_name noise05_tiny_e30 --epoch 30 --noise_ratio 0.5 --noise_mode gaussian
```

## 关键参数

| 参数 | 说明 |
| --- | --- |
| `--dataset` | 数据集名称，例如 `MenClothing_tiny` |
| `--model_name` | 本次实验名称，同时用于日志和模型文件命名 |
| `--use_mqm` | 是否启用模态质量评分模块 |
| `--use_aqg` | 是否启用 aspect 级动态门控 |
| `--fixed_weight` | 是否使用图像/文本 0.5/0.5 固定权重 |
| `--n_aspects` | aspect 数量，需整除 `feat_embed_dim` |
| `--eta_mq` | MQ 分支得分权重 |
| `--lambda_ent` | 门控熵正则权重 |
| `--q_lambda` | 质量分数对门控 logits 的缩放系数 |
| `--noise_ratio` | 注入噪声的物品比例 |
| `--noise_mode` | 噪声类型，支持 `none`、`gaussian`、`zero`、`shuffle` |

## 输出文件

每次实验会在 `codes/logs/` 和 `codes/models/` 下生成结果文件。

```text
logs/
├── all_experiments_summary.csv
├── {dataset}_{model_name}_run.log
├── {dataset}_{model_name}_epoch_metrics.csv
├── {dataset}_{model_name}_summary.csv
└── {dataset}_{model_name}_gate_stats.csv

models/
└── {dataset}_{model_name}
```

其中：

- `run.log` 保存终端训练输出。
- `epoch_metrics.csv` 保存验证集 epoch 级指标。
- `summary.csv` 保存单次实验的 best validation 和 final test 指标。
- `all_experiments_summary.csv` 汇总所有实验。
- `gate_stats.csv` 保存图像/文本平均门控权重与质量分数。
- `models/{dataset}_{model_name}` 保存 best validation recall@20 对应的模型参数。

## 当前已有实验

当前 `codes/logs/all_experiments_summary.csv` 中已汇总的实验包括：

- baseline：`MELON_base_tiny_e30`
- 完整模型：`MELON_MQ_A4_tiny_e30`
- 消融实验：`ab_wo_mqm_tiny_e30`、`ab_wo_aqg_tiny_e30`、`ab_fixed_weight_tiny_e30`
- aspect 数量实验：`ab_A1_tiny_e30`、`ab_A2_tiny_e30`、`ab_A8_tiny_e30`
- 噪声鲁棒性实验：`noise01_tiny_e30`、`noise03_tiny_e30`、`noise05_tiny_e30`
- 参数调节实验：`tune1_eta005_ent5e3_ql03`、`tune2_eta010_ent1e2_ql03`、`tune3_eta005_ent1e2_ql01`

## 参考

本项目基于原 MELON 代码修改：

> MELON: Learning Multi-Aspect Modality Preferences for Accurate Multimedia Recommendation  
> Dongho Jeong, Taeri Kim, Donghyun Cho and Sang-Wook Kim  
> SIGIR 2025

原始代码结构参考了 MONET 项目。
