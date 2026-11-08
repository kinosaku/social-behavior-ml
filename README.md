# 从转发树识别突发事件

给模型一棵转发树（repost tree），让它判断这次传播属于**正常事件**还是**突发事件**。

[中文讲解 PDF](docs/weibo_cascade_guide.pdf) · [English guide](docs/weibo_cascade_guide_en.pdf) · [中文技术报告 PDF](docs/weibo_cascade_technical_report_zh.pdf) · [完整实验结果](results/weibo_cascades/metrics.json)

## 怎么判断

一条微博发出后，有人直接转发原帖，也有人沿着已有转发继续传播。把这些关系连起来，就得到一棵树。每个节点（node）是一条帖子，每条边（edge）从来源帖指向转发帖。

1. **看形状。** 数一数传播接力了几层、最宽的一层有多少帖子、有多少转发直接来自原帖。
2. **比较模型。** 用逻辑回归（logistic regression）分别比较图的形状、帖子向量（post vectors），以及两者合用的效果。再用图同构网络（GIN）从整棵树学习。
3. **用没见过的树检验。** 按传播树划分训练集、验证集和测试集。

## 跑出来什么

图结构与向量合用的逻辑回归（logistic regression）在测试集上表现最好。

| 测试集 | 准确率（accuracy） | 宏平均 F1（macro-F1） | 突发样本识别数 |
|---:|---:|---:|---:|
| 3,098 棵树 | **82.25%** | **0.794** | 696 / 1,033 |

宏平均 F1（macro-F1）给正常和突发两类同等权重。只看准确率的话，把每棵树都判成正常也能得到约 67%，却会漏掉所有突发样本。

实验使用 **20,648 棵模拟传播树**，分类时给模型整棵树。实际传播速度需要时间戳（timestamps）。以上分数衡量模拟条件下的完整传播分类表现。

## 从头运行

使用 Python 3.10 或以上版本，在仓库根目录执行。

```bash
python -m pip install -r requirements.txt
cd experiments/weibo_cascades
python generate_dataset.py --out data
python run_experiments.py
```

如果已有 `train.pt`、`val.pt` 和 `test.pt`，跳过生成步骤，在最后一条命令后加 `--data-dir 数据目录`。运行结果在 `experiments/weibo_cascades/run/`，包括评估指标、逐图预测和模型权重（model weights）。
