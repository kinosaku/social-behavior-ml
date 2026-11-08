# 微博式合成级联实验

每条记录是一棵有向传播树（directed propagation tree）。节点（node）表示帖子，边（edge）由来源帖子指向转发帖。数据由模拟器（simulator）生成，32 维节点向量（node vector）由数值分布采样，类别由生成器指定。数据没有真实微博文本、用户编号或时间戳（timestamp）。

| 文件或字段 | 内容 |
|---|---|
| `train.pt` / `val.pt` / `test.pt` | 14,453 / 3,097 / 3,098 个样本，三个集合都带标签 |
| `x` | 每个节点一行，形状为 `[N, 32]` |
| `edge_index` | 第一行是来源，第二行是转发节点，形状为 `[2, N-1]` |
| `label` | 图级标签（graph-level label），`0` 正常，`1` 突发 |
| `manifest.json` | 原始数据的生成参数、字段和类别计数 |

## 运行

从仓库根目录执行，Python 3.10 或以上。实验实际使用的库版本记录在 [metrics.json](../../results/weibo_cascades/metrics.json) 中。

```bash
python -m pip install -r experiments/weibo_cascades/requirements.txt
python experiments/weibo_cascades/generate_dataset.py --out experiments/weibo_cascades/data
python experiments/weibo_cascades/run_experiments.py
```

如果已经有上传时使用的三份 `.pt`，可以跳过生成，运行时加 `--data-dir 路径`，用 `--out 路径` 指定结果目录。程序按样本顺序计算内容指纹（content fingerprint），可和仓库中的 `metrics.json` 对照。内容指纹比较节点向量、父节点关系和标签，允许序列化（serialization）容器本身有差异。

默认运行两个任务。只跑其中一个时可加 `--task event` 或 `--task link`，并分别指定输出目录。

## 怎么做实验

- **事件分类（event classification）**。以一张完整传播树为单位。对比多数类、只看节点数、七项图特征（graph features）、32 维节点向量均值、二者拼接、图同构网络（GIN）。逻辑回归（logistic regression）用训练集拟合标准化（standardization）和参数。GIN 用验证集的宏平均 F1（macro-F1）挑选轮次，所有分类器使用固定的 0.5 判定阈值（decision threshold）。
- **转发来源预测（repost parent prediction）**。对每张树先观察最早的 70% 节点，向上取整。其余节点按编号依次到达，在所有此前节点中排序，打分完成后才揭示该条真实边。逻辑回归每次用一个真实父节点和最多八个随机候选训练，验证和测试时比较所有候选。候选特征包括根标记、已有出度（out-degree）、深度（depth）、相对顺序间隔和对数顺序间隔。并列排名按均匀随机打破的期望计分。
- **数据划分（data split）**。两个任务沿用原有训练、验证和测试文件，整张级联属于同一个集合。链路实验的测试流程允许在预测下一条边前观察已经到达的真实边，模拟逐条观测的设置。

运行后在输出目录生成 `metrics.json`、`event_test_predictions.csv`、`link_test_predictions.csv` 和 `gin_model.pt`。两个 CSV 的 `graph_index` 都是对应测试文件里的零基序号。

本实验衡量模型在该模拟器（simulator）生成的新级联上的表现。要研究跨事件的用户互动，需要能跨事件匹配的用户编号和真实互动记录。要衡量真实事件预警效果，需要真实时间戳、观测截止点和独立标签。
