# 基于机器学习的社交网络用户行为特征化研究

这是一个面向大学生创新创业训练计划的机器学习项目，主题是“基于机器学习对社交网络用户行为特征化研究”。项目把社交网络中的用户互动抽象为图结构，尝试用图特征、链路预测和事件传播分类来描述用户行为模式。

当前版本主要用于中期检查和后续继续开发，已经包含可运行代码、实验图表和中期材料。

## 研究任务

1. 用户互动预测：把用户看作节点，把转发、评论、提及等互动看作边，训练模型预测两名用户之间未来发生互动的概率。
2. 事件传播识别：把一次事件的传播链路看作一张小图，提取传播深度、扩散宽度、增长速度等结构特征，判断它更接近普通事件还是突发事件。

代码分为两层：

- `src/social_behavior_ml/`：正式 PyTorch 实现，包含图卷积链路预测模型和事件传播分类模型。
- `examples/run_numpy_demo.py`：轻量演示脚本，只依赖 `numpy` 和 `matplotlib`，用于在没有 PyTorch 的电脑上先跑出指标和图片。

## 当前阶段成果

在模拟社交网络和模拟事件传播数据上完成了初步验证：

| 任务 | 方法 | 指标 |
|---|---|---|
| 用户互动链路预测 | 图结构特征 + Logistic Regression 基线 | AUC 0.616，AP 0.651 |
| 事件传播分类 | 传播结构特征 + Logistic Regression 基线 | Accuracy 1.000，AUC 1.000 |

说明：事件分类指标较高，是因为当前模拟数据中普通事件和突发事件的结构差异设置较明显。后续接入真实公开数据后，需要重新训练和评估。

## 参考的 GitHub 项目思路

本项目没有直接复制外部仓库代码，主要借鉴了以下公开项目的任务设定和工程思路：

- [PyTorch Geometric link_pred.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/link_pred.py)：参考其中的训练/验证/测试划分、图编码器和边打分思路。
- [DGL GraphBolt link_prediction.py](https://github.com/dmlc/dgl/blob/master/examples/graphbolt/link_prediction.py)：参考 GraphSAGE 在大图链路预测中的负采样与端到端训练流程。
- [TianBian95/BiGCN](https://github.com/TianBian95/BiGCN)：参考“传播结构可以用于事件/谣言识别”的建模角度。
- [JihoChoi/dynamic-gcn](https://github.com/JihoChoi/dynamic-gcn)：参考动态传播图和注意力机制在社交媒体事件检测中的思路。
- [safe-graph/GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews)：参考把社交内容检测问题组织成图分类任务的工程写法。
- social-network-link-prediction 类仓库中的传统图特征：共同邻居、Jaccard、Preferential Attachment 等可以作为可解释基线。

## 快速运行

如果只想先看演示结果，运行：

```powershell
python examples/run_numpy_demo.py --out runs/demo
```

运行后会得到：

- `runs/demo/metrics.json`：链路预测和事件分类的指标。
- `runs/demo/link_prediction_scores.png`：正负样本得分分布。
- `runs/demo/event_feature_scatter.png`：事件结构特征可视化。

如果电脑已经安装 PyTorch，可以运行正式模型：

```powershell
pip install -e .
python -m social_behavior_ml.train_link_prediction --out runs/link_prediction
python -m social_behavior_ml.train_event_classifier --out runs/event_classifier
```

如果没有安装 PyTorch，也可以只安装轻量演示所需依赖：

```powershell
pip install numpy matplotlib
python examples/run_numpy_demo.py --out runs/demo
```

## 数据格式

项目支持两种使用方式：

1. 没有真实数据时，代码会自动生成一个带社区结构的模拟社交网络，用来完成中期展示。
2. 有真实数据时，把边列表整理成 CSV：

```csv
source,target,timestamp,event_id,action
u001,u019,2026-01-04 12:05:00,e01,retweet
u019,u027,2026-01-04 12:08:00,e01,comment
```

其中 `source` 和 `target` 是用户编号，`timestamp`、`event_id`、`action` 可以暂时为空；链路预测只需要前两列。

## 项目结构

```text
social_behavior_ml_project/
  README.md
  requirements.txt
  pyproject.toml
  data/README.md
  docs/
    中期检查表_已填写.docx
    项目说明与代码导读.docx
  examples/run_numpy_demo.py
  runs/demo/
    metrics.json
    link_prediction_scores.png
    event_feature_scatter.png
  src/social_behavior_ml/
    __init__.py
    data.py
    features.py
    metrics.py
    models.py
    train_link_prediction.py
    train_event_classifier.py
```

## 后续计划

1. 接入公开真实社交网络数据集，替换当前模拟数据。
2. 比较传统图特征、GCN、GraphSAGE、GAT 等模型在链路预测任务上的表现。
3. 在事件传播识别中加入时间序列特征和文本特征，减少只依赖结构特征造成的偏差。
4. 补充消融实验和误差分析，为结题报告提供更可靠的实验依据。
