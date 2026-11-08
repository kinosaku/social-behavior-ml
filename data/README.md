# 数据说明

本项目默认使用合成数据完成算法验证。合成数据不是为了替代真实社交平台数据，而是为了在没有公开授权数据时先验证方法是否能跑通。

若后续拿到真实数据，建议整理为如下边列表：

```csv
source,target,timestamp,event_id,action
u001,u019,2026-01-04 12:05:00,e01,retweet
u019,u027,2026-01-04 12:08:00,e01,comment
```

字段含义：

- `source`：发起互动或转发的用户。
- `target`：被互动或上游传播用户。
- `timestamp`：互动时间，可选。
- `event_id`：所属事件，可选。
- `action`：互动类型，如 retweet/comment/mention，可选。


## 传播级联实验数据

20,648 张模拟传播树（synthetic propagation trees）的生成脚本和字段说明在 [级联实验目录](../experiments/weibo_cascades/README.md)。上面的用户边列表适用于原有用户互动演示，传播级联实验读取 PyTorch 的 `.pt` 文件。
