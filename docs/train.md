# 训练MASR模型

MASR基于pytorch，`MASRModel`是`torch.nn.Module`的子类。这将给熟悉`pytorch`的用户带来极大的方便。

使用MASR的训练功能需要安装以下额外的依赖，既然你浏览到了这里，这些依赖你一定能自行搞定！

* `levenshtein-python`

  计算CER中的编辑距离

* `warpctc_pytorch`

  百度的高性能CTC正反向传播实现的pytorch接口

* `tqdm`

  进度显示

* `tensorboardX`

  为pytorch提供tensorboard支持

* `tensorboard`

  实时查看训练曲线

当然，相信你也有GPU，否则训练将会变得很慢。

**通常，神经网络的训练比搭建要困难得多，然而MASR为你搞定了所有复杂的东西，使用MASR进行训练非常方便。**

如果你只想要使用MASR内置的门卷积网络`GatedConv`的话，首先初始化一个`GatedConv`对象。

```python
from models.conv import GatedConv

model = GatedConv(vocabulary)
```

你需要传入向它`vocabulary`，这是一个字符串，包含你的数据集中所有的汉字。但是注意，`vocabulary[0]`应该被设置成一个无效字符，用于表示CTC中的空标记。

之后，使用`to_train`方法将`model`转化成一个可以训练的对象。

```python
model.to_train()
```

此时`model`则变成可训练的了，使用`fit`方法来进行训练。

```python
model.fit('train.index', 'dev.index', epoch=10)
```

`epoch`表示你想要训练几次，而`train.index`和`dev.index`应该分别为训练数据集和开发数据集（验证集或测试集）的索引文件。

索引文件应具有如下的简单格式：

```python
/path/to/audio/file0.wav,我爱你
/path/to/audio/file1.wav,你爱我吗
...
```

左边是音频文件路径，右边是对应的标注，用逗号（英文逗号）分隔。

`model.fit`方法还包含学习率、batch size、梯度裁剪等等参数，可以根据需要调整，建议使用默认参数。

完整的训练流程参见[train.py](/examples/train.py)。