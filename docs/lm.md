# 使用语言模型提升识别率

我能理解，当你看到「你好很高兴认识你」被识别成「利好很高性任实米」时，一定会感觉很失望。甚至觉得我在骗你：这怎么可能是Github上效果最好的个人项目？

也许你看到别人的首页上展示的识别效果都是全对的，感觉很厉害。其实它们展示的是测试集上部分样本的效果，测试集跟训练集是同分布的，效果自然不会差。本项目在测试集上的识别率高达85%（不含语言模型的情况下），拿出来一点也不虚。甚至有一些项目展示的是训练集上的效果，这就像高考考做过的题一样，毫无意义。

当然，我说了那么多，并不是要告诉你，只能做成这样了，别再奢求更好的结果了。不！**我们可以把「你好很高兴认识你」识别到全对，还能把测试集上的识别率提升到92%甚至更高。**而我们需要的是一个好的语言模型。

## 什么是语言模型

如你所见，「利好很高性任实米」根本就不是一句通顺的话，但是我们的神经网络并没有意识到这一点。为什么呢？也许你不相信，因为它从来没有人对它说过「你好」这句话。训练集中没有「你好」这句话，也没有「很高兴认识你」。它不知道这些话是合理的句子（至少比「利好很高性任实米」更合理。

而语言模型是通过外部语料库训练得到的，这些语料库包含大量的句子，语言模型知道「你好很高兴认识你」比「利好很高性任实米」更像一个句子。换句话说，神经网络的第一个字输出了「利」，但是「你」的概率也并不低，而语言模型可以纠正它的错误，告诉它，真正该输出的是「你」。

下面我将教你如何一步步加入语言模型，完成后你将会得到一个可以说是惊喜的识别效果。**别担心，不需要写任何代码**。

## 添加语言模型

在你尝试添加语言模型之前，请确认你已经安装了`pyaudio`，参见[识别自己的语音](demo.md)。

同时，你还需要安装Flask，这很简单，`pip install flask`即可。

好了，让我们给本项目加入一个来自百度的语言模型。

一个现成的可以使用的语言模型来自百度，你需要[下载它](https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm)。

这一步很简单，因为从百度下载很快。

下载完成后，执行

```sh
cd masr
mkdir lm/
cp ~/Downloads/zh_giga.no_cna_cmn.prune01244.klm lm/
```

将语言模型拷贝到`masr/lm`目录下。

下面是最麻烦的一步了，做好准备。

我们需要安装这个依赖：[ctcdecode](https://github.com/parlance/ctcdecode)，这是一个高效的beam search解码器。

按照它的README，安装它本来很简单，你需要执行：

```sh
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode
pip install .
```

但可能会遇到报错，原因在于它在安装过程中会去下载两个文件，而这两个文件位于Google的服务器上。你可能需要魔法才能访问。而你的命令行可能不会自动使用魔法。

以下是`build.py`中下载文件部分的代码

```python
# Download/Extract openfst, boost
download_extract('https://sites.google.com/site/openfst/home/openfst-down/openfst-1.6.7.tar.gz',
                 'third_party/openfst-1.6.7.tar.gz')
download_extract('https://dl.bintray.com/boostorg/release/1.67.0/source/boost_1_67_0.tar.gz',
                 'third_party/boost_1_67_0.tar.gz')
```

你需要自行下载这两个文件，并把它们解压到`third_party`目录下，然后注释掉这两行。再次执行上述的安装命令，即可成功安装。

好了，恭喜你，已经完成了所以依赖的安装，现在，启动服务器

```sh
python examples/demo-server.py
```

然后，请将`examples/demo-client.py`中的服务器地址的ip部分改成你的服务器ip，如果你都是在本机上进行的，则不需要修改，使用默认的`localhost`即可。

## 感受喜悦

好了，执行

```sh
python examples/demo-client.py
```

在看到「录音中」的提示后，开始说话。

如果想不到说什么，不妨念「举头望明月，低头思故乡」试试。

接下来，就是见证奇迹的时刻。

```
(python) ➜  masr git:(master) ✗ python examples/demo-client.py
录音中(5s)
..................................................
识别结果:
我想找一个漂亮的女朋友
```

