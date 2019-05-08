# 中文普通话自动语音识别

## 简介

这是一个**开箱即用**的中文普通话语音识别工具。

它的设计原则是：

* **开箱即用**：只需要**不到5分钟**，你就可以使用本项目完成第一次中文语音识别
* **接口简洁**
* **模块化，可扩展**

## 5分钟上手

1. 克隆本项目到本地

   ```sh
   git clone https://github.com/lukhy/masr.git
   ```

2. 从[这里](https://pan.baidu.com/s/15Up5wxQ6zeitsrqbpovImg)（**提取码：rv7m**）下载**预训练模型**，并将它拷贝到项目的`pretrained`目录下

   ```sh
   mkdir masr/pretrained
   cp ~/Downloads/wav2letter.pth masr/pretrained/
   cd masr
   ```

3. 安装依赖

   ```sh
   pip install -r requirements.txt
   ```

4. 安装`pyaudio`

   macOS

   ```sh
   brew install portaudio
   pip install pyaudio
   ```

   Ubuntu

   ```sh
   sudo apt-get install python-pyaudio python3-pyaudio
   pip install pyaudio
   ```

   Windows

   ```sh
   pip install pyaudio
   ```

5. 运行示例

   ```sh
   python examples/local-recognize.py
   ```

   看到提示**录音中**后，开始说话，录音持续5秒。然后你会看到识别结果。

## API是怎样的？

不妨看一下`examples/local-recognize.py`文件，你会发现API十分简单。

你只需要加载一个预训练模型（目前仅有Wav2letter），然后调用这个模型的`predict`方法的就行了。`predict`只有一个参数——你想识别的`wav`文件的路径。

```python
from models.wav2letter import Wav2letter

model = Wav2letter.load("pretrained/wav2letter.pth")

model.predict("01.wav") # --> 识别结果（字符串）
```

## 常见问题

#### 是在我自己的电脑上离线识别的吗？

是的，不信你把网断了试试。

#### 我的电脑没有显卡，会不会很慢？

不会。很快。

#### 识别效果差，怎么办？

等下一个版本。

#### 其他问题

**除了以上问题之外**，如果你还有其他**任何**问题，请新建[issue](https://github.com/lukhy/masr/issues/new)让我知道。

