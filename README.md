# MASR 中文语音识别

**MASR**是一个**开箱即用**的中文普通话语音识别工具。

MASR的特点是：

* **开箱即用**：只需要**不到5分钟**，你就可以使用本项目完成第一次中文语音识别
* **使用与训练分离**：直接使用该项目进行语音，不需要GPU。
* **接口简洁**
* **模块化，可扩展**

## 5分钟上手

1. 克隆本项目到本地。

   ```sh
   git clone https://github.com/lukhy/masr.git
   ```

2. 从[这里](https://pan.baidu.com/s/1HmQqZXsyYz28fQ0XTfB8SA)（**提取码：xhks**）下载**预训练模型**和**测试音频文件**，并将它们拷贝到对应位置。

   ```sh
   mkdir masr/pretrained
   cp ~/Downloads/gated-conv.pth masr/pretrained/
   cp ~/Downloads/test.wav masr/
   cd masr
   ```

3. 打开测试文件，听一下，说的是：「你好，很高兴认识你」。

4. 安装依赖。

   ```sh
   pip install -r requirements.txt
   ```

5. 运行示例

   ```sh
   python examples/demo-recognize.py
   ```

   你将看到识别结果。

## 常见问题

### 需要联网才能识别吗？

不需要，不信你把网断了试试。

### 需要GPU吗？

**不需要GPU**。CPU上识别照样飞快。

### 识别效果差，怎么办？

等下一个版本。

### 其他问题

**除了以上问题之外**，如果你还有其他**任何**问题，请新建[issue](https://github.com/lukhy/masr/issues/new)让我知道。