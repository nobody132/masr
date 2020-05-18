# 识别自己的语音

识别自己的语音需要额外安装一个依赖：pyaudio

参考[pyaudio官网](https://people.csail.mit.edu/hubert/pyaudio/)把它装上，然后执行以下命令即可。

```sh
python examples/demo-record-recognize.py
```

请在看到提示「录音中」后开始说话，你有5秒钟的说话时间（可以自己在源码中更改）。

