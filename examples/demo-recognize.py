import _init_path
from models.conv import GatedConv

model = GatedConv.load("pretrained/gated-conv.pth")

text = model.predict("test.wav")

print("")
print("识别结果:")
print(text)
