import _init_path
from models.conv import GatedConv
from record import record

model = GatedConv.load("pretrained/gated-conv.pth")

record("record.wav")

text = model.predict("record.wav")

print("")
print("识别结果:")
print(text)
