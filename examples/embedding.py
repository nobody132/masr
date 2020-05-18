import _init_path
from models.conv import GatedConv
import numpy as np
import torch
import heapq
from torch.nn.utils import remove_weight_norm

torch.set_grad_enabled(False)

model = GatedConv.load("pretrained/gated-conv.pth")
model.eval()

conv = model.cnn[10]
remove_weight_norm(conv)


w = conv.weight.squeeze().detach()
b = conv.bias.unsqueeze(1).detach()

embed = w

vocab = model.vocabulary
v = dict((vocab[i], i) for i in range(len(vocab)))


def cos(c1, c2):
    e1, e2 = embed[v[c1]], embed[v[c2]]
    return (e1 * e2).sum() / (e1.norm() * e2.norm())


def nearest(c, n=5):
    def gen():
        for c_ in v:
            if c_ == c:
                continue
            yield cos(c, c_), c_

    return heapq.nlargest(n, gen())


def main():
    while True:
        c = input("请输入一个汉字：")
        if c not in v:
            print(f"词汇表里没有「{c}」")
            continue
        print("以下是cos相似度最高的")
        for p, c in nearest(c):
            print(c, end=", ")
        print("")


if __name__ == "__main__":
    main()
