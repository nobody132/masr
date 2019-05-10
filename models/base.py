import torch
import torch.nn as nn


class MASRModel(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.config = config

    @classmethod
    def load(cls, path):
        package = torch.load(path)
        state_dict = package["state_dict"]
        config = package["config"]
        m = cls(**config)
        m.load_state_dict(state_dict)
        return m

    def to_train(self):
        from .trainable import TrainableModel

        self.__class__.__bases__ = (TrainableModel,)
        return self

    def predict(self, *args):
        raise NotImplementedError()

    # -> texts: list, len(list) = B
    def _default_decode(self, yp, yp_lens):
        idxs = yp.argmax(1)
        texts = []
        for idx, out_len in zip(idxs, yp_lens):
            idx = idx[:out_len]
            text = ""
            last = None
            for i in idx:
                if i.item() not in (last, self.blank):
                    text += self.vocabulary[i.item()]
                last = i
            texts.append(text)
        return texts

    def decode(self, *outputs):  # texts -> list of size B
        return self._default_decode(*outputs)
