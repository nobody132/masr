import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, **config):
        super().__init__()
        for k in config:
            self.__setattr__(k, config[k])
        self.config = config

    def save(self, path):
        state_dict = self.state_dict()
        config = self.config
        package = {"state_dict": state_dict, "config": config}
        torch.save(package, path)

    @classmethod
    def load(cls, path):
        package = torch.load(path)
        state_dict = package["state_dict"]
        config = package["config"]
        m = cls(**config)
        m.load_state_dict(state_dict)
        return m

    def predict(self, f):
        raise NotImplementedError()

    def loss(self, *inputs):
        raise NotImplementedError()
