import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from .base import MASRModel
import feature


class ConvBlock(nn.Module):
    def __init__(self, conv, p):
        super().__init__()
        self.conv = conv
        nn.init.kaiming_normal_(self.conv.weight)
        self.conv = weight_norm(self.conv)
        self.act = nn.GLU(1)
        self.dropout = nn.Dropout(p, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class GatedConv(MASRModel):
    """ This is a model between Wav2letter and Gated Convnets.
        The core block of this model is Gated Convolutional Network"""

    def __init__(self, vocabulary, blank=0, name="masr"):
        """ vocabulary : str : string of all labels such that vocaulary[0] == ctc_blank  """
        super().__init__(vocabulary=vocabulary, name=name, blank=blank)
        self.blank = blank
        self.vocabulary = vocabulary
        self.name = name
        output_units = len(vocabulary)
        modules = []
        modules.append(ConvBlock(nn.Conv1d(161, 500, 48, 2, 97), 0.2))

        for i in range(7):
            modules.append(ConvBlock(nn.Conv1d(250, 500, 7, 1), 0.3))

        modules.append(ConvBlock(nn.Conv1d(250, 2000, 32, 1), 0.5))

        modules.append(ConvBlock(nn.Conv1d(1000, 2000, 1, 1), 0.5))

        modules.append(weight_norm(nn.Conv1d(1000, output_units, 1, 1)))

        self.cnn = nn.Sequential(*modules)

    def forward(self, x, lens):  # -> B * V * T
        x = self.cnn(x)
        for module in self.modules():
            if type(module) == nn.modules.Conv1d:
                lens = (
                    lens - module.kernel_size[0] + 2 * module.padding[0]
                ) // module.stride[0] + 1
        return x, lens

    def predict(self, path):
        self.eval()
        wav = feature.load_audio(path)
        spec = feature.spectrogram(wav)
        spec.unsqueeze_(0)
        x_lens = spec.size(-1)
        out = self.cnn(spec)
        out_len = torch.tensor([out.size(-1)])
        text = self.decode(out, out_len)
        self.train()
        return text[0]
