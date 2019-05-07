import Levenshtein as Lev
import torch


class CTCGreedyDecoder(object):
    def __init__(self, vocabulary, blank=0):
        self.vocabulary = vocabulary
        self.blank = blank

    def decode(self, outs, out_lens=None):
        idxs = outs.argmax(1)
        texts = []
        for idx, out_len in zip(idxs, out_lens):
            idx = idx[:out_len]
            text = ""
            last = None
            for i in idx:
                if i.item() not in (last, self.blank):
                    text += self.vocabulary[i]
                last = i
            texts.append(text)
        return texts

