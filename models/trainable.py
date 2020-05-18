from .base import MASRModel
import torch
import torch.nn as nn
import torch.optim as optim
import data
from tensorboardX import SummaryWriter
from warpctc_pytorch import CTCLoss
from tqdm import tqdm
from Levenshtein import distance


class TrainableModel(MASRModel):
    def __init__(self, **config):
        super().__init__(**config)

    def save(self, path):
        state_dict = self.state_dict()
        config = self.config
        package = {"state_dict": state_dict, "config": config}
        torch.save(package, path)

    def loss(self, *pred_targets):  # -> loss: scalar tensor
        preds, targets = pred_targets
        return self._default_loss(*preds, *targets)

    def cer(self, texts, *targets):  # -> cer: float
        return self._default_cer(texts, *targets)

    def _default_loss(self, yp, yp_lens, y, y_lens):  # -> ctc_loss: scalar tensor
        criterion = CTCLoss(size_average=True)
        yp = yp.permute(2, 0, 1)  # B * V * T -> T * B * V
        loss = criterion(yp, y, yp_lens, y_lens)
        return loss

    def _default_cer(self, texts, y, y_lens):  # -> cer: float
        index = 0
        cer = 0
        for text, y_len in zip(texts, y_lens):
            target = y[index : (index + y_len)]
            target = "".join(self.vocabulary[i] for i in target)
            print(text, target)
            cer += distance(text, target) / len(target)
            index += y_len
        cer /= len(y_lens)
        return cer

    def test(self, test_index, batch_size=64):  # -> cer: float
        self.eval()
        test_dataset = data.MASRDataset(test_index, self.vocabulary)
        test_loader = data.MASRDataLoader(
            test_dataset, batch_size, shuffle=False, num_workers=16
        )
        test_steps = len(test_loader)
        cer = 0
        for inputs, targets in tqdm(test_loader, total=test_steps):
            x, x_lens = inputs
            x = x.to("cuda")
            outputs = self.forward(x, x_lens)
            texts = self.decode(*outputs)
            cer += self.cer(texts, *targets)
        cer /= test_steps
        self.train()
        return cer

    def fit(
        self,
        train_index,
        dev_index,
        epochs=100,
        train_batch_size=64,
        lr=0.6,
        momentum=0.8,
        grad_clip=0.2,
        dev_batch_size=64,
        sorta_grad=True,
        tensorboard=True,
        quiet=False,
    ):
        self.to("cuda")
        self.train()
        if tensorboard:
            writer = SummaryWriter()
        optimizer = optim.SGD(self.parameters(), lr, momentum, nesterov=True)
        train_dataset = data.MASRDataset(train_index, self.vocabulary)
        train_loader_shuffle = data.MASRDataLoader(
            train_dataset, train_batch_size, shuffle=True, num_workers=16
        )
        if sorta_grad:
            train_loader_sort = data.MASRDataLoader(
                train_dataset, train_batch_size, shuffle=False, num_workers=16
            )
        train_steps = len(train_loader_shuffle)
        gstep = 0
        for epoch in range(epochs):
            avg_loss = 0
            if epoch == 0 and sorta_grad:
                train_loader = train_loader_sort
            else:
                train_loader = train_loader_shuffle
            for step, (inputs, targets) in enumerate(train_loader):
                x, x_lens = inputs
                x = x.to("cuda")
                gstep += 1
                outputs = self.forward(x, x_lens)
                loss = self.loss(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                optimizer.step()
                avg_loss += loss.item()
                if not quiet:
                    print(
                        "[{}/{}][{}/{}]\tLoss = {}".format(
                            epoch + 1, epochs, step + 1, train_steps, loss.item()
                        )
                    )
                if tensorboard:
                    writer.add_scalar("loss/step", loss.item(), gstep)
            cer = self.test(dev_index, dev_batch_size)
            avg_loss /= train_steps
            if not quiet:
                print("Epoch {}\t CER = {}\t".format(epoch + 1, cer))
            if tensorboard:
                writer.add_scalar("cer/epoch", cer, epoch + 1)
                writer.add_scalar("loss/epoch", loss, epoch + 1)
            self.save("pretrained/{}_epoch_{}.pth".format(self.name, epoch + 1))

