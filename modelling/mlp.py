import torch
import numpy as np
from kornia.losses import binary_focal_loss_with_logits
from os.path import join
from torch.nn import Linear
from torch.nn.functional import relu, nll_loss
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from utils import save_to_pickle


class MLP(torch.nn.Module):
    def __init__(self, X_train, y_train, model_path, loss, batch_size=64, epochs=50, weights=None):
        super(MLP, self).__init__()
        self.X_train = X_train
        self.weights = weights
        self.y_train = torch.tensor(y_train, dtype=torch.float)
        self._model_path = join(model_path)
        self.batch_size = batch_size
        self.epochs = epochs
        self.input = Linear(in_features=7, out_features=3)
        self.output = Linear(in_features=3, out_features=2)
        self.loss = loss

    def forward(self, x):
        x = self.input(x)
        x = relu(x)
        out = self.output(x)
        if self.loss == "weighted":
            out = torch.log_softmax(out, dim=-1)
        return out

    def train_mlp(self, optimizer):
        print("Training MLP...")
        ds = TensorDataset(self.X_train, self.y_train)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        best_loss = 9999
        for epoch in range(self.epochs):
            self.train()
            total_loss = 0

            for batch_x, batch_y in tqdm(dl):
                out = self(batch_x)
                if self.loss == "focal":
                    t = [int(el) for el in batch_y]
                    target = torch.tensor(np.eye(2, dtype='uint8')[t], dtype=torch.float)
                    loss = binary_focal_loss_with_logits(out, target=target, reduction="mean")
                elif self.loss == "weighted":
                    loss = nll_loss(out, batch_y.long(), weight=self.weights)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            l = total_loss/self.batch_size
            if epoch % 5 == 0:
                print("\nEpoch: {}, Loss: {}".format(epoch, l))
            if l < best_loss:
                best_loss = l
                print("New best model found at epoch {}. Loss: {}".format(epoch, best_loss))
                save_to_pickle(self._model_path, self)

    def test(self, X_test):
        self.eval()
        preds = self(X_test)
        if self.loss == "focal":
            preds = torch.log_softmax(preds, dim=-1)
        y_p = []
        for p in preds:
            if p[0] < p[1]:
                y_p.append(1)
            else:
                y_p.append(0)
        return y_p

