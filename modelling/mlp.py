import numpy as np
import torch
from os.path import join
from torch.nn import BCELoss, Linear, Softmax
from torch.nn.functional import relu
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from utils import save_to_pickle


class MLP(torch.nn.Module):
    def __init__(self, X_train, y_train, model_path, batch_size=64, epochs=50, weights=None):
        super(MLP, self).__init__()
        self.X_train = X_train
        self.weights = weights
        y_train = np.eye(2, dtype='uint8')[y_train]
        self.y_train = torch.tensor(y_train, dtype=torch.float)
        self._model_path = join(model_path)
        self.batch_size = batch_size
        self.epochs = epochs
        self.input = Linear(in_features=7, out_features=3)
        self.output = Linear(in_features=3, out_features=2)
        self.softmax = Softmax(dim=1)

    def forward(self, x):
        x = self.input(x)
        x = relu(x)
        out = self.output(x)
        out = self.softmax(out)
        return out

    def train_mlp(self, optimizer):
        print("Training MLP...")
        criterion = BCELoss(weight=self.weights)
        ds = TensorDataset(self.X_train, self.y_train)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        best_loss = 9999
        for epoch in range(50):     #self.epochs
            self.train()
            total_loss = 0

            for batch_x, batch_y in tqdm(dl):
                out = self(batch_x)
                loss = criterion(out, batch_y)
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

    def test(self, X_test, y_test):
        self.eval()
        preds = self(X_test)
        y_p = []
        for p in preds:
            if p[0] < p[1]:
                y_p.append(1)
            else:
                y_p.append(0)
        return y_p

