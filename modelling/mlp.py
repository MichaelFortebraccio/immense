import torch.nn
from torch.nn import Linear, BCELoss, Sigmoid, Softmax
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from keras.utils import to_categorical
import numpy as np
from os.path import join
from sklearn.metrics import classification_report
from utils import plot_confusion_matrix


class MLP(torch.nn.Module):
    def __init__(self, X_train, y_train, model_dir, batch_size=128, epochs=50):
        super(MLP, self).__init__()
        self.X_train = X_train
        self.nz = np.count_nonzero(y_train)
        y_train = to_categorical(y_train)
        self.y_train = torch.tensor(y_train, dtype=torch.float) #.reshape(-1, 1)
        self.batch_size = batch_size
        self.epochs = epochs
        self._model_path = join(model_dir, "mlp.h5")
        self.input = Linear(in_features=7, out_features=3) #self.X_train.shape[1]
        self.output = Linear(in_features=3, out_features=2)
        #self.sigmoid = Sigmoid()
        self.softmax = Softmax(dim=1)

    def forward(self, x):
        x = self.input(x)
        x = F.relu(x)
        out = self.output(x)
        out = self.softmax(out)
        #out = self.sigmoid(x)
        return out

    def train_mlp(self, optimizer):
        print("Training MLP...")
        pos_weight = self.X_train.shape[0]/(self.nz)
        neg_weight = self.X_train.shape[0]/(2*(self.X_train.shape[0]-self.nz))
        criterion = BCELoss()
        ds = TensorDataset(self.X_train, self.y_train)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.epochs):
            self.train()
            for batch_x, batch_y in dl:
                out = self(batch_x)
                loss = criterion(out, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(loss)
            for batch_x, batch_y in dl:
                outputs = self(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                print("\n")
                print(outputs)


    def test(self, X_test, y_test):
        preds = self(X_test)
        y_p = []
        for p in preds:
            if round(p[0]) == 0:
                y_p.append(1)
            elif round(p[0]) == 1:
                y_p.append(0)
        plot_confusion_matrix(y_true=y_test, y_pred=y_p)
        return classification_report(y_true=y_test, y_pred=y_p)