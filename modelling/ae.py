import torch.nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import Sequential, Linear, ReLU, MSELoss
from torch.optim import Adam
from utils import save_to_pickle
import numpy as np

seed = 123
np.random.seed(seed)


class AE(torch.nn.Module):
    def __init__(self, X_train, epochs, batch_size, lr, name):
        super(AE, self).__init__()
        self._X_train = torch.tensor(X_train, dtype=torch.float32)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.name = name
        input_dim = X_train.shape[1]
        self.encoder = Sequential(
            Linear(in_features=input_dim, out_features=int(input_dim/2)),
            ReLU(),
            Linear(in_features=int(input_dim/2), out_features=int(input_dim/4)),
            ReLU(),
        )
        self.decoder = Sequential(
            Linear(in_features=int(input_dim/4), out_features=int(input_dim/2)),
            ReLU(),
            Linear(in_features=int(input_dim/2), out_features=input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        out = self.decoder(encoded)
        return out

    def train_autoencoder_content(self):
        """
        Method that builds and trains the autoencoder that processes the textual content
        """
        print("\nTraining autoencoder")
        loss_function = MSELoss()
        opt = Adam(self.parameters(), self.lr)
        ds = TensorDataset(self._X_train)
        dl = DataLoader(ds, batch_size=self.batch_size)
        best_loss = 9999
        for epoch in range(self.epochs):
            self.train()
            total_loss = 0
            for batch in dl:
                out = self(batch[0])
                loss = loss_function(out, batch[0])
                opt.zero_grad()
                loss.backward()
                total_loss += loss
                opt.step()
            total_loss = total_loss/len(dl)
            if total_loss < best_loss:
                best_loss = total_loss
                print("Found best model at epoch {}. Loss: {}".format(epoch, best_loss))
                save_to_pickle(self.name, self)

    def predict(self, x):
        self.eval()
        return self(x)


    """def train_autoencoder_node(self, embedding_size):
        "
        Method that trains the autoencoder used for generating the node embeddings. Differently from the content
        autoencoder, here we train the model and then discard the decoder
        Args:
            embedding_size: Desired embedding dimension
        "
        if exists(self._model_dir):
            return self.load_autoencoder()
        else:
            print("Training node autoencoder")
            input_len = self._input_len
            f = 3       # Factor that regulates the architecture. Eg if f=2, the dimension of the layers will be gradually halved until the bottleneck reaches the desired dimension
            encoder_layers_dimensions = [self._input_len]   # Store the dimensions of the encoder layers, so we already know what will be the dimensions of the decoder layers
            input_features = Input(shape=(input_len,))
            encoded = input_features
            input_len = int(input_len/f)
            while input_len > embedding_size:
                encoder_layers_dimensions.append(input_len)
                encoded = Dense(units=input_len, activation='sigmoid')(encoded)
                input_len = int(input_len/f)
            encoded = Dense(units=embedding_size, activation='sigmoid')(encoded)
            decoded = Dense(units=input_len, activation='sigmoid')(encoded)
            input_len *= f
            encoder_layers_dimensions = encoder_layers_dimensions[::-1]     # Reverse the list
            for d in encoder_layers_dimensions:
                decoded = Dense(units=d, activation='sigmoid')(decoded)
            autoencoder = Model(input_features, decoded)
            opt = Adam(learning_rate=0.05)
            early_stopping = EarlyStopping(monitor='val_loss', patience=20)
            lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto')
            autoencoder.compile(optimizer=opt, loss='mse', metrics=['mse'])
            autoencoder.fit(self._X_train, self._X_train, epochs=self.epochs, batch_size=self.batch_size,
                            validation_split=0.2, callbacks=[early_stopping, lr_reducer])
            encoder = Model(input_features, encoded)
            encoder.save(self._model_dir)
            return encoder"""
