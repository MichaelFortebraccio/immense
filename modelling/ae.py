from keras.layers import Dense, Input
from keras import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import load_model
from keras.activations import sigmoid
import numpy as np
import tensorflow as tf
from os.path import exists, join

seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)


class AE:
    def __init__(self, X_train, name, model_dir, epochs, batch_size, lr):
        self._X_train = X_train
        self._input_len = self._X_train.shape[1]
        self._model_dir = join(model_dir, "{}_{}.h5".format(name, X_train.shape[1]))
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def train_autoencoder_content(self):
        """
        Method that builds and trains the autoencoder that processes the textual content
        """
        if exists(self._model_dir):
            return self.load_autoencoder()
        else:
            print("Training autoencoder")
            input_features = Input(shape=(self._input_len,))
            # encoder
            encoded = Dense(units=int(self._input_len/2), activation='sigmoid')(input_features)
            encoded = Dense(units=int(self._input_len/4), activation='sigmoid')(encoded)
            # latent-space
            encoded = Dense(units=int(self._input_len/8), activation='sigmoid')(encoded)
            # decoded
            decoded = Dense(units=int(self._input_len/4), activation='sigmoid')(encoded)
            decoded = Dense(units=int(self._input_len/2), activation='sigmoid')(decoded)
            decoded = Dense(units=self._input_len, activation='sigmoid')(decoded)
            autoencoder = Model(input_features, decoded)
            opt = Adam(learning_rate=0.05)
            early_stopping = EarlyStopping(monitor='val_loss', patience=20)
            lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto')

            autoencoder.compile(optimizer=opt, loss='mse', metrics=['mse'])
            x_train_sigmoid = sigmoid(tf.constant(self._X_train, dtype=tf.float32)).numpy()
            autoencoder.fit(self._X_train, x_train_sigmoid, epochs=self.epochs, batch_size=self.batch_size,
                            validation_split=0.2, callbacks=[early_stopping, lr_reducer])
            autoencoder.save(self._model_dir)
            return autoencoder

    def train_autoencoder_node(self, embedding_size):
        """
        Method that trains the autoencoder used for generating the node embeddings. Differently from the content
        autoencoder, here we train the model and then discard the decoder
        Args:
            embedding_size: Desired embedding dimension
        """
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
            return encoder

    def load_autoencoder(self):
        return load_model(self._model_dir)
