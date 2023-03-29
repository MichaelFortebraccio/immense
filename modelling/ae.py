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
    def __init__(self, input_len, X_train, label, model_dir):
        self._input_len = input_len
        self._X_train = X_train
        self._model_dir = join(model_dir, "autoencoder{}.h5".format(label))

    def train_autoencoder(self):
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
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
                                       min_delta=0.0001, cooldown=0, min_lr=0)
        autoencoder.compile(optimizer=opt, loss='mse', metrics=['mse'])
        x_train_sigmoid = sigmoid(tf.constant(self._X_train, dtype=tf.float32)).numpy()
        autoencoder.fit(self._X_train, x_train_sigmoid, epochs=100, batch_size=128, validation_split=0.2,
                        callbacks=[early_stopping, lr_reducer])
        autoencoder.save(self._model_dir)
        return autoencoder

    def load_autoencoder(self):
        """
        This method loads and returns the autoencoder model. If it hasn't been trained, it trains it
        :return:
        """
        if not exists(self._model_dir):
            print("The autoencoder does not exist. Now I am training it...")
            self.train_autoencoder()
        return load_model(self._model_dir)
