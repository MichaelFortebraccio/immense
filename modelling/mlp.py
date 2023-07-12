from keras.layers import Input, Dense
from keras import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model
from os.path import join
from sklearn.metrics import classification_report
from utils import plot_confusion_matrix


class MLP:
    def __init__(self, X_train, y_train, model_dir, batch_size=128, epochs=50, lr=0.03):
        self.X_train = X_train
        if len(y_train.shape) == 1:
            y_train = to_categorical(y_train)
        self.y_train = y_train
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self._model_path = join(model_dir, "mlp.h5")

    def train(self):
        print("Training MLP...")
        input_layer = Input(shape=(self.X_train.shape[1],))
        x = Dense(units=3, activation='sigmoid')(input_layer)
        output = Dense(units=2, activation='softmax')(x)
        mod = Model(input_layer, output)
        opt = Adam(learning_rate=self.lr)
        mod.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        mod.fit(self.X_train, y=self.y_train, batch_size=self.batch_size, epochs=self.epochs, validation_split=0.2, verbose=0)
        self.model = mod

    def test(self, X_test, y_test):
        preds = self.model.predict(X_test)
        y_p = []
        for p in preds:
            if round(p[0]) == 0:
                y_p.append(1)
            elif round(p[0]) == 1:
                y_p.append(0)
        plot_confusion_matrix(y_true=y_test, y_pred=y_p)
        return classification_report(y_true=y_test, y_pred=y_p)

    def load_weights(self):
        self.model = load_model(self._model_path)
