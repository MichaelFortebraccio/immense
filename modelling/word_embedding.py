import numpy as np
from gensim.models import Word2Vec
from os.path import exists
from utils import load_from_pickle, save_to_pickle

seed = 123
np.random.seed(seed)


class WordEmb:
    def __init__(self, token_word, embedding_size, window, epochs, model_dir):
        self._token_word = token_word
        self._word_vec_dict = {}
        self._model_dir = model_dir
        self.embedding_size = embedding_size
        self.window = window
        self.epochs = epochs
        self.model = None

    def load_dict(self):
        if self.model is None:
            raise Exception("Please train W2V model")
        w2v_model = self.model
        vocab = w2v_model.wv.index_to_key
        for word in vocab:
            self._word_vec_dict[word] = w2v_model.wv.get_vector(word)

    def train_w2v(self):
        w2v_model = Word2Vec(vector_size=self.embedding_size, seed=seed, window=self.window, min_count=0, sg=1, workers=5)
        w2v_model.build_vocab(self._token_word, min_count=1)
        total_examples = w2v_model.corpus_count
        w2v_model.train(self._token_word, total_examples=total_examples, epochs=self.epochs)
        self.model = w2v_model

    def load_model(self):
        return Word2Vec.load(self._model_dir)

    def text_to_vec(self, users):
        self.load_dict()
        d = {}
        for u in users:
            list_temp = []
            for w in users[u]:
                embed_vector = self._word_vec_dict.get(w)
                if embed_vector is not None:  # word is in the vocabulary learned by the w2v model
                    list_temp.append(embed_vector)
                else:
                    list_temp.append(np.zeros(shape=(self.embedding_size)))
            list_temp = np.array(list_temp)
            list_temp = np.sum(list_temp, axis=0)
            d[u] = list_temp
        #list_tot = np.asarray(list_tot)
        return d
