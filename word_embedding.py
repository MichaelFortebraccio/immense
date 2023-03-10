from gensim.models import Word2Vec
import numpy as np
from os.path import exists
seed = 123
np.random.seed(seed)


class WordEmb:
    def __init__(self, token_word):
        self._token_word = token_word
        self._word_vec_dict = {}

    def load_dict(self):
        if exists('model/w2v_text.h5'):
            w2v_model = Word2Vec.load('model/w2v_text.h5')
            vocab = w2v_model.wv.index_to_key
            for word in vocab:
                self._word_vec_dict[word] = w2v_model.wv.get_vector(word)
        else:
            print('Please train W2V model.')

    def train_w2v(self):
        print('start w2v training')
        w2v_model = Word2Vec(vector_size=512, seed=seed, window=5, min_count=0, sg=1, workers=1)
        w2v_model.build_vocab(self._token_word, min_count=1)
        total_examples = w2v_model.corpus_count
        w2v_model.train(self._token_word, total_examples=total_examples, epochs=10)
        w2v_model.save('model/w2v_text.h5')
        print('end w2v training')

    def text_to_vec(self, tweets):
        self.load_dict()
        list_tot = []
        for tw in tweets:
            list_temp = []
            for t in tw:
                embed_vector = self._word_vec_dict.get(t)
                if embed_vector is not None:  # word is in the vocabulary learned by the w2v model
                    list_temp.append(embed_vector)
                else:
                    list_temp.append(np.zeros(shape=(512)))
            list_temp = np.array(list_temp)
            list_temp = np.sum(list_temp, axis=0)
            list_tot.append(list_temp)
        list_tot = np.asarray(list_tot)
        return list_tot