from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string


class TextPreprocessing:
    def __init__(self):
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.porter_stemmer = PorterStemmer()

    def preprocessing_text(self, text):
        list_sentences = []
        new_string = text.translate(str.maketrans('', '', string.punctuation))      # Remove punctuation
        text_tokens = word_tokenize(new_string)
        for n in text_tokens:
            new_string = self.porter_stemmer.stem(n)
            new_string = self.wordnet_lemmatizer.lemmatize(new_string)
            list_sentences.append(new_string)
        return list_sentences

    def token_list(self, df) -> dict:
        d = {}
        #list_sentences = []
        for i, r in df.iterrows():
            try:
                splitted = r['text_cleaned'].split(' ')
                splitted = self.remove_nonalpha(splitted)
                d[r['id']] = splitted
            except AttributeError:
                print("met nan. skipping")
        return d

    def remove_nonalpha(self, tl):
        to_pop = []
        for i in range(len(tl)):
            if not tl[i].isalpha():
                to_pop.append(i)
        to_pop.reverse()
        for idx in to_pop:
            tl.pop(idx)
        return tl

