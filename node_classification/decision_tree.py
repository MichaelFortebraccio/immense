from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from gensim.models import Word2Vec
import pickle


def train_decision_tree(train_set, save_path, train_set_labels, name):
    print("Training {} decision tree".format(name))
    cls = DecisionTreeClassifier(criterion="gini")
    cls.fit(train_set, train_set_labels)
    pickle.dump(cls, open(save_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def test_decision_tree(test_set_ids, cls: DecisionTreeClassifier, n2v_model: Word2Vec):
    mod = n2v_model.wv
    test_set = [mod.vectors[mod.key_to_index[i]] for i in test_set_ids]
    predictions = cls.predict(test_set)
    leaf_id = cls.apply(test_set)
    purity = 1 - cls.tree_.impurity[leaf_id]
    #print(classification_report(test_set_labels, predictions, labels=[0,1]))
    return predictions, purity


def load_decision_tree(path):
    print("Loading decision tree")
    return pickle.load(open(path, 'rb'))
