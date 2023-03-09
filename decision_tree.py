from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from gensim.models import Word2Vec


def train_decision_tree(train_set_ids, train_set_labels, n2v_model:Word2Vec):
    mod = n2v_model.wv
    train_set = [mod.vectors[mod.key_to_index[i]] for i in train_set_ids]
    cls = DecisionTreeClassifier(criterion="gini")
    cls.fit(train_set, train_set_labels)
    return cls


def test_decision_tree(test_set_ids, test_set_labels, cls: DecisionTreeClassifier, n2v_model:Word2Vec):
    mod = n2v_model.wv
    test_set = [mod.vectors[mod.key_to_index[i]] for i in test_set_ids]
    predictions = cls.predict(test_set)
    print(classification_report(test_set_labels, predictions, labels=[0,1]))

