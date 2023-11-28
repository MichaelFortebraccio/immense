import pickle
from sklearn.ensemble import RandomForestClassifier


def train_decision_tree(train_set, save_path, train_set_labels, name):
    print("Training {} decision tree".format(name))
    cls = RandomForestClassifier(criterion="gini", max_depth=5)
    cls.fit(train_set, train_set_labels)
    pickle.dump(cls, open(save_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def test_decision_tree(test_set, cls: RandomForestClassifier):
    predictions = cls.predict(test_set)
    leaf_id = cls.apply(test_set)
    purities = []
    for i in range(predictions.shape[0]):
        purity = 0
        for j in range(len(cls.estimators_)):
            purity += 1 - cls.estimators_[j].tree_.impurity[leaf_id[i][j]]
        purities.append(purity/len(cls.estimators_))
    #purity = 1 - cls.tree_.impurity[leaf_id]
    return predictions, purities


def load_decision_tree(path):
    print("Loading decision tree")
    return pickle.load(open(path, 'rb'))
