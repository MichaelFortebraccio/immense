import pickle
from sklearn.ensemble import RandomForestClassifier
from utils import save_to_pickle


def train_random_forest(train_set, dst_dir, train_set_labels, name):
    print("Training {} random forest".format(name))
    cls = RandomForestClassifier(criterion="gini", max_depth=5, min_samples_split=10, n_estimators=10)
    cls.fit(train_set, train_set_labels)
    save_to_pickle(dst_dir, cls)


def test_random_forest(test_set, cls: RandomForestClassifier):
    predictions = cls.predict(test_set)
    leaf_id = cls.apply(test_set)
    purities = []
    for i in range(predictions.shape[0]):
        purity = 0
        for j in range(len(cls.estimators_)):
            purity += 1 - cls.estimators_[j].tree_.impurity[leaf_id[i][j]]
        purities.append(purity/len(cls.estimators_))
    return predictions, purities

