import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def decision_tree_backward(X, y, n_selected_features):
    """
    This function implements the backward feature selection algorithm based on decision tree
    """

    n_samples, n_features = X.shape
    # using 10 fold cross validation
    kf = KFold(n_splits=10)
    # choose decision tree as the classifier
    clf = DecisionTreeClassifier()

    # selected feature set, initialized to be empty
    F = range(n_features)
    idx = 0
    count = n_features
    while count > n_selected_features:
        max_acc = 0
        for i in range(n_features):
            if i in F:
                np.delete(F, i)
                #F.remove(i)
                X_tmp = X[:, F]
                acc = 0
                for train_Id, test_Id in kf.split(X_tmp):
                    clf.fit(X_tmp[train_Id], y[train_Id])
                    y_predict = clf.predict(X_tmp[test_Id])
                    acc_tmp = accuracy_score(y[test_Id], y_predict)
                    acc += acc_tmp
                acc = float(acc)/10
                np.append(F, i)
                # F.append(i)
                # record the feature which results in the largest accuracy
                if acc > max_acc:
                    max_acc = acc
                    idx = i
        # delete the feature which results in the largest accuracy
        np.delete(F, idx)
        # F.remove(idx)
        count -= 1
    return np.array(F)
