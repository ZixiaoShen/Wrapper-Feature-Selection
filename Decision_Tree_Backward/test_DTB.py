import scipy.io
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from Decision_Tree_Backward import decision_tree_backward

# mat = scipy.io.loadmat('/home/zealshen/DATA/DATAfromASU/FaceImageData/COIL20.mat')
# X = mat['X']
# X = X.astype(float)
# y = mat['Y']
# y = y[:, 0]
# n_samples, n_features = X.shape
#
# idx = decision_tree_backward.decision_tree_backward(X, y, 100)
# print(idx)


url = ''