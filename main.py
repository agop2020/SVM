import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

#print(cancer.feature_names)
#print(cancer.target_names)

x = cancer.data
y = cancer.target

#not recommended to go above 0.3 for training data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.2)

#print(x_train, y_train)
classes = ['malignant', 'benign']

#clf = KNeighborsClassifier(n_neighbors=9) -> KNN test for comparison
#no parameters in SVC, low accuracy
#kernel linear means a linear function is used to add the third dimension
#C=0 means hard margin, C=2 means double the points allowed in that margin.

clf = svm.SVC(kernel="linear", C=2)
clf.fit(x_train, y_train)

y_prediction = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_prediction)

print(acc)


