import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def test_model(model):
    model.fit(X_train, y_train.ravel())
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

X = np.load("inputs_classification.npy")
y = np.load("labels_classification.npy")
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)

model_list = {
            "LinearSVC": LinearSVC(max_iter=10000),
            "KNeighbors": KNeighborsClassifier(),
            "SVC" : SVC(max_iter=10000),
            "RandomForestClassifier ": RandomForestClassifier(),
            "AdaBoostClassifier ": AdaBoostClassifier(),
            "GradientBoostingClassifier ": GradientBoostingClassifier()
            }

for (name, model) in model_list.items():
    print(name, "accuracy:", test_model(model))