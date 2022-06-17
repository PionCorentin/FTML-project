import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def test_model(model):
    model.fit(X_train, y_train.ravel())
    y_pred = model.predict(X_test)
    return r2_score(y_test, y_pred)

X = np.load("inputs.npy")
y = np.load("labels.npy")
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)

model_list = {
    "LinearRegression": LinearRegression(),
    "Lasso": Lasso(alpha=1.0),
    "Ridge": Ridge(alpha=6.0)
}

for (name, model) in model_list.items():
    print(name, "accuracy:", test_model(model))
