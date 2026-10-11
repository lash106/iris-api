import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


X, y = load_iris(return_X_y=True)
model = RandomForestClassifier(random_state=42).fit(X, y)
joblib.dump(model, "model.joblib")
print("model saved")