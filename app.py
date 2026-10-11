import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Iris Classifier API")
model = joblib.load("model.joblib")
NAMES = ["setosa", "versicolor", "virginica"]

class Flower(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(f: Flower):
    x = [[f.sepal_length, f.sepal_width, f.petal_length, f.petal_width]]
    return {"species": NAMES[int(model.predict(x)[0])]}