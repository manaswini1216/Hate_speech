from fastapi import FastAPI
from pydantic import BaseModel
from backend.predictor import predict_text
import os
import json
from datetime import datetime

app = FastAPI()

# ---------------- INPUT VALIDATION ---------------- #

class TextRequest(BaseModel):
    text: str

# ---------------- LOGGING ---------------- #

def save_log(input_text, result):

    log_data = {
        "timestamp": str(datetime.now()),
        "input_text": input_text,
        "result": result
    }

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    log_path = os.path.join(BASE_DIR, "logs", "predictions.jsonl")
    with open(log_path, "a") as file:

        file.write(json.dumps(log_data) + "\n")
# ---------------- HOME ROUTE ---------------- #

@app.get("/")
def home():

    return {
        "message": "Hate Speech Detection API Running"
    }

# ---------------- PREDICTION ROUTE ---------------- #

@app.post("/predict")
def predict(data: TextRequest):

    text = data.text.strip()

    # empty input validation
    if len(text) == 0:

        return {
            "error": "Empty input"
        }

    # long text validation
    if len(text) > 500:

        return {
            "error": "Text too long"
        }

    # model prediction
    result = predict_text(text)

    # save logs
    save_log(text, result)

    return result
