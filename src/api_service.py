from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import pickle
import os
import uvicorn

from src.data_preprocessing import preprocess_text, download_nltk_resources
from fastapi.middleware.cors import CORSMiddleware



model = None
tfidf_vectorizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tfidf_vectorizer

    # Startup
    download_nltk_resources()

    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(base_path, ".."))

        model_path = os.path.join(project_root, "models", "spam_classifier.pkl")
        tfidf_path = os.path.join(project_root, "models", "tfidf_vectorizer.pkl")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        with open(tfidf_path, "rb") as f:
            tfidf_vectorizer = pickle.load(f)

        print("Models loaded successfully")

    except Exception as e:
        raise RuntimeError(f"Startup failed: {e}")

    yield

    # Shutdown
    model = None
    tfidf_vectorizer = None


app = FastAPI(
    title="SMS Spam Classifier API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://localhost:4200"
    ],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


class Message(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)


class PredictionResponse(BaseModel):
    is_spam: bool
    prediction: str
    probability_spam: float
    probability_ham: float


@app.get("/")
def read_root():
    return {"message": "SMS Spam Classifier API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict_spam(message_data: Message):
    if model is None or tfidf_vectorizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    cleaned_text = preprocess_text(message_data.message)
    vector = tfidf_vectorizer.transform([cleaned_text])

    probabilities = model.predict_proba(vector)[0]
    spam_prob = probabilities[1]

    THRESHOLD = 0.25
    prediction = 1 if spam_prob >= THRESHOLD else 0

    return {
        "is_spam": bool(prediction),
        "prediction": "SPAM" if prediction == 1 else "HAM (Not Spam)",
        "probability_spam": float(spam_prob),
        "probability_ham": float(probabilities[0]),
    }

@app.get("/health")
def health_check():
    if model is None or tfidf_vectorizer is None:
        return {"status": "error"}
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("api_service:app", host="0.0.0.0", port=8000)
