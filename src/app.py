"""Inferencia de sentimiento sobre las reseñas de WeLoveReviews.

Lee data/raw/reviews.csv, predice una puntuación de 1 a 5 estrellas por reseña
con el modelo fijado más abajo y escribe data/processed/reviews_with_sentiment.csv.

Uso, desde la carpeta del proyecto:

    python src/app.py
"""

from pathlib import Path

import pandas as pd
from transformers import pipeline

# Modelo y revisión fijados: el hash es el commit del repositorio del modelo en
# Hugging Face, así que otra persona que clone el proyecto carga los mismos pesos.
MODEL_ID = "nlptown/bert-base-multilingual-uncased-sentiment"
MODEL_REVISION = "8f6f4e3a8f70be4b65d3a4a8762b6d781cda240d"

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_DIR / "data" / "raw" / "reviews.csv"
OUTPUT_PATH = PROJECT_DIR / "data" / "processed" / "reviews_with_sentiment.csv"

REQUIRED_COLUMNS = ["review_id", "rating", "review_text"]
BATCH_SIZE = 32

# Mapeo del enunciado: 1-2 estrellas negativo, 3 neutral, 4-5 positivo.
STARS_TO_SENTIMENT = {
    1: "negative",
    2: "negative",
    3: "neutral",
    4: "positive",
    5: "positive",
}


def load_reviews(path=RAW_PATH):
    """Carga el CSV de reseñas y comprueba lo mínimo que necesita la inferencia."""
    reviews = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in reviews.columns]
    if missing:
        raise ValueError(f"Faltan columnas en {path}: {missing}")

    reviews["review_text"] = reviews["review_text"].astype("string").str.strip()
    empty = reviews["review_text"].isna() | (reviews["review_text"] == "")
    if empty.any():
        ids = reviews.loc[empty, "review_id"].tolist()
        raise ValueError(f"Reseñas sin texto, no se pueden clasificar: {ids}")
    return reviews


def load_classifier():
    """Carga el modelo una sola vez; el resultado se reutiliza para todas las reseñas."""
    return pipeline(
        "text-classification",
        model=MODEL_ID,
        revision=MODEL_REVISION,
        device=-1,  # CPU: para 500 reseñas cortas no hace falta GPU
    )


def predict_sentiment(classifier, texts):
    """Devuelve estrellas predichas, banda de sentimiento y confianza por texto."""
    outputs = classifier(list(texts), batch_size=BATCH_SIZE, truncation=True)
    # Las etiquetas del modelo son "1 star", "2 stars", ..., "5 stars".
    stars = [int(output["label"].split()[0]) for output in outputs]
    return pd.DataFrame(
        {
            "predicted_stars": stars,
            "sentiment": [STARS_TO_SENTIMENT[star] for star in stars],
            "confidence": [round(output["score"], 4) for output in outputs],
        }
    )


def main():
    reviews = load_reviews()
    classifier = load_classifier()

    predictions = predict_sentiment(classifier, reviews["review_text"])
    predictions.index = reviews.index
    enriched = pd.concat([reviews, predictions], axis=1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(OUTPUT_PATH, index=False)

    shares = enriched["sentiment"].value_counts(normalize=True).mul(100).round(1)
    print(f"Modelo: {MODEL_ID}@{MODEL_REVISION[:8]}")
    print(f"Reseñas procesadas: {len(enriched)}")
    for band in ("positive", "neutral", "negative"):
        print(f"  {band:<8} {shares.get(band, 0.0):5.1f} %")
    print(f"Salida: {OUTPUT_PATH.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
