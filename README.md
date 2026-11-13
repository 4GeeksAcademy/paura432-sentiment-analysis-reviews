# Análisis de sentimiento en reseñas — WeLoveReviews

Proyecto del módulo «Usar modelos de predicción» de 4Geeks (*Análisis de Sentimiento en Reseñas de Clientes — WeLoveReviews*), hecho sobre un fork de [`machine-learning-python-template`](https://github.com/4GeeksAcademy/machine-learning-python-template).

La pregunta de negocio es si lo que dicen 500 reseñas escritas coincide con la puntuación media del negocio, 4,5 / 5. Se responde con un modelo de sentimiento ya entrenado de Hugging Face, sin entrenar nada. El análisis, los resultados y las conclusiones están en el notebook; este README solo explica qué hay en el repositorio y cómo ejecutarlo.

## Contenido

| Ruta | Qué es |
| --- | --- |
| `src/explore.ipynb` | Notebook ejecutado: objetivos, exploración, plan y modelo, inferencia, desglose frente al 4,5, falsos negativos, muestra revisada a mano, paso a producción, conclusiones y un anexo que compara con un segundo modelo |
| `src/app.py` | Ruta de inferencia de producción: lee `data/raw/reviews.csv` y escribe `data/processed/reviews_with_sentiment.csv` |
| `data/raw/reviews.csv` | Las 500 reseñas del enunciado (`review_id`, `rating`, `review_text`) |
| `data/processed/reviews_with_sentiment.csv` | Las mismas 500 reseñas con `predicted_stars` (1–5), `sentiment` (`negative`, `neutral`, `positive`) y `confidence` |
| `requirements.txt` | Dependencias del template más `torch` y `transformers`, todas con versión fija |

El resto (`.devcontainer/`, `src/utils.py`, `.env.example`, `models/`, `data/interim/`) viene del template y este proyecto no lo usa.

## Datos

`data/raw/reviews.csv` es una copia sin cambios del archivo del proyecto `existing-model-sentiment-analysis-reviews` del [syllabus de 4Geeks](https://github.com/4GeeksAcademy/ai-engineering-syllabus/tree/main/content/projects/existing-model-sentiment-analysis-reviews): 500 filas, UTF-8, finales de línea CRLF. Son reseñas de ejemplo en inglés sobre un único local y no contienen datos personales.

`data/processed/reviews_with_sentiment.csv` lo genera `src/app.py`.

## Modelo

| | Modelo | Revisión fijada |
| --- | --- | --- |
| Análisis y producción | `nlptown/bert-base-multilingual-uncased-sentiment` | `8f6f4e3a8f70be4b65d3a4a8762b6d781cda240d` |
| Anexo del notebook | `tabularisai/multilingual-sentiment-analysis` | `eea032081f8d247b4303ef3565e7cec1b6f201c9` |

Los dos se cargan con `pipeline()` de `transformers`, una sola vez por ejecución, y se fijan por nombre y por revisión (el commit del modelo en Hugging Face). Los pesos se descargan a la caché local de Hugging Face en la primera ejecución (unos 640 MB y 520 MB); no están en el repositorio.

El modelo principal predice de 1 a 5 estrellas. Se agrupan en bandas: 1–2 `negative`, 3 `neutral`, 4–5 `positive`.

## Ejecución

En Codespaces el entorno instala `requirements.txt` al crearse. En local hace falta Python 3.11 o superior (probado con 3.12 y 3.13):

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

En Linux, `pip` descarga por defecto la variante de `torch` con CUDA, que ocupa varios GB. En una máquina sin GPU basta la de CPU, que se instala antes que el resto:

```sh
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

La inferencia de producción clasifica las 500 reseñas en CPU y termina imprimiendo el reparto por bandas y la ruta del CSV:

```sh
python src/app.py
```

Para volver a ejecutar el notebook entero, con sus salidas:

```sh
jupyter nbconvert --to notebook --execute --inplace src/explore.ipynb
```

El notebook ejecuta también `src/app.py` y comprueba que el CSV de producción coincide con sus propias predicciones.
