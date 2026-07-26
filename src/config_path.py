from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

# 1. Путь к базе данных
DB_PATH = BASE_DIR / "data" / "database.db"

# 2. Пути к моделям
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"