"""
Configuration and constants for Premier League Transfer Value Predictor.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = BASE_DIR / "models"
VISUALIZATIONS_DIR = BASE_DIR / "visualizations"

DATASET_PATH = DATA_DIR / "premier_league_players.csv"
MODEL_PATH = MODELS_DIR / "transfer_value_model.joblib"

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, MODELS_DIR, VISUALIZATIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Football-Data.org API Configuration
# Get your free API key at https://www.football-data.org/client/register or set in .env
FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "")
FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"
PREMIER_LEAGUE_CODE = "PL"

# Model Features
NUMERICAL_FEATURES = ["goals", "assists", "minutes_played", "age"]
CATEGORICAL_FEATURES = ["position"]
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET = "actual_value_eur_m"

# Valid positions in the Premier League
VALID_POSITIONS = ["Forward", "Midfielder", "Defender", "Goalkeeper"]
