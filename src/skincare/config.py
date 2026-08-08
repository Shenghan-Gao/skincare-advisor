from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW, PROCESSED, KNOWLEDGE = DATA / "raw", DATA / "processed", DATA / "knowledge"
MODELS = ROOT / "models"
FIXTURES = ROOT / "fixtures"

SKIN_TYPES = ["oily", "dry", "combination", "normal"]
CONCERNS = ["acne", "dark_spots", "redness", "large_pores", "wrinkles", "dryness"]

IMAGE_SIZE = 224
SEED = 42
