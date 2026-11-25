# model_loader.py - gives the app access to the saved model

import sys
from config import SRC_DIR

sys.path.append(SRC_DIR)   # so we can import ml/src/predict.py
import predict


def get_model():
    """The saved pipeline (it is loaded only once)."""
    return predict.load_model()


def get_model_info():
    """The content of ml/artifacts/model_info.json."""
    return predict.load_info()