from pathlib import Path
from typing import Any

import bentoml
import joblib
import pandas as pd

from src.features import FEATURE_COLUMNS

MODEL_DIR = Path(__file__).resolve().parent / "models"


@bentoml.service
class StudentDropoutService:
    def __init__(self):
        # Load BentoML model
        # self.model = bentoml.sklearn.load_model("student_dropout_rf")
        self.model = joblib.load(MODEL_DIR / "best_model.pkl")
        # Load preprocessing objects
        self.preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")
        self.label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")    

    @bentoml.api
    def predict(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Predict student dropout status.
        """

        try:
            missing_features = [
                column for column in FEATURE_COLUMNS if column not in input_data
            ]
            if missing_features:
                return {
                    "status": "error",
                    "message": f"Missing required fields: {missing_features}",
                }

            # Convert JSON to DataFrame
            input_df = pd.DataFrame([{column: input_data[column] for column in FEATURE_COLUMNS}])

            # Apply preprocessing
            transformed_input = self.preprocessor.transform(input_df)

            # Make prediction
            prediction = self.model.predict(transformed_input)

            # Convert numeric prediction back to original label
            prediction_label = self.label_encoder.inverse_transform(prediction)[0]

            return {
                "prediction": prediction_label
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
            
            
# bentoml serve service:StudentDropoutService --reload
