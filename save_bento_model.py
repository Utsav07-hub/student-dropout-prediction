import joblib
import bentoml

# Load trained model
model = joblib.load("models/best_model.pkl")


# Save model to BentoML Model Store
bentoml.sklearn.save_model(
    "student_dropout_rf",
    model
)

print("Model saved to BentoML Model Store")