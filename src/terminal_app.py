from __future__ import annotations

from dataclasses import dataclass
from joblib import load
from typing import List, Tuple


@dataclass
class FeatureInfo:
    """
    Metadata for a patient feature used in the terminal application.
    """
    name: str
    description: str
    data_type: str


class HeartApp:
    """
    Interactive Terminal application that loads a trained model 
    to predict heart disease risk based on user input.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        # Load the pre-trained model (joblib format)
        self.model = load(self.model_path)
        self.features: List[FeatureInfo] = [
            FeatureInfo("age", "Age in years", "int"),
            FeatureInfo("sex", "Sex (1 = male, 0 = female)", "int"),
            FeatureInfo("cp", "Chest pain type (0-3)", "int"),
            FeatureInfo("trestbps", "Resting blood pressure", "int"),
            FeatureInfo("chol", "Serum cholesterol mg/dl", "int"),
            FeatureInfo("fbs", "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)", "int"),
            FeatureInfo("restecg", "Resting electrocardiographic results (0-2)", "int"),
            FeatureInfo("thalach", "Max heart rate achieved", "int"),
            FeatureInfo("exang", "Exercise induced angina (1 = yes, 0 = no)", "int"),
            FeatureInfo("oldpeak", "ST depression induced by exercise", "float"),
            FeatureInfo("slope", "Slope of peak exercise ST segment (0-2)", "int"),
            FeatureInfo("ca", "Number of major vessels colored by fluoroscopy (0-4)", "int"),
            FeatureInfo("thal", "Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)", "int"),
        ]

    def prompt_for_inputs(self) -> List[float]:
        """Prompts the user sequentially for each required clinical feature."""
        values: List[float] = []
        for feature in self.features:
            while True:
                user_input = input(f"Enter {feature.description} ({feature.name}): ")
                try:
                    value = int(user_input) if feature.data_type == "int" else float(user_input)
                    values.append(value)
                    break
                except ValueError:
                    print("Invalid value. Please enter a numeric value.")
        return values

    def predict(self, values: List[float]) -> Tuple[int, float]:
        """Performs inference using the loaded model."""
        prediction = self.model.predict([values])[0]
        probability = self.model.predict_proba([values])[0][1]
        return int(prediction), float(probability)

    def run(self) -> None:
        """Main application loop for terminal interaction."""
        print("\n--- Heart Disease Prediction Tool ---")
        print("Please provide patient data below for risk assessment.\n")

        while True:
            # Step 1: Data entry
            values = self.prompt_for_inputs()
            
            # Step 2: Model Inference
            prediction, probability = self.predict(values)
            
            # Step 3: Output results
            risk_label = "LIKELY" if prediction == 1 else "UNLIKELY"
            print(f"\nResult: Heart disease is {risk_label}.")
            print(f"Confidence (Probability): {probability:.2%}\n")

            choice = input("Assess another patient? (y/n): ").strip().lower()
            if choice != "y":
                print("Exiting. Stay healthy!")
                break