"""Flask application for the Vardaan breast tumor classification project."""

from __future__ import annotations

import logging
import os
import pickle
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, url_for

from chatbot import chatbot_response

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "Breast_cancer_model.pkl"))
DATABASE_PATH = Path(
    os.getenv("DATABASE_PATH", BASE_DIR / "instance" / "feedback.db")
)

FEATURE_NAMES = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_model(model_path: Path) -> Any:
    """Load the trusted, locally stored scikit-learn model."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file was not found at '{model_path}'. "
            "Set MODEL_PATH or retrain the model."
        )

    with model_path.open("rb") as model_file:
        return pickle.load(model_file)


model = load_model(MODEL_PATH)


def init_db() -> None:
    """Create the feedback table when it does not already exist."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                navigation INTEGER,
                design TEXT,
                design_suggestions TEXT,
                useful TEXT,
                accuracy INTEGER,
                trust TEXT,
                features TEXT,
                chatbot TEXT,
                general_feedback TEXT
            )
            """
        )


def parse_prediction_input(form_data: Any) -> np.ndarray:
    """Validate and convert the 30 submitted form values."""
    values: list[float] = []

    for feature in FEATURE_NAMES:
        raw_value = str(form_data.get(feature, "")).strip()

        if not raw_value:
            raise ValueError(f"Missing required value: {feature}")

        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"Invalid numerical value for {feature}") from exc

        if not np.isfinite(value):
            raise ValueError(f"Value for {feature} must be finite")

        values.append(value)

    return np.asarray(values, dtype=float).reshape(1, -1)


def create_app() -> Flask:
    """Application factory used by local development, tests, and Gunicorn."""
    flask_app = Flask(__name__, static_url_path="/static")
    flask_app.config["JSON_SORT_KEYS"] = False

    init_db()

    @flask_app.get("/")
    def home():
        return render_template("index.html")

    @flask_app.get("/about")
    def about():
        return render_template("about.html")

    @flask_app.get("/prevention")
    def prevention():
        return render_template("prevention.html")

    @flask_app.get("/precautions")
    def precautions():
        return render_template("precautions.html")

    @flask_app.get("/check_risk")
    def check_risk():
        return render_template("check_risk.html")

    @flask_app.get("/contact")
    def contact():
        return render_template("contact.html")

    @flask_app.get("/chatbot")
    def chatbot_page():
        return render_template("chatbot.html")

    @flask_app.post("/chatbot_response")
    def chatbot_reply():
        payload = request.get_json(silent=True) or {}
        user_message = str(payload.get("message", "")).strip()

        if not user_message:
            return jsonify({"response": "Please enter a message."}), 400

        reply, status_code = chatbot_response(user_message)
        return jsonify({"response": reply}), status_code

    @flask_app.route("/feedback", methods=["GET", "POST"])
    def feedback():
        if request.method == "GET":
            return render_template("feedback.html")

        feedback_values = (
            request.form.get("navigation"),
            request.form.get("design"),
            request.form.get("design_suggestions"),
            request.form.get("useful"),
            request.form.get("accuracy"),
            request.form.get("trust"),
            request.form.get("features"),
            request.form.get("chatbot"),
            request.form.get("general_feedback"),
        )

        with sqlite3.connect(DATABASE_PATH) as connection:
            connection.execute(
                """
                INSERT INTO feedback (
                    navigation,
                    design,
                    design_suggestions,
                    useful,
                    accuracy,
                    trust,
                    features,
                    chatbot,
                    general_feedback
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                feedback_values,
            )

        return redirect(url_for("thank_you"))

    @flask_app.get("/thank_you")
    def thank_you():
        return render_template("thank_you.html")

    @flask_app.post("/predict")
    def predict():
        try:
            input_array = parse_prediction_input(request.form)
            prediction = int(model.predict(input_array)[0])
        except ValueError as exc:
            logger.info("Prediction validation failed: %s", exc)
            return (
                render_template(
                    "check_risk.html",
                    prediction_text=(
                        "⚠️ Please enter valid numerical values in all 30 fields."
                    ),
                ),
                400,
            )
        except Exception:
            logger.exception("Unexpected prediction failure")
            return (
                render_template(
                    "check_risk.html",
                    prediction_text=(
                        "⚠️ The prediction service is temporarily unavailable."
                    ),
                ),
                500,
            )

        result = (
            "Malignant (Cancerous)"
            if prediction == 1
            else "Benign (Non-Cancerous)"
        )
        return render_template("result.html", result=result)

    @flask_app.get("/health")
    def health():
        return jsonify(
            {
                "status": "healthy",
                "model_loaded": model is not None,
                "expected_features": len(FEATURE_NAMES),
            }
        )

    return flask_app


app = create_app()


if __name__ == "__main__":
    debug_enabled = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_enabled)
