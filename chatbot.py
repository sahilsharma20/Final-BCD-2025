"""Optional informational chatbot used by the Flask application."""

from __future__ import annotations

import os

CHATBOT_INSTRUCTIONS = """
You are the informational assistant for an educational breast tumor
classification project.

Rules:
- Explain the software, dataset, input features, and general breast-cancer
  awareness information in clear language.
- Never diagnose a user or interpret personal symptoms as a diagnosis.
- Never recommend treatment or medication.
- State that the machine-learning prediction is educational and is not a
  substitute for a qualified clinician, pathology, imaging, or screening.
- For urgent or alarming symptoms, advise the user to contact a qualified
  healthcare professional or local emergency service.
- Keep responses concise and avoid overstating model accuracy.
""".strip()


def chatbot_response(user_message: str) -> tuple[str, int]:
    """Return an OpenAI response when optional configuration is available."""
    message = user_message.strip()

    if not message:
        return "Please enter a message.", 400

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model_name = os.getenv("OPENAI_MODEL", "").strip()

    if not api_key or not model_name:
        return (
            "The optional chatbot is not configured. "
            "The prediction tool and educational pages remain available.",
            503,
        )

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model_name,
            instructions=CHATBOT_INSTRUCTIONS,
            input=message,
        )
        output = response.output_text.strip()

        if not output:
            return "The chatbot returned an empty response. Please try again.", 502

        return output, 200
    except ImportError:
        return (
            "The optional chatbot dependency is not installed. "
            "Install the project requirements and restart the app.",
            503,
        )
    except Exception:
        return (
            "The chatbot is temporarily unavailable. "
            "Please use the educational pages or try again later.",
            502,
        )
