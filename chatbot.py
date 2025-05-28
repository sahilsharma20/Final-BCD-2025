import openai
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify

app = Flask(__name__)  # Create the Flask app instance

# ✅ Load the .env file
load_dotenv()

# ✅ Get API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ Check if API key is loaded properly
if not openai.api_key:
    raise ValueError("⚠️ ERROR: OpenAI API Key is missing! Check .env file.")


@app.route('/chatbot_response', methods=['POST'])
# ✅ Fix: Define chatbot_response with a parameter
def chatbot_response(user_message):
    try:
        if not user_message:
            return "⚠️ No message provided."

        # OpenAI API request
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_message}]
        )
        
        # Extract chatbot reply
        return response.choices[0].message.content.strip()

    except openai.error.OpenAIError as e:
        return f"⚠️ OpenAI API Error: {str(e)}"
    except Exception as e:
        return f"⚠️ Server Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)