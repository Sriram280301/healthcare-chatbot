import os
import json
import random
import base64
import re
import requests
import speech_recognition as sr
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Restrict CORS for security

# Load Google Gemini API Key from environment variable
GEMINI_API_KEY = "AIzaSyDvTUXs9AXShlhtSJLAvLy4PjYEP0Z8saw"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

# Load datasets
with open('intents.json') as json_file:
    intents = json.load(json_file)

with open('blood_bank.json') as json_file:
    blood_bank_data = json.load(json_file)

with open('hospital.json') as json_file:
    hospital_data = json.load(json_file)["Sheet1"]

with open('health_insurance_schemes.json') as json_file:
    health_schemes_data = json.load(json_file)["insurance_schemes"]

# Medical keywords for Gemini AI
MEDICAL_KEYWORDS = ["doctor", "medicine", "hospital", "disease", "health", "treatment",
                    "symptoms", "diagnosis", "surgery", "blood", "insurance", "pharmacy", "remedies"]

# Location extraction regex patterns
LOCATION_PATTERNS = [
    r"\b(?:in|around|near|at|banks in)\s+([\w\s]+)",
    r"\b([A-Z][a-z]+)\b"
]

def extract_location(message):
    """Extracts a location keyword from a user query."""
    for pattern in LOCATION_PATTERNS:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None

def is_medical_question(message):
    """Check if the message contains any medical-related keywords."""
    return any(keyword in message.lower() for keyword in MEDICAL_KEYWORDS)

def dynamic_search(data, location, search_fields):
    """Search for matching records in a dataset based on location and fields."""
    results = []
    location_lower = location.lower()
    for entry in data:
        for field in search_fields:
            if field in entry and location_lower in entry[field].lower():
                results.append(entry)
                break
    return results

def search_healthcare_schemes():
    """Retrieve all healthcare schemes."""
    return health_schemes_data if health_schemes_data else []

def get_gemini_response(message):
    """Fetch a response from Gemini API for medical-related queries."""
    if not GEMINI_API_KEY:
        return "Error: Gemini API key is missing."

    payload = {"contents": [{"parts": [{"text": message}]}]}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(GEMINI_API_URL, json=payload, headers=headers, timeout=10)
        response.raise_for_status()  # Raises error for non-200 responses
        result = response.json()

        if "candidates" in result and result["candidates"]:
            full_text = result["candidates"][0].get("content", {}).get("parts", [{}])[0].get("text", "")
            return full_text.strip() if full_text else "I'm sorry, I couldn't find an answer."
        return "I'm sorry, I couldn't find an answer."

    except requests.exceptions.RequestException as e:
        return f"Error communicating with Gemini API: {str(e)}"

@app.route('/get', methods=['POST'])
def handle_request():
    """Chatbot API endpoint."""
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "Invalid request. Please provide a message."}), 400

    message = data['message'].strip()
    location = extract_location(message)

    # Blood Bank Search
    if "blood bank" in message.lower():
        if location:
            results = dynamic_search(blood_bank_data, location, ["DISTRICT", "STATE", "ADDRESS"])
            return jsonify({"type": "blood_bank", "location": location, "results": results[:3]}) if results else jsonify({"error": f"No blood banks found in '{location}'."}), 404
        return jsonify({"error": "Please specify a location (e.g., 'Find blood banks in Mumbai')."}), 400

    # Hospital Search
    if "hospital" in message.lower():
        if location:
            results = dynamic_search(hospital_data, location, ["DISTRICT", "STATE", "ADDRESS"])
            return jsonify({"type": "hospital", "location": location, "results": results[:3]}) if results else jsonify({"error": f"No hospitals found in '{location}'."}), 404
        return jsonify({"error": "Please specify a location (e.g., 'Hospitals in Bangalore')."}), 400

    # Health Insurance Schemes
    if "health insurance" in message.lower() or "medical scheme" in message.lower():
        results = search_healthcare_schemes()
        return jsonify({"type": "healthcare_schemes", "results": results}) if results else jsonify({"error": "No healthcare schemes found."}), 404

    # Other medical-related questions (Use Gemini API)
    if is_medical_question(message):
        response = get_gemini_response(message)
        return jsonify({"type": "chatbot", "response": response}), 200

    return jsonify({"response": "Sorry, I couldn't understand that."}), 200

@app.route('/get-voice', methods=['POST'])
def get_voice():
    """Speech-to-text API endpoint."""
    data = request.get_json()
    audio_base64 = data.get('audio', '')

    if not audio_base64:
        return jsonify({"error": "Audio data not provided."}), 400

    audio_data = base64.b64decode(audio_base64)
    temp_audio_file = 'temp_audio.wav'

    with open(temp_audio_file, 'wb') as audio_file:
        audio_file.write(audio_data)

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(temp_audio_file) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)

            # Process the recognized text like a standard request
            return handle_request().jsonify({"message": text})

    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand audio."}), 400
    except sr.RequestError as e:
        return jsonify({"error": f"Speech recognition service error: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
