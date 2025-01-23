from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pickle
import random
import json
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
from flask_cors import CORS
import speech_recognition as sr
import pyttsx3
import base64

app = Flask(__name__)
CORS(app)

# Load the trained model and related data
MODEL_DIR = 'model'
model = load_model(f'{MODEL_DIR}/chatbot_model.h5')
lemmatizer = WordNetLemmatizer()
words = pickle.load(open(f'{MODEL_DIR}/words.pkl', 'rb'))
classes = pickle.load(open(f'{MODEL_DIR}/classes.pkl', 'rb'))

# Load intents, hospital, and blood bank datasets
with open('intents.json') as json_file:
    intents = json.load(json_file)

with open('blood_bank.json') as json_file:
    blood_bank_data = json.load(json_file)

with open('hospital.json') as json_file:
    hospital_data = json.load(json_file)["Sheet1"]

# Helper functions
def clean_up_sentence(sentence):
    sentence_words = tf.keras.preprocessing.text.text_to_word_sequence(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bow(sentence):
    sentence_words = clean_up_sentence(sentence)
    return np.array([1 if word in sentence_words else 0 for word in words])

def predict_class(sentence):
    bow_data = bow(sentence)
    res = model.predict(np.array([bow_data]))[0]
    error_threshold = 0.25
    results = [(i, r) for i, r in enumerate(res) if r > error_threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return classes[results[0][0]] if results else "unknown"

def get_response(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

def search_blood_bank(query):
    query = query.lower().strip()
    matches = [
        {
            "Hospital Name": bank.get("HospitalName", "N/A"),
            "District": bank.get("DISTRICT", "N/A"),
            "Address": bank.get("ADDRESS", "N/A"),
            "Contact No": bank.get("HOSPITAL CONTACT NO", "N/A"),
        }
        for bank in blood_bank_data
        if query in bank.get('DISTRICT', '').lower() or query in bank.get('HospitalName', '').lower()
    ]
    return matches

def search_hospital(query):
    query = query.lower().strip()
    matches = [
        {
            "Hospital Name": hospital.get("HospitalName", "N/A"),
            "District": hospital.get("DISTRICT", "N/A"),
            "Address": hospital.get("ADDRESS", "N/A"),
            "Contact No": hospital.get("HOSPITAL CONTACT NO", "N/A"),
        }
        for hospital in hospital_data
        if query in hospital.get('DISTRICT', '').lower() or query in hospital.get('HospitalName', '').lower()
    ]
    return matches

def search_healthcare_schemes():
    with open('health_insurance_schemes.json') as f:
        schemes_data = json.load(f)
    return schemes_data.get('health_insurance_schemes', [])

# API endpoints
@app.route('/get', methods=['POST'])
def handle_request():
    message = request.json.get('message', '').strip()
    if not message:
        return jsonify({"error": "Message cannot be empty."}), 400

    intent = predict_class(message)

    if intent == "blood_bank_search":
        location_keywords = message.lower().split("in")[-1].strip() if "in" in message.lower() else message
        results = search_blood_bank(location_keywords)
        if results:
            return jsonify({"type": "blood_bank", "results": random.sample(results, min(3, len(results)))})
        return jsonify({"type": "blood_bank", "error": f"No blood banks found for '{location_keywords}'."}), 404

    elif intent == "hospital_search":
        location_keywords = message.lower().split("in")[-1].strip() if "in" in message.lower() else message
        results = search_hospital(location_keywords)
        if results:
            return jsonify({"type": "hospital", "results": random.sample(results, min(3, len(results)))})
        return jsonify({"type": "hospital", "error": f"No hospitals found for '{location_keywords}'."}), 404

    elif intent == "health_insurance_info":
        results = search_healthcare_schemes()
        if results:
            return jsonify({"type": "healthcare_schemes", "results": results})
        return jsonify({"type": "healthcare_schemes", "error": "No healthcare schemes found."}), 404

    else:
        response = get_response(intent)
        if response:
            return jsonify({"type": "chatbot", "response": response})
        return jsonify({"error": "Unable to process the request."}), 500

@app.route('/get-voice', methods=['POST'])
def get_voice():
    audio_base64 = request.json.get('audio', None)
    if not audio_base64:
        return jsonify({"error": "Audio data not provided."}), 400

    audio_data = base64.b64decode(audio_base64)
    temp_audio_file = 'temp_audio.wav'
    with open(temp_audio_file, 'wb') as audio_file:
        audio_file.write(audio_data)

    r = sr.Recognizer()
    try:
        with sr.AudioFile(temp_audio_file) as source:
            audio = r.record(source)
            text = r.recognize_google(audio)
            return jsonify({"transcription": text})
    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand audio."}), 400
    except sr.RequestError as e:
        return jsonify({"error": f"Speech recognition service error: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
