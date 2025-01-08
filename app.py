from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pickle
import random
import json
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model and related data
MODEL_DIR = 'model'
model = load_model(f'{MODEL_DIR}/chatbot_model.h5')
lemmatizer = WordNetLemmatizer()
words = pickle.load(open(f'{MODEL_DIR}/words.pkl', 'rb'))
classes = pickle.load(open(f'{MODEL_DIR}/classes.pkl', 'rb'))

# Load intents and blood bank dataset
with open('intents.json') as json_file:
    intents = json.load(json_file)

with open('blood_bank.json') as json_file:
    blood_bank_data = json.load(json_file)

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
    error_threshold = 0.25  # Only consider results above this threshold
    results = [(i, r) for i, r in enumerate(res) if r > error_threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return classes[results[0][0]] if results else "unknown"

def get_response(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

def search_blood_bank(query):
    """
    Search the blood bank data for entries matching the specified city, district, or state.
    :param query: The location to search for (city, district, or state).
    :return: A list of matching blood bank data.
    """
    query = query.lower().strip()
    exact_matches = []
    partial_matches = []

    for bank in blood_bank_data:
        # Normalize the fields for comparison
        city = bank.get('City', '').lower()
        district = bank.get('District', '').lower()
        state = bank.get('State', '').lower()

        # Prioritize exact matches for City or District
        if query == city or query == district:
            exact_matches.append({
                "Blood Bank Name": bank.get("Blood Bank Name", "N/A"),
                "City": bank.get("City", "N/A"),
                "State": bank.get("State", "N/A"),
                "District": bank.get("District", "N/A"),
                "Address": bank.get("Address", "N/A"),
                "Contact No": bank.get("Contact No", "N/A"),
                "Service Time": bank.get("Service Time", "N/A"),
                "Helpline": bank.get("Helpline", "N/A")
            })
        # Add partial matches for fallback
        elif query in city or query in district or query in state:
            partial_matches.append({
                "Blood Bank Name": bank.get("Blood Bank Name", "N/A"),
                "City": bank.get("City", "N/A"),
                "State": bank.get("State", "N/A"),
                "District": bank.get("District", "N/A"),
                "Address": bank.get("Address", "N/A"),
                "Contact No": bank.get("Contact No", "N/A"),
                "Service Time": bank.get("Service Time", "N/A"),
                "Helpline": bank.get("Helpline", "N/A")
            })

    # Return exact matches first, then partial matches
    if exact_matches:
        return exact_matches
    elif partial_matches:
        return partial_matches
    else:
        return []




@app.route('/get', methods=['POST'])
def handle_request():
    """
    Handle user requests to fetch blood bank data or chatbot responses.
    :return: JSON response with results or error messages.
    """
    message = request.json.get('message', '').strip()
    if not message:
        return jsonify({"error": "Message cannot be empty."}), 400

    # Predict the intent
    intent = predict_class(message)

    if intent == "blood_bank_search":
        # Extract location from the message
        if "in" in message.lower():
            location_keywords = message.lower().split("in")[-1].strip()
        else:
            location_keywords = message

        # Search for blood banks based on city or district
        results = search_blood_bank(location_keywords)
        if results:
            return jsonify({"type": "blood_bank", "results": results})
        else:
            return jsonify({"type": "blood_bank", "error": f"No blood banks found in '{location_keywords}'."}), 404
    else:
        # Handle other chatbot intents
        response = get_response(intent)
        if response:
            return jsonify({"type": "chatbot", "response": response})
        else:
            return jsonify({"error": "Unable to process the request."}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
