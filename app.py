from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import random
import json
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
app = Flask(__name__, template_folder='src')


# Load model and data
model = load_model('model/chatbot_model.h5')
lemmatizer = WordNetLemmatizer()
words = pickle.load(open('model/words.pkl', 'rb'))
classes = pickle.load(open('model/classes.pkl', 'rb'))

# Utility functions
def clean_up_sentence(sentence):
    sentence_words = tf.keras.preprocessing.text.text_to_word_sequence(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bow(sentence):
    sentence_words = clean_up_sentence(sentence)
    return np.array([1 if word in sentence_words else 0 for word in words])

def predict_class(sentence):
    bow_data = bow(sentence)
    res = model.predict(np.array([bow_data]))[0]
    return classes[np.argmax(res)]

def get_response(tag):
    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get', methods=['POST'])
def chatbot_response():
    message = request.json.get('message')
    intent = predict_class(message)
    response = get_response(intent)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
