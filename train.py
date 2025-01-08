import tensorflow as tf
import json
import numpy as np
import random
import pickle
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

# Ensure required directories exist
MODEL_DIR = 'model'
os.makedirs(MODEL_DIR, exist_ok=True)

# Load intents JSON
INTENTS_FILE = 'intents.json'

if not os.path.exists(INTENTS_FILE):
    raise FileNotFoundError(f"{INTENTS_FILE} not found.")

with open(INTENTS_FILE, 'r') as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()
label_encoder = LabelEncoder()

# Initialize preprocessing variables
words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Process intents data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        tokens = tf.keras.preprocessing.text.text_to_word_sequence(pattern)
        words.extend(tokens)
        documents.append((tokens, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Adding the blood bank search patterns
for pattern in ["Blood bank in {city}", "Blood bank in {state}", "Where is blood bank in {city}?"]:
    tokens = tf.keras.preprocessing.text.text_to_word_sequence(pattern)
    words.extend(tokens)
    documents.append((tokens, 'blood_bank_search'))
    if 'blood_bank_search' not in classes:
        classes.append('blood_bank_search')

# Lemmatize and sort unique words
words = sorted(set([lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]))
classes = sorted(set(classes))

# Save words and classes for future use
pickle.dump(words, open(os.path.join(MODEL_DIR, 'words.pkl'), 'wb'))
pickle.dump(classes, open(os.path.join(MODEL_DIR, 'classes.pkl'), 'wb'))

# Prepare training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = [1 if w in doc[0] else 0 for w in words]
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle and convert to numpy arrays
random.shuffle(training)
training = np.array(training, dtype=object)
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Build the model
def build_model(input_shape, output_shape):
    model = Sequential([
        Dense(128, input_shape=(input_shape,), activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(output_shape, activation='softmax')
    ])
    model.compile(
        optimizer=SGD(learning_rate=0.01, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Create and train the model
model = build_model(len(train_x[0]), len(classes))
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save the trained model
MODEL_FILE = os.path.join(MODEL_DIR, 'chatbot_model.h5')
model.save(MODEL_FILE)
print(f"Model training complete and saved to '{MODEL_FILE}'")
