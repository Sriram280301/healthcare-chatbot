import os
import json
import re
import base64
import requests
import speech_recognition as sr
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins (for testing)

# Load Google Gemini API Key securely from environment variable
GEMINI_API_KEY = "AIzaSyDvTUXs9AXShlhtSJLAvLy4PjYEP0Z8saw"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# Load datasets
with open('intents.json') as f:
    intents = json.load(f)

with open('blood_bank.json') as f:
    blood_bank_data = json.load(f)

with open('hospital.json') as f:
    hospital_data = json.load(f)["Sheet1"]

with open('health_insurance_schemes.json') as f:
    health_schemes_data = json.load(f)["insurance_schemes"]

# Medical keywords for triggering Gemini AI


MEDICAL_KEYWORDS = {
    "doctor", "medicine", "disease", "health", "treatment",
    "symptoms", "diagnosis", "surgery", "insurance", "pharmacy",
    "remedies", "fever", "cancer", "infection", "virus", "bacteria",
    "diabetes", "hypertension", "stroke", "heart attack", "cardiology",
    "neurology", "orthopedics", "oncology", "radiology", "pathology",
    "allergy", "asthma", "arthritis", "therapy", "rehabilitation", "mental health",
    "depression", "anxiety", "psychiatry", "psychology", "vaccination",
    "immunization", "pediatrics", "gynecology", "pregnancy", "maternity",
    "obstetrics", "dermatology", "ophthalmology", "dental", "ENT", "urology",
    "gastroenterology", "nephrology", "pulmonology", "endocrinology",
    "hematology", "toxicology", "first aid", "emergency", "ICU", "ambulance",
    "antibiotics", "antivirals", "painkillers", "prescription", "pharmacology",
    "nutrition", "diet", "exercise", "physiotherapy", "homeopathy",
    "alternative medicine", "acupuncture", "nursing", "caregiver", "medication",
    "organ transplant", "dialysis", "vaccines", "COVID-19",
    "HIV", "AIDS", "cough", "cold", "flu", "migraine", "cholesterol",
    "obesity", "stroke", "cardiac arrest", "sugar level",
    "chronic disease", "acute illness", "genetic disorder", "therapy",
    "telemedicine", "healthcare provider", "nurse", "clinic", "medical test",
    "X-ray", "CT scan", "MRI", "ultrasound", "biopsy", "chemotherapy",
    "radiation therapy", "surgical procedure", "infection control",
    "pain management", "prognosis", "prostate", "liver disease", "kidney disease",
    "Alzheimer", "Parkinson", "mental disorder", "speech therapy", "eye care",
    "vision", "hearing loss", "sleep disorder", "skin disease", "wound care",
    "burns", "fracture", "bone marrow", "immunotherapy", "gene therapy",
    "rehab center", "physical health", "public health", "epidemic",
    "pandemic", "waterborne diseases", "food poisoning", "drug allergy",
    "anesthesia", "sterilization", "infection prevention", "home nursing",
    "medical tourism", "quarantine", "isolation", "clinical trials",
    "health policy", "medical research", "bioethics", "euthanasia",
    "fertility", "sexual health", "birth control", "contraception",
    "hormone therapy", "palliative care", "geriatrics", "hospice care",
    "sleep apnea", "dementia", "osteoporosis", "tuberculosis", "vaccination schedule",
    "public health programs", "medical ethics", "DNA testing", "forensic medicine",
    "sudden infant death syndrome", "prenatal care", "postnatal care",
    "spinal cord injury", "physical disability", "dehydration",
    "malnutrition", "hypothermia", "hyperthermia", "burn treatment",
    "cosmetic surgery", "plastic surgery", "dermatitis", "eczema", "psoriasis",
    "lupus", "rheumatology", "chronic pain", "autoimmune diseases", "dysentery",
    "malaria", "jaundice", "brain tumor", "Zika virus", "Ebola", "Marburg virus",
    "Monkeypox", "Nipah virus", "Dengue", "Chikungunya", "Hantavirus", "Rabies",
    "Leprosy", "Lyme disease", "Chronic obstructive pulmonary disease (COPD)",
    "Pulmonary fibrosis", "Pneumonia", "Legionnaires' disease", "Multiple sclerosis (MS)",
    "Epilepsy", "Huntington’s disease", "Guillain-Barré syndrome", "Myasthenia gravis",
    "Amyotrophic lateral sclerosis (ALS)", "Meningitis", "Encephalitis", "Aneurysm",
    "Aortic dissection", "Arrhythmia", "Pericarditis", "Endocarditis", "Venous thromboembolism (VTE)",
    "Deep vein thrombosis (DVT)", "Pulmonary embolism", "Crohn’s disease", "Ulcerative colitis",
    "Celiac disease", "Irritable bowel syndrome (IBS)", "Hepatitis (A, B, C, D, E)", "Pancreatitis",
    "Gallstones", "Gastroesophageal reflux disease (GERD)", "Peptic ulcer disease", "Leukemia",
    "Lymphoma", "Melanoma", "Sarcoma", "Colorectal cancer", "Pancreatic cancer", "Bladder cancer",
    "Ovarian cancer", "Cervical cancer", "Esophageal cancer", "Stomach cancer", "Thyroid cancer",
    "Bone cancer", "Abdominal aortic aneurysm", "Achilles tendinopathy", "Acne", "Acute cholecystitis",
    "Acute lymphoblastic leukaemia", "Acute myeloid leukaemia", "Atopic eczema", "Acute pancreatitis",
    "Alcohol-related liver disease", "Allergic rhinitis", "Anal cancer", "Angina", "Arterial thrombosis",
    "Back problems", "Bacterial vaginosis", "Benign prostate enlargement", "Bile duct cancer",
    "Binge eating", "Bipolar disorder", "Bowel cancer", "Bowel incontinence",
    "Bowel polyps", "Brain tumours", "Breast cancer", "Bronchiectasis", "Bronchitis", "Bulimia nervosa",
    "Carcinoid syndrome", "Cardiovascular disease", "Carpal tunnel syndrome", "Coronary heart disease",
    "Costochondritis", "Croup", "Cystic fibrosis", "Cystitis", "Catarrh", "Cellulitis", "Cerebral palsy",
    "Cervical spondylosis", "Chest infection", "Chickenpox", "Chilblains", "Chlamydia",
    "Chronic fatigue syndrome", "Chronic kidney disease", "Chronic lymphocytic leukaemia",
    "Chronic myeloid leukaemia", "Chronic pancreatitis", "Cirrhosis", "Clavicle fracture",
    "Clostridium difficile", "Cold sore", "Coma", "Common cold", "Concussion",
    "Congenital heart disease", "Conjunctivitis", "Constipation", "Diabetic retinopathy",
    "Discoid eczema", "Ebola virus disease", "Escherichia coli", "Ewing sarcoma",
    "Eye cancer", "Fibroids", "Fibromyalgia", "Flu", "Frozen shoulder", "Gallbladder cancer",
    "Gastroenteritis", "Gastro-oesophageal reflux disease (GORD)", "Gout", "Genital herpes",
    "Genital warts", "Germ cell tumours", "Functional neurological disorder",
    "Haemorrhoids (piles)", "Hand, foot and mouth disease", "Hay fever", "Head and neck cancer",
    "Heart block", "Hodgkin lymphoma", "Idiopathic pulmonary fibrosis", "Insomnia", "Itchy skin",
    "Kaposi’s sarcoma", "Kidney cancer", "Kidney infection", "Kidney stones", "Labyrinthitis",
    "Laryngeal cancer", "Lichen planus", "Lipoedema", "Liver cancer", "Liver tumours",
    "Lymphogranuloma venereum (LGV)", "Lung cancer", "Loss of libido", "Malignant brain tumour",
    "Motor neurone disease", "Mouth cancer", "Non-Hodgkin lymphoma", "Oesophageal cancer",
    "Osteosarcoma", "Overactive thyroid", "Pancreatic cancer", "Pelvic inflammatory disease",
    "Raynaud’s phenomenon", "Rosacea", "Rheumatoid arthritis", "Respiratory syncytial virus (RSV)",
    "Shortness of breath", "Sickle cell disease", "Testicular cancer", "Trichomonas infection",
    "Urinary incontinence", "Urticaria (hives)", "Wilms’ tumour", "Womb cancer", "Yellow fever","dysentery","malaria","jaundice","brain tumor","covid-19","blood pressure"
}


# Location extraction regex patterns
LOCATION_PATTERNS = [r"\b(?:in|around|near|at|banks in)\s+([\w\s]+)", r"\b([A-Z][a-z]+)\b"]


def extract_location(message):
    """Extracts location from user input."""
    for pattern in LOCATION_PATTERNS:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def is_medical_question(message):
    """Checks if the message is related to medical queries."""
    return any(keyword in message.lower() for keyword in MEDICAL_KEYWORDS)


def search_data(dataset, location, search_fields):
    """Search for matching records based on location."""
    if not location:
        return []
    location_lower = location.lower()
    return [entry for entry in dataset if any(location_lower in entry.get(field, "").lower() for field in search_fields)]


def get_gemini_response(message):
    """Fetches a concise response from Gemini API for medical queries."""
    if not GEMINI_API_KEY:
        return "Error: Missing Gemini API key."

    payload = {"contents": [{"parts": [{"text": message}]}]}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(GEMINI_API_URL, json=payload, headers=headers, timeout=90)
        response.raise_for_status()
        result = response.json()

        if result.get("candidates"):
            full_text = result["candidates"][0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
            return ' '.join(full_text.split('. ')[:2])  # Return only the first one or two sentences

        return "I'm sorry, I couldn't find an answer."
    except requests.exceptions.RequestException as e:
        return f"Error communicating with Gemini API: {str(e)}"



@app.route('/get', methods=['POST'])
def handle_request():
    """Handles chatbot requests."""
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "Invalid request. Please provide a message."}), 400

    message = data['message'].strip()
    location = extract_location(message)

    # Blood Bank Search
    if "blood bank" in message.lower():
        results = search_data(blood_bank_data, location, ["DISTRICT", "STATE", "ADDRESS"])
        if results:
            return jsonify({"type": "blood_bank", "location": location, "results": results[:3]}), 200
        return jsonify({"error": f"No blood banks found in '{location}'."}), 404

    # Hospital Search
    if "hospital" in message.lower():
        results = search_data(hospital_data, location, ["DISTRICT", "STATE", "ADDRESS"])
        if results:
            return jsonify({"type": "hospital", "location": location, "results": results[:3]}), 200
        return jsonify({"error": f"No hospitals found in '{location}'."}), 404

    # Health Insurance Schemes
    if "health insurance" in message.lower() or "medical scheme" in message.lower():
        if health_schemes_data:
            return jsonify({"type": "healthcare_schemes", "results": health_schemes_data}), 200
        return jsonify({"error": "No healthcare schemes found."}), 404

    # Other medical-related questions (Use Gemini API)
    if is_medical_question(message):
        response = get_gemini_response(message)
        return jsonify({"type": "chatbot", "response": response}), 200

    return jsonify({"response": "Sorry, I couldn't understand that."}), 200


@app.route('/get-voice', methods=['POST'])
def get_voice():
    """Processes voice input and converts to text."""
    data = request.get_json()
    audio_base64 = data.get('audio', '')

    if not audio_base64:
        return jsonify({"error": "Audio data not provided."}), 400

    try:
        # Decode base64 audio
        audio_data = base64.b64decode(audio_base64)
        temp_audio_file = 'temp_audio.wav'

        # Save as temporary file
        with open(temp_audio_file, 'wb') as audio_file:
            audio_file.write(audio_data)

        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_audio_file) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)

            # Process the recognized text
            return jsonify({"message": text, "response": handle_request().json["response"]})

    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand audio."}), 400
    except sr.RequestError as e:
        return jsonify({"error": f"Speech recognition service error: {e}"}), 500

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Flask is working!"}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
