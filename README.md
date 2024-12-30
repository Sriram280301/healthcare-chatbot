
# Healthcare Chatbot

A machine learning-based healthcare chatbot designed to interact with users and provide responses based on predefined intents. This project uses Flask for the backend, TensorFlow for model loading and prediction, and NLTK for natural language processing.

## Features

- **AI-based chatbot**: Responds to user queries using a trained deep learning model.
- **Flask web application**: Provides an interactive web interface for users to chat with the bot.
- **Intent classification**: Classifies user input into predefined categories and returns an appropriate response.
- **Real-time interaction**: Handles real-time communication with the chatbot via an API endpoint.

## Project Structure

```
├── app.py                  # Main Flask application file
├── model/                  # Contains trained models and files
│   ├── chatbot_model.h5    # Trained deep learning model
│   ├── words.pkl           # Tokenized words
│   └── classes.pkl         # Intent categories
├── src/                    # Frontend files
│   └── index.html          # HTML file for the chatbot interface
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Requirements

- Python 3.11+
- TensorFlow
- Flask
- NLTK
- NumPy

You can install all the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Installation

Follow these steps to set up the project:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/iamraghavan/healthcare-chatbot.git
   cd healthcare-chatbot
   ```

2. **Set up a virtual environment** (optional, but recommended):

   - **Windows**:
     ```bash
     python -m venv venv
     .env\Scriptsctivate
     ```

   - **macOS/Linux**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask app**:

   ```bash
   python app.py
   ```

5. Open your browser and go to `http://127.0.0.1:5000` to interact with the chatbot.

## Usage

- You can interact with the chatbot directly in the web interface or send POST requests to the `/get` API endpoint.
- To use the API, send a POST request with the message, for example using **Postman** or **cURL**:

   **POST request**:
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{"message": "Hello"}' http://127.0.0.1:5000/get
   ```

   **Response**:
   ```json
   {
     "response": "Hello! How can I assist you today?"
   }
   ```

## Training the Model

If you want to train your own model, follow these steps:

1. Prepare your training data in the `intents.json` format.
2. Run the `train.py` script to train the model:

   ```bash
   python train.py
   ```

   This will generate the necessary files (`chatbot_model.h5`, `words.pkl`, `classes.pkl`) which are used by the chatbot application.

## Notes

- Ensure that you have **NLTK**'s **wordnet** data downloaded. You can do this by running:

  ```python
  import nltk
  nltk.download('wordnet')
  ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to contribute to this project by opening issues or pull requests. If you find a bug or have any suggestions, please let me know!
