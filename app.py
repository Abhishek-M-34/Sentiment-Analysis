from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

app = Flask(__name__) 

# Load model
def load_model():
    try:
        with open("sentiment.pkl",'rb') as file:
            model = pickle.load(file)
        
        vectorizer = None
        
        try:
            with open("vectorizer.pkl",'rb') as file:
                vectorizer = pickle.load(file)
        except Exception as e:
            print(f'Vectorizer not found or invalid {e}')
        
        return model,vectorizer
    except FileNotFoundError as e:
        print(f'Model not found or invalid {e}')
        return None,None
    except Exception as e:
        print(f'Error loading artifacts {e}')
        return None,None
    
model,vectorizer = load_model()

def preprocess(sentence_input):
    text = re.sub(r'[^0-9a-zA-Z]+',' ',sentence_input).split()
    words = [x.lower() for x in text if x not in stopwords.words('english')]
    lemma = WordNetLemmatizer()
    word = [lemma.lemmatize(word,'v') for word in words ]
    word = ' '.join(word)
    return word

def resultOutput(result):
    if result == 1:
        return "Positive"
    elif result == 0:
        return "Neutral"
    else:
        return "Negative"
    
def sentiment_prediction(sentence_input):
    try:
        sentence_value = preprocess(sentence_input) # possible error (float not used, cause it cames as sentence)
        if sentence_value is None:
            return "Enter Sentence to analyse sentiment."
    
        if vectorizer is not None and hasattr(vectorizer,'transform'):
            vectorized_data = vectorizer.transform(sentence_value)
        else:
            print("Vectorizer not available or invalid")

        prediction = model.predict(vectorized_data)
        probabilities = model.predict_proba(vectorized_data)
        predicted_sentiment = int(prediction[0])
        confidence = probabilities[0][predicted_sentiment]

        return predicted_sentiment,confidence 
    except Exception as e:
        return f"Prediction Error {e}"
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    try:
        sentence_input = request.form.get('sentence',"")

        if not sentence_input:
            return jsonify({
                'success': False,
                'error': 'Please enter the sentence first.'
            })
        result, confidence = sentiment_prediction(sentence_input)

        if isinstance(result, str) and (result.startswith("Error") or result.startswith("Prediction Error")):
            return jsonify({
                'success': False,
                'error': result
            })
        result_output = resultOutput(result)

        return jsonify({
            'success': True,
            'prediction': result_output,
            'confidence': f"{confidence:.2%}",
            'details': {
                'sentence': sentence_input,
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Unexpected error: {str(e)}'
        })
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)