import joblib
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
def clean_text(text):
    text = text.lower()
    text = ''.join(c for c in text if c not in string.punctuation)
    words = text.split()
    stop_words = stopwords.words('english')
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)
model, vectorizer = joblib.load("models/sms_model.pkl")

def predict_sms(message):
    cleaned = clean_text(message)
    vectorized = vectorizer.transform([cleaned])
    result = model.predict(vectorized)[0]
    return "Spam" if result == 1 else "Legit"
msg = input("Enter an SMS: ")
print("Prediction:", predict_sms(msg))
