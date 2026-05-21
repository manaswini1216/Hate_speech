import pickle
import re
import nltk
from nltk.corpus import stopwords
from neutralizer import neutralize_text
# ---------------- LOAD FILES ---------------- #

model = pickle.load(open(r"c:\Users\Manaswini\Downloads\project\hate_speech\backend\hate_speech_model.pkl", "rb"))

vectorizer = pickle.load(open(r"c:\Users\Manaswini\Downloads\project\hate_speech\backend\tfidf_vectorizer.pkl", "rb"))

toxic_words = pickle.load(open(r"c:\Users\Manaswini\Downloads\project\hate_speech\backend\toxic_words.pkl", "rb"))

# ---------------- CLEAN TOXIC WORDS ---------------- #

cleaned_toxic_words = set()

for word in toxic_words:

    cleaned = re.sub(r'[^a-zA-Z]', '', str(word).lower()).strip()

    if len(cleaned) > 2:
        cleaned_toxic_words.add(cleaned)

toxic_words = cleaned_toxic_words

# ---------------- NLTK ---------------- #

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stop_words.add("rt")

# ---------------- PREPROCESSING ---------------- #

def remove_entity(raw_text):

    entity_regex = r"&[^\s;]+;"

    text = re.sub(entity_regex, "", raw_text)

    return text


def change_user(raw_text):

    regex = r"@([^ ]+)"

    text = re.sub(regex, "user", raw_text)

    return text


def remove_url(raw_text):

    url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"

    text = re.sub(url_regex, '', raw_text)

    return text


def remove_noise_symbols(raw_text):

    text = raw_text.replace('"', '')
    text = text.replace("'", '')
    text = text.replace("!", '')
    text = text.replace("`", '')
    text = text.replace("..", '')

    return text


def remove_stopwords(raw_text):

    tokenize = nltk.word_tokenize(raw_text)

    text = [word for word in tokenize if word.lower() not in stop_words]

    text = " ".join(text)

    return text


def preprocess_text(text):

    clean = [text]

    clean = [change_user(text) for text in clean]

    clean = [remove_entity(text) for text in clean]

    clean = [remove_url(text) for text in clean]

    clean = [remove_noise_symbols(text) for text in clean]

    clean = [remove_stopwords(text) for text in clean]

    return clean[0]

# ---------------- TOXIC WORD EXTRACTION ---------------- #

def extract_toxic_words(text):

    words = text.lower().split()

    toxic_matches = []

    for word in words:

        cleaned_word = re.sub(r'[^a-zA-Z]', '', word)

        if cleaned_word in toxic_words and len(cleaned_word) > 2:
            toxic_matches.append(cleaned_word)

    return list(set(toxic_matches))

# ---------------- LABEL MAPPING ---------------- #

label_map = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neutral"
}

# ---------------- SEVERITY ---------------- #

def get_severity(prediction):

    if prediction == "Hate Speech":
        return "High"

    elif prediction == "Offensive Language":
        return "Medium"

    return "Low"

# ---------------- MAIN PREDICTION FUNCTION ---------------- #
def generate_explanation(prediction, toxic_words):

    if prediction == "Neutral":
        return "No harmful or offensive language detected."

    if toxic_words:

        toxic_text = ", ".join(toxic_words)

        return f"Flagged due to potentially toxic words: {toxic_text}"

    return "Text classified as potentially harmful by ML model."
def predict_text(text):

    cleaned_text = preprocess_text(text)

    vectorized_text = vectorizer.transform([cleaned_text])

    prediction = model.predict(vectorized_text)[0]

    probabilities = model.predict_proba(vectorized_text)[0]

    confidence = round(float(max(probabilities)) * 100, 2)

    prediction_label = label_map[prediction]
    neutralized_text = None

    if prediction_label != "Neutral":

      neutralized_text = neutralize_text(text)

    severity = get_severity(prediction_label)

    toxic_matches = extract_toxic_words(text)
    explanation = generate_explanation(
    prediction_label,
    toxic_matches
)

    result = {
        "prediction": prediction_label,
        "confidence": confidence,
        "severity": severity,
        "toxic_words": toxic_matches,
        "explanation": explanation,
        "neutralized_text": neutralized_text
    }

    return result

# ---------------- TEST ---------------- #

if __name__ == "__main__":

    text = "i hate you idiot"

    output = predict_text(text)

    print(output)