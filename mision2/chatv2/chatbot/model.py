import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
# CountVectorizer convierte texto en un vector
from sklearn.naive_bayes import MultinomialNB
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR,"model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR,"vectorizer.pkl")
ANSWERS_PATH = os.path.join(MODEL_DIR,"answers.pkl")
