import pandas as pd ## lib for tabular data
import matplotlib.pyplot as plt ### lib for visualization
import seaborn as sns  ### lib for visualization
import re   ### for regex expressions
from nltk.corpus import stopwords ### NLTK for NLP taskss
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier

def preprocessing_step(text):
  text = text.lower()
  ### Remove any special charchter
  text =re.sub('[^a-zA-Z]',' ',text)
  tokens = word_tokenize(text)
  stop_words = set(stopwords.words('english'))
  filtered_tokens = [word for word in tokens if word not in stop_words]
  stemmer = PorterStemmer()
  stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
  stemmed_tokens = ' '.join(stemmed_tokens)
  return stemmed_tokens