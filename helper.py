import re # regex
from nltk.corpus import stopwords # NLTK for NLP tasks
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


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
