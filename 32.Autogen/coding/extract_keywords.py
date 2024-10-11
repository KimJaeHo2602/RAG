# filename: extract_keywords.py
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# NLTK 리소스 다운로드
nltk.download('punkt')
nltk.download('stopwords')

sentence = "The quick brown fox jumps over the lazy dog."
tokens = word_tokenize(sentence)
stop_words = set(stopwords.words('english'))

keywords = [word for word in tokens if word.isalnum() and word.lower() not in stop_words]
print(keywords)