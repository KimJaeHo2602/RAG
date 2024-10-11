# filename: install_and_extract_keywords.py
import subprocess
import sys

# NLTK 설치
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'nltk'])

import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

sentence = "The quick brown fox jumps over the lazy dog."
tokens = word_tokenize(sentence)
stop_words = set(stopwords.words('english'))

keywords = [word for word in tokens if word.isalnum() and word.lower() not in stop_words]
print(keywords)