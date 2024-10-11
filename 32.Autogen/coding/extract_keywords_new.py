# filename: extract_keywords_new.py
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# NLTK 리소스 다운로드 (이미 다운로드된 경우 주석 처리)
# nltk.download('punkt')
# nltk.download('stopwords')

sentence = "Artificial intelligence is transforming the way we live and work."
tokens = word_tokenize(sentence)
stop_words = set(stopwords.words('english'))

keywords = [word for word in tokens if word.isalnum() and word.lower() not in stop_words]
print(keywords)