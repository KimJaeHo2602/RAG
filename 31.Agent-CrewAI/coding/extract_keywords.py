# filename: extract_keywords.py
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# NLTK 리소스 다운로드 (처음 한 번만 실행)
nltk.download('punkt')
nltk.download('stopwords')

# 예시 문장
sentence = "Artificial intelligence is transforming the way we live and work."

# 단어 토큰화
words = word_tokenize(sentence)

# 불용어 제거
stop_words = set(stopwords.words('english'))
keywords = [word for word in words if word.isalnum() and word.lower() not in stop_words]

print(keywords)