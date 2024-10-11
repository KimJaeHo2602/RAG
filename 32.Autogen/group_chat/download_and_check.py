# filename: download_and_check.py
import pandas as pd

# 데이터 다운로드
url = "https://github.com/mwaskom/seaborn-data/raw/master/titanic.csv"
data = pd.read_csv(url)

# 데이터셋의 열 출력
print(data.columns.tolist())