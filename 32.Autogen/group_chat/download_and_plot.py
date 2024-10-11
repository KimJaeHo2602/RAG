# filename: download_and_plot.py
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 다운로드
url = "https://github.com/mwaskom/seaborn-data/raw/master/titanic.csv"
data = pd.read_csv(url)

# 데이터셋의 열 출력
print(data.columns.tolist())

# 'age'와 'pclass' 간의 관계 시각화
plt.figure(figsize=(10, 6))
plt.scatter(data['pclass'], data['age'], alpha=0.5)
plt.title('Relationship between Age and Passenger Class')
plt.xlabel('Passenger Class (pclass)')
plt.ylabel('Age')
plt.grid(True)

# 차트 파일로 저장
plt.savefig('age_pclass_relationship.png')
plt.close()