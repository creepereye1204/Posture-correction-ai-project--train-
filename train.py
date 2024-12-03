import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import joblib

# CSV 파일 읽기
data = pd.read_csv("landmarks.csv")

# 데이터 확인 (필요시)
print(data.head())

# 입력 변수와 레이블 분리
X = data.drop(["y"], axis=1)  # 특징 변수
y = data["y"]  # 레이블

# 데이터셋을 학습셋과 테스트셋으로 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 생성
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 모델 학습
model.fit(X_train, y_train)

# 테스트 데이터에 대한 예측
y_pred = model.predict(X_test)

# 결과 출력
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 모델을 저장할 수도 있습니다 (선택사항)
joblib.dump(model, "random_forest_model.pkl")

# 랜덤 포레스트의 첫 번째 트리 시각화
plt.figure(figsize=(20, 10))
plot_tree(model.estimators_[0], feature_names=X.columns, filled=True, rounded=True)
plt.title("Visualizing Random Forest - Decision Tree")
plt.show()
