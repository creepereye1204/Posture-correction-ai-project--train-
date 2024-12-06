import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # RandomForestClassifier 모듈 임포트
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import joblib

# 데이터 로드
data = pd.read_csv("posture.csv")

# 데이터 확인
print(data.head())

# Feature와 Label 구분
X = data.drop(["Label"], axis=1)
y = data["Label"]

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest 모델 생성
model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)  # n_estimators 추가

# 모델 학습
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 정확도 및 분류 리포트 출력
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 모델 저장
joblib.dump(model, "random_forest_model.pkl")  # 모델 이름 수정

# 첫 번째 결정 트리 시각화 (랜덤 포레스트는 여러 트리를 가지므로 첫 번째 트리만 시각화)
plt.figure(figsize=(20, 10))
plot_tree(
    model.estimators_[0], feature_names=X.columns, filled=True, rounded=True, class_names=["Normal", "Abnormal"]
)  # 첫 번째 트리 시각화
plt.title("Visualizing Random Forest")
plt.show()

# 각 리프 노드에서 예측된 클래스 값 출력
n_trees = len(model.estimators_)
for i in range(n_trees):
    tree = model.estimators_[i]
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    value = tree.tree_.value

    for node in range(n_nodes):
        if children_left[node] != children_right[node]:  # 리프 노드가 아닐 경우
            continue
        # 클래스 예측 값 (리프 노드의 가장 높은 값)
        predicted_class = value[node].argmax()
        print(f"Tree {i}, Node {node}: Predicted class = {predicted_class} (0: Normal, 1: Abnormal)")
