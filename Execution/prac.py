
import os
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, make_scorer
current_directory = os.getcwd()
data_path = current_directory + "\\data"
# 데이터프레임 생성 (앞서 생성한 데이터프레임을 사용)
result_df = pd.read_csv(data_path +"\\result.csv")
# 이 데이터프레임에는 Date, lda_topic_0, lda_topic_1, ..., lda_topic_19, target 컬럼이 있어야 합니다.

# Feature(독립변수)와 Target(종속변수)를 나눕니다.
X = result_df.drop(columns=['Date', 'Target'])  # Feature로 lda_topic 컬럼들을 사용
y = result_df['Target'] # Target으로 'target' 컬럼을 사용

# Lasso 회귀 모델 생성
lasso_model = Lasso(alpha=1.0)  # alpha는 L1 규제 강도 조절

# 100회의 10겹 교차 검증 수행
num_cv_iterations = 100
cv_object = KFold(n_splits=10, shuffle=True, random_state=42)

# 교차 검증을 통한 모델 평가 (MSE를 사용하여)
scorer = make_scorer(mean_squared_error, greater_is_better=False)
mse_scores = cross_val_score(lasso_model, X, y, cv=cv_object, scoring='neg_mean_squared_error', n_jobs=-1)
mse_scores = -mse_scores  # 스코어 값은 양수로 변환

# 각 교차 검증에서의 MSE 평균과 표준편차 출력
print(f"Mean MSE: {mse_scores.mean()}")
print(f"Std MSE: {mse_scores.std()}")