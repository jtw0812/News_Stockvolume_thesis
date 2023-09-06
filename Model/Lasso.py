import sys,os
current_directory = os.getcwd()
sys.path.append(current_directory)
import path

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.metrics import accuracy_score

class lasso():
    def __init__(self, df, stock):
        self.df = df
        self.stock = stock

    def run_lasso(self, num_cv_iterations, n_splits, random_state):
        # Feature(독립변수)와 Target(종속변수)를 나눕니다.
        X = self.df.drop(columns=['Date', 'Target'])  # Feature로 lda_topic 컬럼들을 사용
        y = self.df['Target']  # Target으로 'target' 컬럼을 사용

        X = np.array(X)
        y = np.array(y)

        # 10-겹 교차 검증을 위한 준비
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        # 각 테스트 세트의 MSE를 저장할 리스트
        mse_scores = []

        # 각 교차 검증에서의 모델 가중치를 저장할 리스트
        model_weights = []

        # 10-겹 교차 검증을 수행
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # 모델 초기화 (예: 라쏘 회귀)
            lasso_model = Lasso(alpha=1.0)

            # 모델을 학습
            lasso_model.fit(X_train, y_train)

            # 테스트 세트에 대한 예측
            y_pred = lasso_model.predict(X_test)

            # MSE 계산 및 저장
            mse = mean_squared_error(y_test, y_pred)
            mse_scores.append(mse)

            # 모델 가중치 저장
            model_weights.append(lasso_model.coef_)

        # 각 테스트 세트에서의 MSE를 출력
        for i, mse in enumerate(mse_scores):
            print(f"Fold {i+1} MSE: {mse}")

        # 10-겹 교차 검증의 평균 MSE 계산
        average_mse = np.mean(mse_scores)
        print(f"Average MSE: {average_mse}")

        # 모델 가중치 평균 계산
        average_weights = np.mean(model_weights, axis=0)

        # 최종 LASSO 모델 생성
        final_lasso_model = Lasso(alpha=1.0)
        final_lasso_model.coef_ = average_weights  # 가중치 설정
        self.subject_importance = list(average_weights)
        self.model = final_lasso_model
    
    def peak_day_comparison(self):
        top_5_percent = self.df[self.df['Target'] >= self.df['Target'].quantile(0.95)]
        X = top_5_percent.iloc[:, 2:32]
        top_5_percent['Target_predict'] = self.model.predict(X)
        threshold = 0.3
        top_5_percent['Success'] = top_5_percent['Target_predict'] >= (top_5_percent['Target'] * (threshold))

        # 정확도 계산
        accuracy = accuracy_score(top_5_percent['Success'], [True] * len(top_5_percent))
        self.peakday_accuracy = accuracy
        print(accuracy)
    
    # def lasso_stock_predict():
