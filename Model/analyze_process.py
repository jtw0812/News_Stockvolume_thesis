import sys,os
current_directory = os.getcwd()
sys.path.append(current_directory)
import path
import importlib
# importlib.reload(path_ipynb)
import pandas as pd
import numpy as np
from collections import Counter
import ast
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, make_scorer,r2_score
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split

import LDA

import stockdata_process
import re

def make_lasso_df(df):
    df = df[['Date','lda_topics','num_w','Target']]
    result_df = df.pivot_table(index=['Date', 'Target'], columns='lda_topics', values='num_w', aggfunc='sum').reset_index()
    result_df = result_df.fillna(0)
    return result_df

def expand_ngrams(ngram_list, up_down):
    counts = Counter()
    counts.update(ngram_list)
    # print(counts)

    ngrams_expanded = []
    up_counts = []
    down_counts = []
    for ngram, count in counts.items():
        ngrams_expanded.append(ngram)
        if up_down == 1:
            up_counts.append(count)
            down_counts.append(0)
        else:
            up_counts.append(0)
            down_counts.append(count)

    return pd.DataFrame({'ngram': ngrams_expanded, 'up': up_counts, 'down': down_counts})


# 각 행마다 n-gram을 풀어서 데이터프레임 재구성 후 결합


class predict_stock():    
    def __init__(self, stock_df, article_df, stock_name):
        self.stock_df = stock_df
        self.article_df = article_df
        self.stock_name = stock_name
    
    def get_LDA(self, num_subject, random_state, passes):
        self.num_subject = num_subject
        self.LDA_model, self.LDA_df, self.corpus = LDA.get_LDA(self.article_df, 'Content', num_subject, random_state, passes)
        print(self.LDA_model.print_topics(num_words=10))

    def get_Lasso_df(self):        
        self.all_df = stockdata_process.get_analyze_df(self.LDA_df, self.stock_df)
        self.model_df = make_lasso_df(self.all_df)

    def Lasso_run(self, num_cv_iterations, n_splits):
        # Feature(독립변수)와 Target(종속변수)를 나눕니다.
        X = self.model_df.drop(columns=['Date', 'Target'])  # Feature로 lda_topic 컬럼들을 사용
        y = self.model_df['Target']  # Target으로 'target' 컬럼을 사용

        X = np.array(X)
        y = np.array(y)

        # 초기화된 최종 가중치 리스트
        final_weights = None

        # num_cv_iterations 만큼 반복
        for _ in range(num_cv_iterations):
            # 10-겹 교차 검증을 위한 준비
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

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

                # 모델 가중치 저장
                model_weights.append(lasso_model.coef_)

            # 10-겹 교차 검증에서의 평균 가중치 계산
            average_weights = np.mean(model_weights, axis=0)

            # 최종 가중치 업데이트
            if final_weights is None:
                final_weights = average_weights
            else:
                final_weights += average_weights

        # 최종 가중치를 num_cv_iterations로 나누어 평균 계산
        final_weights /= num_cv_iterations

        # 최종 LASSO 모델 생성
        final_lasso_model = Lasso(alpha=1.0, fit_intercept=True)
        final_lasso_model.coef_ = final_weights  # 가중치 설정
        self.subject_importance = list(final_weights)
        self.Lasso_model = final_lasso_model
        print(self.subject_importance)
        print(len(self.subject_importance))        
    
    def Lasso_run_accuracy(self):
        X = self.model_df.drop(columns=['Date', 'Target'])  # Feature로 lda_topic 컬럼들을 사용
        y = self.model_df['Target']  # Target으로 'target' 컬럼을 사용

        # 데이터를 학습용과 테스트용으로 나눕니다.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Lasso 회귀 모델 생성 및 학습
        lasso_model = Lasso(alpha=1.0)  # alpha는 L1 규제 강도 조절
        lasso_model.fit(X_train, y_train)

        # 테스트 데이터로 예측
        y_pred = lasso_model.predict(X_test)

        # 모델 평가 (예시로 MSE와 R-squared 사용)
        self.mse = mean_squared_error(y_test, y_pred)
        self.r2 = r2_score(y_test, y_pred)

        print(f"Mean Squared Error (MSE): {self.mse}")
        print(f"R-squared (R2): {self.r2}")
        

        top_5_percent = self.model_df[self.model_df['Target'] >= self.model_df['Target'].quantile(0.95)]
        
        X = top_5_percent.iloc[:, 2:self.num_subject+2]
        top_5_percent['Target_predict'] = lasso_model.predict(X)
        threshold = 0.3
        top_5_percent['Success'] = top_5_percent['Target_predict'] >= (top_5_percent['Target'] * (threshold))

        # 정확도 계산
        self.accuracy = accuracy_score(top_5_percent['Success'], [True] * len(top_5_percent))
        print(self.accuracy)
    
    def article_score(self):
        topics_info = self.LDA_model.print_topics(num_words=10)
        top_words_per_topic = {}

        # topics_info를 순회하면서 딕셔너리에 단어와 가중치 추가
        for topic_id, topic_info in topics_info:
            words_and_weights = topic_info.split(' + ')
            word_weight_dict = {}
            for word_weight in words_and_weights:
                weight, word = word_weight.split('*')
                word = word.strip('"')
                weight = float(weight)
                word_weight_dict[word] = weight
            top_words_per_topic[topic_id] = word_weight_dict

        # for topic_id, word_weights in top_words_per_topic.items():
        #     for word, weight in word_weights.items():
        #         # Lasso 회귀 모델의 가중치와 주어진 가중치를 곱하여 업데이트
        #         updated_weight = weight * self.subject_importance[topic_id]  # 주제 ID와 일치
        #         word_weights[word] = updated_weight
        # # self.all_df['Score'] = self.all_df.apply(lambda row: sum(top_words_per_topic[row['lda_topics']].get(token, 0) for token in row['ngram_token']), axis=1)
        self.all_df['no_Lasso_Score'] = self.all_df.apply(lambda row: sum(top_words_per_topic.get(row['lda_topics'], {}).get(token, 0) for token in row['ngram_token']), axis=1)
        self.all_df['no_Lasso_Score'] = self.all_df['no_Lasso_Score'] - self.all_df['no_Lasso_Score'].mean()
        for topic_id, word_weights in top_words_per_topic.items():
            for word, weight in word_weights.items():
                # Lasso 회귀 모델의 가중치와 주어진 가중치를 곱하여 업데이트
                updated_weight = weight * self.subject_importance[topic_id]  # 주제 ID와 일치
                word_weights[word] = updated_weight
        # self.all_df['Score'] = self.all_df.apply(lambda row: sum(top_words_per_topic[row['lda_topics']].get(token, 0) for token in row['ngram_token']), axis=1)
        self.all_df['Score'] = self.all_df.apply(lambda row: sum(top_words_per_topic.get(row['lda_topics'], {}).get(token, 0) for token in row['ngram_token']), axis=1)

        score_df = self.all_df[['Date','Score']]
        no_lasso_score_df = self.all_df[['Date','no_Lasso_Score']]
        stock_df = self.all_df[['Date','Close']]

        score_df['Date'] = pd.to_datetime(score_df['Date'])
        no_lasso_score_df['Date'] = pd.to_datetime(no_lasso_score_df['Date'])
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])

        # 월별 기사 점수 합산
        score_df['YearMonth'] = score_df['Date'].dt.strftime('%Y-%m')
        article_score_monthly = score_df.groupby('YearMonth')['Score'].sum().reset_index()

        no_lasso_score_df['YearMonth'] = no_lasso_score_df['Date'].dt.strftime('%Y-%m')
        article_score_monthly_no_lasso = no_lasso_score_df.groupby('YearMonth')['no_Lasso_Score'].sum().reset_index()

        # 월별 주식 상승률 계산
        stock_df['YearMonth'] = stock_df['Date'].dt.strftime('%Y-%m')
        monthly_stock_change = (stock_df.groupby('YearMonth')['Close'].last() / stock_df.groupby('YearMonth')['Close'].last().shift(-1) - 1) * 100# 상승률 계산
        article_score_monthly['stock_change'] = monthly_stock_change.reset_index(drop=True)
        article_score_monthly_no_lasso['stock_change'] = monthly_stock_change.reset_index(drop=True)
        self.stock_comparison_df = article_score_monthly
        self.stock_comparison_no_lasso_df = article_score_monthly_no_lasso
    # def korea_bank_implement(self):
    #     self.korea_df = self.all_df[['Date','Close','ngram_token','1month_updown']]
    #     # korea_df['ngram_token'] = korea_df['ngram_token'].apply(ast.literal_eval)
    #     result = pd.concat([expand_ngrams(row['ngram_token'], row['1month_updown']) for _, row in self.korea_df.iterrows()])
    #     result = result.groupby('ngram').sum().reset_index()
    #     # 없는 단어가 있을경우 무한대가 될 수 있으므로 1을 더해준다.
    #     result['up'] += 1
    #     result['down'] += 1

    #     result['p_score'] = (result['up']/result['up'].sum())/(result['down']/result['down'].sum())

    #     result['sentiment'] = result['p_score'].apply(lambda x: '긍정' if x >= 1.3 else ('부정' if x <= 1/1.3 else '중립'))
    #     positive = set(list(result[result['sentiment'] == '긍정']['ngram']))
    #     negative = set(list(result[result['sentiment'] == '부정']['ngram']))
    #     monthly_tokens = self.korea_df.groupby(self.korea_df['Date'].dt.strftime('%Y-%m'))['ngram_token'].sum()
        
    #     def calculate_sentiment_score(tokens):
    #         scores = []
    #         for sentence_ngrams in tokens:
    #             # 문장 내에서 긍정 단어와 부정 단어의 출현 빈도 계산
    #             positive_count = sum(1 for ngram in sentence_ngrams if ngram in positive)
    #             negative_count = sum(1 for ngram in sentence_ngrams if ngram in negative)
                
    #             # 감정 점수 계산
    #             if (positive_count + negative_count) != 0:
    #                 sentiment_score = (positive_count - negative_count) / (positive_count + negative_count)
    #             else:
    #                 sentiment_score = 0  # 분모가 0이면 0으로 처리
                
    #             scores.append(sentiment_score)
            
    #         return scores
    #     monthly_sentiment_scores = monthly_tokens.apply(calculate_sentiment_score)
    #     print(monthly_sentiment_scores)

    def korea_bank_implement(self):
        
        korea_df = self.all_df[['Date','Close','ngram_token','1month_updown']]
        # korea_df['ngram_token'] = korea_df['ngram_token'].apply(ast.literal_eval)
        
        result = pd.concat([expand_ngrams(row['ngram_token'], row['1month_updown']) for _, row in korea_df.iterrows()])
        # print(result)
        result = result.groupby('ngram').sum().reset_index()
        # print(result)
        # 없는 단어가 있을경우 무한대가 될 수 있으므로 1을 더해준다.
        result['up'] += 1
        result['down'] += 1        
        result['p_score'] = (result['up']/result['up'].sum())/(result['down']/result['down'].sum())
        font_path = 'C:/Windows/Fonts/malgun.ttf'
        result['sentiment'] = result['p_score'].apply(lambda x: '긍정' if x >= 1.3 else ('부정' if x <= 1/1.3 else '중립'))
        self.dictionary = result[['ngram','sentiment','up','down']]
        positive_result = result[result['sentiment'] == '긍정']
        negative_result = result[result['sentiment'] == '부정']
        self.positive_wordcloud = WordCloud(font_path = font_path, width=800, height=400, background_color='white').generate_from_frequencies(positive_result.set_index('ngram')['up'])
        self.negative_wordcloud = WordCloud(font_path = font_path, width=800, height=400, background_color='white').generate_from_frequencies(negative_result.set_index('ngram')['down'])
        self.positive_word = set(list(result[result['sentiment'] == '긍정']['ngram']))
        # print(positive)
        self.negative_word = set(list(result[result['sentiment'] == '부정']['ngram']))
        # print(negative)
        monthly_tokens = korea_df.groupby(korea_df['Date'].dt.strftime('%Y-%m'))['ngram_token'].sum()
        monthly_df = monthly_tokens.reset_index()
        monthly_df.columns = ['YearMonth', 'ngram_token']
        # print(monthly_tokens)
        
        def calculate_sentiment_score(ngram_token):
            positive_count = 0
            negative_count = 0
            for token in ngram_token:
                if token in self.positive_word:
                    positive_count += 1
                if token in self.negative_word:
                    negative_count += 1
            return (positive_count - negative_count) / (positive_count + negative_count)
            
        monthly_df['score'] = monthly_df['ngram_token'].apply(calculate_sentiment_score)
        stock_df = self.all_df[['Date','Close']]
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        stock_df['YearMonth'] = stock_df['Date'].dt.strftime('%Y-%m')
        monthly_stock_change = (stock_df.groupby('YearMonth')['Close'].last() / stock_df.groupby('YearMonth')['Close'].last().shift(-1) - 1) * 100# 상승률 계산
        monthly_df['stock_change'] = monthly_stock_change.reset_index(drop=True)

        # monthly_df['stock_change'] = self.stock_comparison_df['stock_change']
        self.korea_bank_df = monthly_df[['YearMonth','score','stock_change']]

    def get_FVE(self): #Fraction of Volume Explained
        self_top_5_percent = self.all_df[self.all_df['Target'] >= self.all_df['Target'].quantile(0.95)]
        topic_counts = self_top_5_percent['lda_topics'].value_counts()
        total_count = len(self_top_5_percent)
        topic_ratios = topic_counts / total_count

        topic_counts_all = self.all_df['lda_topics'].value_counts()
        total_count_all = len(self.all_df)
        topic_ratios_all = topic_counts_all / total_count_all 

        def get_top_words(lda_model):
            topic_words = {}
            for topic_id in lda_model.print_topics(num_words=6, num_topics = 30):
                
                topic_num = topic_id[0]
                topic_word = topic_id[1]
                result = re.sub(r'[^가-힣a-zA-Z\s]+', '', topic_word)
                topic_words[topic_num] = result
                
            return topic_words

        # 각 주제별 상위 5개 단어 추출
        lda_topics = get_top_words(self.LDA_model)  # lda_model은 적절한 모델을 로드해야 합니다.

        # 'lda_topic' 번호를 기반으로 각 주제의 상위 5개 단어를 매핑하여 'topic_word' 칼럼 생성

        # 데이터프레임으로 변환
        result_df = pd.DataFrame({'lda_topic': topic_ratios.index, 'FVE_rate_Peak': topic_ratios.values})
        result_df_all = pd.DataFrame({'lda_topic': topic_ratios_all.index, 'FVE_rate_All': topic_ratios_all.values})
        result_df

        # a = pd.DataFrame(lda_topics)
        a = list(lda_topics.keys())
        b = list(lda_topics.values())
        data = {'lda_topic': a,
                'topic_word': b}
        
        topic_word = pd.DataFrame(data)
        FVE_dataframe_ = pd.merge(result_df, topic_word, on = 'lda_topic', how = 'left')
        FVE_dataframe = pd.merge(FVE_dataframe_, result_df_all, on = 'lda_topic', how = 'left')
        FVE_dataframe['difference'] = FVE_dataframe['FVE_rate_Peak'] / FVE_dataframe['FVE_rate_All']
        self.FVE_dataframe = FVE_dataframe.sort_values(by='difference', ascending=False)
        self.FVE_dataframe['FVE'] = self.FVE_dataframe['difference'] / self.FVE_dataframe['difference'].sum()
        self.FVE = self.FVE_dataframe[['topic_word','FVE']]

