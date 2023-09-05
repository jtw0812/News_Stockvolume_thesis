import sys,os
current_directory = os.getcwd()
sys.path.append(current_directory)
import path

from ekonlpy.tag import Mecab
from ekonlpy.sentiment import MPKO
mpko = MPKO()

def get_tokens_ngrams(text): #내용집어넣으면 바로 ngrams화 해주는 함수 만들기
    tokens = mpko.tokenize(text)
    return tokens

def column_ngramize(df, column):
    df['n_grams'] = df[column].apply(lambda text: get_tokens_ngrams(text))
    return df


