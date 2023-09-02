
import sys, os

current_script_path = os.getcwd() # 현재 스크립트 파일의 경로를 가져오기
current_directory = os.path.dirname(current_script_path) # 현재 스크립트 파일의 디렉토리 경로 (상위 디렉토리)를 계산
ekonl_directory = os.path.join(current_script_path, 'text_preprocess') # eKoNL.py 파일이 위치한 디렉토리와 Execution 디렉토리의 경로를 계산
# execution_directory = current_directory
# 해당 디렉토리들을 모듈 검색 경로에 추가
print(current_script_path)
print(current_directory)
print(ekonl_directory)
sys.path.append(ekonl_directory)
sys.path.append(current_directory)
import eKoNL as ek
import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

def get_LDA(document, column, num_topics, random_state, passes):
    token = list(ek.column_ngramize(document,column)['n_grams']) #텍스트 전처리
    dictionary = corpora.Dictionary(token)
    corpus = [dictionary.doc2bow(doc) for doc in token]
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics = num_topics, random_state = random_state, passes = passes)
    return lda_model, corpus



