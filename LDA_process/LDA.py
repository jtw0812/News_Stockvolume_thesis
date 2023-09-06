import sys,os
current_directory = os.getcwd()
sys.path.append(current_directory)
import path
import importlib

import eKoNL as ek



import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

# def get_LDA(document, column, num_topics, random_state, passes):
#     token = list(ek.column_ngramize(document,column)['n_grams']) #텍스트 전처리
#     dictionary = corpora.Dictionary(token)
#     corpus = [dictionary.doc2bow(doc) for doc in token]
#     lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics = num_topics, random_state = random_state, passes = passes)
#     doc_topics = [lda_model[doc] for doc in corpus]
#     document['lda_topics'] = doc_topics
#     return lda_model, document, corpus

def get_LDA(document, column, num_topics, random_state, passes):
    document['ngram_token'] = ek.column_ngramize(document, column)['n_grams'] + ek.column_tokenize(document,column)['tokens']
    token = list(document['ngram_token'])  # 텍스트 전처리
    dictionary = corpora.Dictionary(token)
    num_w = [len(doc) for doc in token]
    document['num_w'] = num_w
    corpus = [dictionary.doc2bow(doc) for doc in token]
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=random_state, passes=passes)      
    max_prob_topics = [max(doc, key=lambda x: x[1])[0] for doc in lda_model[corpus]] # 각 문서에 대해 가장 확률이 높은 주제 선택        
    document['lda_topics'] = max_prob_topics # 선택한 주제를 문서에 할당
    
    return lda_model, document, corpus

