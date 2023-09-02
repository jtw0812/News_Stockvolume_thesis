import sys, os

# 현재 스크립트가 위치한 디렉토리의 경로를 계산
current_directory = os.getcwd()

# eKoNL.py 파일이 위치한 디렉토리와 Execution 디렉토리의 경로를 계산
ekonl_directory = os.path.join(current_directory, 'text_preprocess')
Crawling_directory = os.path.join(current_directory, 'Crawling')
execution_directory = current_directory

# 해당 디렉토리들을 모듈 검색 경로에 추가
sys.path.append(ekonl_directory)
sys.path.append(execution_directory)
sys.path.append(Crawling_directory)
import os
import crawling

keywords = '현대차'
startdate = "20180101"
enddate = "20180131"

news = crawling.crawling_get_news(keywords, startdate, enddate)

folder_path = os.path.join(execution_directory, "data")

xlsx_file_name = '{}_{}_{}.xlsx'.format(keywords, startdate, enddate)

news.to_excel(xlsx_file_name, index=False)

print('엑셀 저장 완료 | 경로 : {}\\{}\n'.format(folder_path, xlsx_file_name))
