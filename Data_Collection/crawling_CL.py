##검색어 검색시 뉴스 크롤링 코드
import selenium
from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import bs4
from openpyxl.workbook import Workbook
import re
import os
from tqdm import tqdm
import time
from fake_useragent import UserAgent


def crawaling_get_driver(url, is_headless):
    options = webdriver.ChromeOptions()
    if is_headless:
        options.add_argument('headless')  # 창이 없이 크롬이 실행이 되도록 만든다
    options.add_argument("--start-maximized")  # 창이 최대화 되도록 열리게 한다.
    options.add_argument("disable-infobars")  # 안내바가 없이 열리게 한다.
    options.add_argument('--disable-dev-shm-usage')  # 공유메모리를 사용하지 않는다
    options.add_argument("disable-gpu")  # 크롤링 실행시 GPU를 사용하지 않게 한다.
    options.add_argument("--disable-extensions")  # 확장팩을 사용하지 않는다.
    driver = webdriver.Chrome(service=Service(), options=options)
    # 싸이트에 접근하기 위해 get메소드에 이동할 URL을 입력한다.
    driver.get(url)
    return driver

stop = []
def crawling_get_news(keyword, startdate, enddate, company_name):
    news_data = []

    for dt in pd.date_range(start=startdate, end=enddate):

        data = {}
        page = 1
        tmpdt = dt.strftime("%Y.%m.%d")
        print('기사 날짜 : ', tmpdt)
        print('기업 이름 : ', company_name)

        while True:
            driver = crawaling_get_driver('https://www.naver.com/', True)
            time.sleep(0.3)
            start = (10 * (page - 1)) + 1
            time.sleep(0.3)
            URL = "https://search.naver.com/search.naver?where=news&sm=tab_pge&query={0}&sort=0&photo=0&field=0&pd=3&ds={1}&de={2}&cluster_rank=11&mynews=1&office_type=1&office_section_code=3&news_office_checked=1018&nso=so:r,p:from{3}to{4},a:all&start={5}".format(
                keyword, tmpdt, tmpdt, tmpdt.replace(".", ""), tmpdt.replace(".", ""), start)
            driver.get(URL)
            time.sleep(0.3)
            bsObj = bs4.BeautifulSoup(driver.page_source, "html.parser")
            ua = UserAgent(use_cache_server=True)
            h = {
                'User-Agent': ua.random}
            #'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'
            # response = requests.get(URL, headers = h)
            # bsObj = BeautifulSoup(response.text, 'html.parser')
            try:
                hrefs = [a['href'] for a in bsObj.find_all('a', {'class': 'info'})]
            except:
                pass
            try:
                bsObj.select("div.not_found02")[0].text
                break
            except IndexError:
                pass
            # print(hrefs)
            hrefs = hrefs[1::2]
            # print(hrefs)
            if "비정상적인 검색" in bsObj.text:
                print('비정상 기사 날짜 : ', tmpdt)
                stop.append(tmpdt)
                return pd.DataFrame(news_data)
            
            for href in tqdm(hrefs):

                # print('기사 URL :', href)

                response = requests.get(href, headers=h)

                soup = BeautifulSoup(response.text, 'html.parser')

                # 정규표현식을 이용한 전처리
                try:
                    news_content = soup.find('article', {'id': 'dic_area'}).text
                    pattern_list = [
                        r'\w+\s+특파원\s+=\s*',  # 'XXX 특파원 = '
                        r'\[\w+\s+\w+\s+\w+\s+\]',  # '[AFP 연합뉴스 자료사진]'
                        r'\.\w+',  # '.kr'
                        r'\[.*?\]',  # '[이데일리 XXX 기자]'
                        r'\(.*?\)',  # '(사진=XXXX XXXX)'
                        r'.*?=',  # '윤영숙 특파원 ='
                        r'\w+@\w+\.\w+',  # 'ysyoon@yna.co.kr'
                        r'\(끝\)',  # '(끝)'
                        r'■',
                        r'△'
                        ]
                    for pattern in pattern_list:
                        news_content = re.sub(pattern, '', news_content)
                    news_content = news_content.replace('\n', '').replace('\r','').replace('<br>', '').replace('\t', '')
                except:
                    continue
                # print(news_content)
                # news_content = news_content.replace(, '')
                news_title = soup.find('div', {'class' : 'media_end_head_title'}).text
                news_date = soup.find('span', {'class': 'media_end_head_info_datestamp_time _ARTICLE_DATE_TIME'}).text[
                            :10]
                news_title = news_title.replace('\n', '').replace('\r', '').replace('<br>', '').replace('\t', '')
                news_data.append({'Date': news_date, 'Company':company_name, 'Article_address' : href,'news_title' : news_title, 'Content': news_content, })
            try:
                bsObj.select("div.not_found02")[0].text
                break
            except IndexError:
                pass
            driver.close()
            page = page + 1

    df = pd.DataFrame(news_data)

    return df

def collect_news_for_multiple_keywords(keywords, startdate, enddate):
    all_news_data = []
    for keyword, company_name in keywords.items():
        news_data = crawling_get_news(keyword, startdate, enddate, company_name)
        all_news_data.append(news_data)
        # 크롤링이 완료될 때마다 데이터를 XLSX 파일로 저장
        company_folder_path = os.getcwd()
        company_name_excel_file_name = '크롤링 중간 저장_{}_{}_{}.xlsx'.format(keyword, startdate, enddate)
        news_data.to_excel(company_name_excel_file_name)
        print('크롤링이 완료되었습니다. 데이터를 저장했습니다. | 경로 : {}\\{}'.format(company_folder_path, company_name_excel_file_name))
    combined_news_data = pd.concat(all_news_data, ignore_index=True)
    return combined_news_data

# 여러 기업 키워드 리스트
company_keywords = {
    '삼성바이오로직스': '삼성바이오로직스',
    '셀트리온':'셀트리온',
    'SK바이오사이언스':'SK바이오사이언스',
    '셀트리온헬스케어':'셀트리온헬스케어',
    'SK바이오팜':'SK바이오팜',
    'HLB':'HLB',
    '한미약품':'한미약품',
    '씨젠':'씨젠',
    '알테오젠':'알테오젠',
    'SK케미칼':'SK케미칼',
    '신풍제약':'신풍제약',
    '메지온':'에지온',
    '바이오니아':'바이오니아'
}
startdate = "20220101"
enddate = "20221231"
combined_news = collect_news_for_multiple_keywords(company_keywords, startdate, enddate)
folder_path = os.getcwd()
xlsx_file_name = '기업 통합_{}_{}.xlsx'.format(startdate, enddate)
combined_news.to_excel(xlsx_file_name, index=False)
print('엑셀 저장 완료 | 경로 : {}\\{}\n'.format(folder_path, xlsx_file_name))
print('===============크롤링이 완료되었습니다!!!!박수!!!!!==================================')