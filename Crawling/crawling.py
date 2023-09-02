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


def crawling_get_driver(url, is_headless):
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

stop =[]
def crawling_get_news(keyword, startdate, enddate):
    news_data = []

    for dt in pd.date_range(start=startdate, end=enddate):

        data = {}
        page = 1
        tmpdt = dt.strftime("%Y.%m.%d")
        print('기사 날짜 : ', tmpdt)

        while True:
            driver = crawling_get_driver('https://www.naver.com/', True)
            time.sleep(0.3)
            start = (10 * (page - 1)) + 1
            time.sleep(0.3)
            URL = "https://search.naver.com/search.naver?where=news&sm=tab_pge&query={0}&sort=0&photo=0&field=0&pd=3&ds={1}&de={2}&cluster_rank=11&mynews=1&office_type=1&office_section_code=3&news_office_checked=1018&nso=so:r,p:from{3}to{4},a:all&start={5}".format(
                keyword, tmpdt, tmpdt, tmpdt.replace(".", ""), tmpdt.replace(".", ""), start)
            driver.get(URL)
            time.sleep(0.3)
            bsObj = bs4.BeautifulSoup(driver.page_source, "html.parser")
            h = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'}
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
                        r'\[.*?\]',  # '[이데일리 XXX 기자]'
                        r'\(.*?\)',  # '(사진=XXXX XXXX)'
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
                news_date = soup.find('span', {'class': 'media_end_head_info_datestamp_time _ARTICLE_DATE_TIME'}).text[
                            :10]

                news_data.append({'Date': news_date, 'Article_address' : href,'Content': news_content, })

            try:
                bsObj.select("div.not_found02")[0].text
                break
            except IndexError:
                pass
            driver.close()
            page = page + 1

    df = pd.DataFrame(news_data)

    return df




