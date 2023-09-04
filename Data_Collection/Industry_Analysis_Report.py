import requests
from bs4 import BeautifulSoup
import os
import fitz
import re
import pandas as pd
from os import listdir
from os.path import isfile, join

def industry_analysis_report_crawling(Sectors='자동차'):
    url_base = 'https://finance.naver.com/research/industry_list.naver?keyword=&brokerCode=&writeFromDate=&writeToDate=&searchType=upjong&upjong=%C0%DA%B5%BF%C2%F7&x=38&y=17&page='
    download_directory = f'../data/temp_data/{Sectors}_pdf'  # PDF 파일을 저장할 디렉토리

    if not os.path.exists(download_directory):
        os.makedirs(download_directory)

    max_pages = 100  # 최대 페이지 수 설정
    securities_per_page = 30  # 한 페이지당 다운로드할 리포트 수 설정

    for page in range(1, max_pages + 1):  # 1부터 최대 페이지까지 반복
        url = url_base + str(page)
        response = requests.get(url)
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')

        tr_tags = soup.find_all('tr')  # 모든 <tr> 태그를 찾아냄
        securities_name = []

        count = 0  # Counter to keep track of extracted values
        for tr in tr_tags:
            td_elements = tr.find_all('td')

            # Check if the list of td_elements has at least 3 elements (for your desired third <td>)
            if len(td_elements) >= 3:
                third_td = td_elements[2]
                securities_name.append(third_td.text)
                count += 1  # Increment the counter
                if count >= securities_per_page:
                    break  # Exit the loop when desired securities_per_page values are extracted
            else:
                pass

        date_td_elements = soup.find_all('td', class_='date')
        date = [element.get_text() for index, element in enumerate(date_td_elements) if index % 2 == 0]

        file_td_elements = soup.find_all('td', class_='file')
        for i, td in enumerate(file_td_elements):
            a_tag = td.find('a')
            if a_tag and a_tag.get('href') and a_tag.get('href').endswith('.pdf'):
                pdf_url = a_tag.get('href')
                pdf_name = pdf_url.split('/')[-1]

                # PDF 파일 이름 생성 (securities_name_date_pdf_name)
                pdf_name_formatted = f"{Sectors}_{date[i]}_{securities_name[i]}_{pdf_name}"

                pdf_path = os.path.join(download_directory, pdf_name_formatted)

                pdf_response = requests.get(pdf_url)
                with open(pdf_path, 'wb') as pdf_file:
                    pdf_file.write(pdf_response.content)

                print(f"Downloaded: {pdf_name_formatted}, {i+1} / {securities_per_page} from page {page}")

def pdf2txt(Sectors='자동차'):
    pdf_folder = f'../data/temp_data/{Sectors}_pdf'  # PDF 파일을 저장한 디렉토리
    txt_folder = f'../data/temp_data/{Sectors}_pdf_to_txt'  # 텍스트 파일을 저장할 디렉토리

    if not os.path.exists(txt_folder):
        os.makedirs(txt_folder)

    pdf_files = [file for file in os.listdir(pdf_folder) if file.endswith(".pdf") and file.startswith(Sectors)]

    n = 1
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        txt_file = pdf_file.replace(".pdf", ".txt")
        txt_path = os.path.join(txt_folder, txt_file)

        try:
            pdf_document = fitz.open(pdf_path)
            text = ""

            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                text += page.get_text("text")

            # Modify the pattern according to your needs
            pattern = r"- \d+ -"
            text = re.sub(pattern, "", text)

            with open(txt_path, "w", encoding="utf-8") as text_file:
                text_file.write(text)

            print(f"{pdf_file} 변환 및 저장 완료", n, '/', len(pdf_files))
            n += 1

        except Exception as e:
            print(f"오류 발생: {pdf_file} - {e}")

        finally:
            if 'pdf_document' in locals():
                pdf_document.close()

def txt2csv(Sectors='자동차'):
    txt_folder = f'../data/temp_data/{Sectors}_pdf_to_txt'  # 텍스트 파일을 저장한 디렉토리
    output_file = f'../data/{Sectors}_combined_report.csv'  # 결과 CSV 파일

    # 데이터프레임 생성
    columns = ['Date', 'Securities_company', 'Report']
    df = pd.DataFrame(columns=columns)

    # 텍스트 파일 목록 가져오기
    txt_files = [f for f in listdir(txt_folder) if isfile(join(txt_folder, f))]
    txt_files.sort()

    # 각 텍스트 파일 처리
    rows_to_concat = []
    n = 1
    for txt_file in txt_files:
        # 파일 이름에서 정보 추출
        date = txt_file.split('_')[1]
        securities_company = txt_file.split('_')[2]

        # 텍스트 파일 읽기
        with open(os.path.join(txt_folder, txt_file), 'r', encoding='utf-8', errors='ignore') as f:
            txt = f.read()

        # 행 정보 저장
        row = {'Date': date, 'Securities_company': securities_company, 'Report': txt}
        rows_to_concat.append(row)

        # 진행률 출력
        print(n, '/', len(txt_files))
        n += 1

    # 데이터프레임에 행 추가
    df = pd.concat([df, pd.DataFrame(rows_to_concat)], ignore_index=True)

    # 결과를 CSV 파일로 저장
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    df.info()

# 업종을 지정하여 함수 호출
industry_analysis_report_crawling('자동차')
pdf2txt('자동차')
txt2csv('자동차')
