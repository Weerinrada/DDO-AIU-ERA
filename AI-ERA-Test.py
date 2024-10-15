import streamlit as st
import re
import uuid
import concurrent.futures
import boto3
from langchain_community.chat_models import BedrockChat
from langchain_core.prompts import ChatPromptTemplate
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import yfinance as yf
from botocore.config import Config
import time
from PIL import Image
from googlesearch import search
from fuzzywuzzy import fuzz
from io import StringIO

# Constants and configurations
REGION_NAME = "ap-southeast-1"
MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"


custom_config = Config(
    read_timeout=400,
    connect_timeout=400,
    retries={"max_attempts": 3},
)

# Initialize session state
if "company_name" not in st.session_state:
    st.session_state.company_name = ""
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "input_key" not in st.session_state:
    st.session_state.input_key = str(uuid.uuid4())


def initialize_bedrock_client(
    aws_access_key_id, aws_secret_access_key, aws_session_token
):
    return boto3.client(
        "bedrock-runtime",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        region_name=REGION_NAME,
        config=custom_config,
    )


def search_news(query):
    url = f"https://www.googleapis.com/customsearch/v1"
    params = {
        "key": os.environ.get("api_key"),
        "cx": os.environ.get("cse_id"),
        "q": query,
        "sort": "date",
        "dateRestrict": "m[18]",
        "start": 1,
    }
    response = requests.get(url, params=params, timeout=30)
    results = response.json()
    return results


def clean_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
    text = soup.get_text(separator=" ")
    clean_text = " ".join(text.split())
    return clean_text


def get_juristic_id_news(company_name, llm):
    start_search_juris_id = time.time()
    juris_query = search(f"เลขนิติบุคคล {company_name}", num_results=3, advanced=True)
    juris_id = {}
    for i, result in enumerate(juris_query):
        result_data = {
            "title": result.title,
            "url": result.url,
            "description": result.description,
        }
        juris_id[f"result_{i}"] = result_data
    print(
        f"\n Running time process Search for Juristic ID: {time.time() - start_search_juris_id}"
    )

    prompt = f"""เลขทะเบียนนิติบุคคล of {company_name} using information from {juris_id}?
    Please provide only the เลขทะเบียน หรือ เลขนิติบุคคลของ or juristic id without any additional text. """
    comp_id = llm.invoke(prompt, temperature=0.0, top_p=0.95)
    juristic_id = (
        comp_id.content.strip() if hasattr(comp_id, "content") else str(comp_id).strip()
    )
    print(f"Juristic_id: {juristic_id}")
    url_juristic_id = f"https://data.creden.co/company/general/{juristic_id}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36"
    }
    response_juris_id = requests.get(url_juristic_id, headers=headers)
    comp_data = clean_html(response_juris_id.text)

    prompt_comp_name = f"""What is the full company name in Thai language from {comp_data}? 
    Please provide only the company name without any additional text.
    without The full company name in Thai language is: """
    response_comp_name = llm.invoke(prompt_comp_name, temperature=0.0, top_p=1)
    comp_name = (
        response_comp_name.content.strip()
        if hasattr(response_comp_name, "content")
        else str(response_comp_name).strip()
    )
    print(f"Full company's Name: {comp_name}")
    # If not found, ask Claude for help
    prompt_symbol = f"""What is the stock symbol or ชื่อย่อในตลาดหลักทรัพย์ for the {comp_name} using {comp_data}?
    Please provide only the symbol without any additional text.
    """

    response_symbol = llm.invoke(prompt_symbol, temperature=0.0, top_p=1)
    if hasattr(response_symbol, "content"):
        symbol_ai = response_symbol.content.strip()
    else:
        symbol_ai = str(response_symbol).strip()

    print(f"Symbol AI: {symbol_ai}")

    url = "https://www.set.or.th/dat/eod/listedcompany/static/listedCompanies_th_TH.xls"
    response = requests.get(url)
    print("Response: ", response)
    if response.status_code == 200:
        # dfs = pd.read_html(response.text)
        dfs = pd.read_html(StringIO(response.text))
        print("Data: ", dfs)
        df = dfs[0]
        df.columns = df.iloc[1]
        df = df.iloc[2:].reset_index(drop=True)
        if (
            "I apologize" in symbol_ai
            or "I do not have any information" in symbol_ai
            or "there is no stock symbol" in symbol_ai
        ):
            print("No stock symbol found for the given company.")
            result = df[
                # df["บริษัท"].str.contains(company_name, case=False, na=False, regex=False)
                (df["บริษัท"].apply(lambda x: fuzzy_match(x, comp_name)))
            ]
            if result.empty:
                print(f"No matching company found for '{comp_name}'")
            else:
                print(f"Found company information without stock symbol:")
            print(result)
        else:
            result = df[
                (df["บริษัท"].apply(lambda x: fuzzy_match(x, comp_name)))
                & (df["หลักทรัพย์"] == symbol_ai)
            ]
            if result.empty:
                print(
                    f"No matching company found for '{comp_name}' with symbol '{symbol_ai}'"
                )
            else:
                print(f"Found company information:")
            print(result)

    if not result.empty:
        symbol = result.iloc[0]["หลักทรัพย์"]
        symbol_with_bk = f"{symbol}.BK"
    else:
        symbol_with_bk = None

    start_search = time.time()
    result_query = search_news(f"ข่าวเกี่ยวกับ {comp_name}")
    company_news = [
        {
            "title": item["title"],
            "url": item["link"],
            "snippet": item.get("snippet", "No snippet available"),
        }
        for item in result_query.get("items", [])
    ]
    print(f"\n Running time process Search for News: {time.time() - start_search}")

    # else:
    #     symbol_with_bk = None
    return juristic_id, symbol_with_bk, company_news, juris_id


def get_financial_data(juristic_id, symbol=None):
    fin_data = {}
    data = None
    url_fin = None
    comp_id = juristic_id

    if symbol and symbol.endswith(".BK"):
        new_symbol = symbol.split(".")[0]
        url_fin = [
            {
                "title": "ข้อมูลบริษัทจาก Creden data",
                "url": f"https://data.creden.co/company/general/{comp_id}",
            },
            {
                "title": "ข้อมูลบริษัทจาก SET",
                "url": f"https://www.set.or.th/th/market/product/stock/quote/{new_symbol}/financial-statement/company-highlights",
            },
        ]
    else:
        url_fin = [
            {
                "title": "ข้อมูลบริษัทจาก Creden data",
                "url": f"https://data.creden.co/company/general/{comp_id}",
            }
        ]

    if symbol:
        stock = yf.Ticker(symbol)
        asset_profile = stock.info
        bal_sheet = stock.balance_sheet
        income_statement = stock.financials

        fin_data = {
            "assetProfile": asset_profile,
            "balanceSheet": bal_sheet,
            "incomeStatementHistory": income_statement,
        }
    else:
        fin_data = {
            "assetProfile": {},
            "balanceSheet": pd.DataFrame(),
            "incomeStatementHistory": pd.DataFrame(),
        }

    url = f"https://data.creden.co/company/general/{comp_id}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    data = clean_html(response.text)

    return fin_data, data, url_fin


def format_analysis(analysis):
    return [line.strip() for line in analysis.split("\n") if line.strip()]


def get_comp_info(llm, company_name, fin_data, data, company_news, company_officers):
    comp_profile = fin_data["assetProfile"]
    system_template = """You are specialized in financial analysis and credit analysis for auto loans. Your task is to analyze financial data and provide insights."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template_company_detail = """Analyze the following data of the company {company_name} using {data} or {comp_profile} and give the answer in Thai language:

    Company information:
    {comp_profile} and {data} 

    and
    {company_news}

    Please provide a comprehensive analysis of the company's data, including:

    Company Overview: Summarize the company overview, including the full name. If the company name doesn't match or there's no information in the stock market, indicate that it's not a listed company and use information from {data} and {company_news}. If the company name matches or has information in the stock market, indicate that it's a listed company, using information from {comp_profile} and {company_news}. If there's no juristic ID, use information from {company_news} 
        - Registered capital
        - Registration date
        - Company status, e.g., still operating or dissolved
        - Changes in company status (must be provided, if not available, explain)
        - Juristic ID of the company
        - Business size (S/M/L)
        - Business group
        - Type of juristic or company type
        - Company address or location (include postal code if available)
        - Phone number (must be provided)
        - Website (must be provided, if not available, explain)
        - display all list company_officers names from {company_officers} (must be provided in English language only, if no information, explain), request answer in English language
    do not show any financial data.
    Please structure your answer in clear paragraphs, use short sentences for easy reading, and use headings or bullet points for sub-topics as appropriate."""

    human_message_prompt_company_detail = HumanMessagePromptTemplate.from_template(
        human_template_company_detail
    )
    chat_prompt_comp_detail = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt_company_detail]
    )
    messages_comp_detail = chat_prompt_comp_detail.format_prompt(
        company_name=company_name,
        data=data,
        comp_profile=comp_profile,
        company_news=company_news,
        company_officers=company_officers,
    ).to_messages()

    response = llm.invoke(
        messages_comp_detail, temperature=0.0, max_tokens=4096, top_p=0.99, top_k=250
    )
    return response.content if hasattr(response, "content") else str(response)


def get_comp_fin(llm, company_name, fin_data, data, company_news):
    system_template = """You are specialized in financial analysis and credit analysis for auto loans. Your task is to analyze financial data and provide insights."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template_company_fin = """Analyze the following data of the company {company_name} and give the answer in Thai language:

    information:
    {company_news} and {data}
    
1. Analysis of financial statements and the company's financial situation. If it's a listed company, analyze from {fin_data}. If not a listed company in the stock market DO NOT Show any Data and explain why:
    - Income statement, analyzing financial statements (สินทรัพย์รวม (Total Assets), Equity, Total Revenue, Cost Of Revenue, Net Profit or (Net Income From Continuing Operation Net Minority Interest)(กำไรสุทธิ), Profit (Loss) from Other Activities (กำไร (ขาดทุน) จากกิจกรรมอื่น) (with symbol +/-), Total Debt, Net Dept, for the past 3 years, (Must be provided data in table format only)
    - Shareholders' equity or Stockholders Equity
    - Show liquidity indicators such as Return of Asset (ROA in %, yoy), Return of Equity (ROE in %, yoy), D/E Ratio (Debt to Equity Ratio), current ratio, and quick ratio
    - Summarize the company's financial liquidity
    - If some information is missing, explain why and how it might affect the analysis

2. Trend Analysis:
    - Forecast trends of key financial indicators for the next 6 months

3. Analysis of related news about {company_name}, {company_news}:
    - If news information is available: Summarize key points from news related to the company
    - If no news information is available: State that no recent news was found and explain the potential impact of this lack of information on the analysis
    - Analyze the impact of news (if any) on the company's financial status and operations, and potential impacts on company employees

4. Considerations for Leasing Credit Approval based on {data}, {company_news}, {fin_data}:
    - Analyze the suitability of approving leasing credit for employees of this company for car leasing product
    - Consider risk factors and positive factors that may affect employees' ability to repay debt

5. Brief Summary and Recommendations:
    - Summarize the company's current financial situation and provide advice regarding credit consideration for employees of this company

    โปรดวิเคราะห์อย่างละเอียดเพื่อนำผลการวิเคราะห์ไปประกอบการตัดสินใจพิจารณาความเสี่ยงทางด้านการเงินของบริษัทต่อไป

    Please structure your answer in clear paragraphs, use short sentences for easy reading, and use headings or bullet points for sub-topics as appropriate."""

    human_message_prompt_company_fin = HumanMessagePromptTemplate.from_template(
        human_template_company_fin
    )
    chat_prompt_comp_info = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt_company_fin]
    )
    messages_comp_info = chat_prompt_comp_info.format_prompt(
        company_name=company_name,
        data=data,
        fin_data=fin_data,
        company_news=company_news,
    ).to_messages()

    response = llm.invoke(
        messages_comp_info, temperature=0.0, max_tokens=4096, top_p=0.9999
    )
    return response.content if hasattr(response, "content") else str(response)


def run_analysis_in_parallel(
    llm, company_name, data, fin_data, company_news, company_officers
):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        comp_info = executor.submit(
            get_comp_info,
            llm,
            company_name,
            fin_data,
            data,
            company_news,
            company_officers,
        )
        comp_fin = executor.submit(
            get_comp_fin, llm, company_name, fin_data, data, company_news
        )

    comp_info = comp_info.result()
    comp_fin = comp_fin.result()
    comp_info = format_analysis(str(comp_info))
    comp_fin = format_analysis(str(comp_fin))

    return comp_info, comp_fin


def fuzzy_match(x, keyword, threshold=80):
    return fuzz.partial_ratio(x.lower(), keyword.lower()) >= threshold


def extract_table_data(data):
    table_data = []
    table_started = False
    header = None
    for line in data:
        line = line.strip()
        if "|" in line and not table_started:
            table_started = True
            header = [cell.strip() for cell in line.split("|") if cell.strip()]
            table_data.append(header)
        elif table_started and (re.match(r"^[-|]+$", line) or line == ""):
            continue
        elif table_started and ("\t" in line or "|" in line):
            parts = line.split("\t") if "\t" in line else line.split("|")
            parts = [p.strip() for p in parts if p.strip()]
            if parts:
                table_data.append(parts)
        elif table_started and not line:
            break
    return table_data


def create_markdown_table(data):
    if not data:
        return "ไม่มีข้อมูลสำหรับสร้างตาราง"
    col_widths = [max(len(str(row[i])) for row in data) for i in range(len(data[0]))]
    header = (
        "| "
        + " | ".join(f"{data[0][i]:<{col_widths[i]}}" for i in range(len(data[0])))
        + " |"
    )
    separator = "|" + "|".join("-" * (width + 2) for width in col_widths) + "|"
    rows = [
        "| " + " | ".join(f"{row[i]:<{col_widths[i]}}" for i in range(len(row))) + " |"
        for row in data[1:]
    ]
    return "\n".join([header, separator] + rows)


def make_clickable(title, url):
    return f'<a href="{url}" target="_blank">{title}</a>'


def slowly_display_text(text, delay=0.001):
    placeholder = st.empty()
    displayed_text = ""
    for char in text:
        displayed_text += char
        placeholder.markdown(displayed_text)
        time.sleep(delay)


def setup_sidebar():
    st.sidebar.image(
        "images/AIU.jpeg",
        width=100,
        caption="UnderWriting",
        use_column_width=True,
        clamp=True,
    )
    st.sidebar.title("ข้อมูลเพิ่มเติม")
    st.sidebar.info(
        """
        ### เกี่ยวกับ AI E.R.A.
        AI E.R.A. คือเครื่องมือรวมรวมข้อมูลทางการเงินและข่าวสารเพื่อนำมาวิเคราะห์ข้อมูลทางการเงินของบริษัทโดยใช้ AI 
        
        พัฒนาโดยทีม AI-UNDERWRITING [DDO-Krungsri Auto]

        ### วิธีใช้งาน
        1. กรอกชื่อบริษัทหรือหน่วยงานที่ต้องการค้นหาในช่องที่กำหนด
        2. คลิกปุ่ม "ค้นหาและวิเคราะห์ข้อมูล"
        3. รอผลการวิเคราะห์สักครู่
        
        ### วิธีการดาวน์โหลดผลการวิเคราะห์
        1. เปิดหน้า Web ให้เลือกที่ : ที่มุมขวา แล้วกด Print เลือก บันทึกเเป็น pdf
        2. กด Ctrl + P และเลือก บันทึกเเป็น pdf

        ### ติดต่อเรา
        หากมีปัญหาในการใช้งาน กรุณาติดต่อ:
        - อีเมล: ai_era@krungsri.com
        - โทร: 02-123-4567
        - Line Official: @AIERA
        """
    )


def setup_main_content():
    banner_image = Image.open("images/J7.jpeg")
    resized_image = banner_image.resize((1000, 200))
    st.image(resized_image, use_column_width=True)
    st.markdown(
        """
        <style>
        .bold-text { font-weight: bold; }
        .underline-text { text-decoration: underline; }
        .bold-underline-text { font-weight: bold; text-decoration: underline; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def setup_credentials():
    with st.sidebar:
        aws_access_key_id = st.text_input("AWS Access Key", type="password")
        aws_secret_access_key = st.text_input("AWS Secret Key", type="password")
        aws_session_token = st.text_input("AWS Session Token", type="password")
        st.markdown(
            "[Get AWS credentials](https://d-966771b2b1.awsapps.com/start/#/?tab=accounts)"
        )
        api_key = st.text_input("Google API Key", type="password")
        cse_id = st.text_input("Google CSE ID", type="password")
    return aws_access_key_id, aws_secret_access_key, aws_session_token, api_key, cse_id


def set_environment_variables(
    aws_access_key_id, aws_secret_access_key, aws_session_token, api_key, cse_id
):
    os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
    os.environ["AWS_SESSION_TOKEN"] = aws_session_token
    os.environ["api_key"] = api_key
    os.environ["cse_id"] = cse_id


def clear_data():
    st.session_state.company_name = ""
    st.session_state.analysis_done = False
    st.session_state.input_key = str(uuid.uuid4())
    st.rerun()


def display_financial_analysis(formatted_financial_analysis):
    table_data = extract_table_data(formatted_financial_analysis)
    table_created = False
    for item in formatted_financial_analysis:
        if ("|" in item or "\t" in item) and not table_created:
            financial_table = create_markdown_table(table_data)
            st.markdown(financial_table)
            table_created = True
        elif "|" not in item and "\t" not in item and item.strip() != "":
            slowly_display_text(item, delay=0.001)


def display_references(company_news, url_fin):
    if company_news:
        st.markdown("### แหล่งอ้างอิง:")
        for url in url_fin:
            clickable_title1 = make_clickable(url["title"], url["url"])
            st.markdown(clickable_title1, unsafe_allow_html=True)
        for entry in company_news:
            clickable_title = make_clickable(entry["title"], entry["url"])
            st.markdown(clickable_title, unsafe_allow_html=True)


def display_feedback():
    feedback = [
        {
            "Feedback": "Feedback",
            "link": "https://forms.office.com/r/ST5ngnxfNB?origin=lprLink",
        }
    ]
    if feedback:
        st.markdown("### ขอบคุณที่ใช้บริการ")
        st.markdown("กรุณาให้ Feedback กับการใช้งาน AI E.R.A. ในครั้งนี้ โดยคลิกที่ลิงค์ด้านล่างนี้")
        for feed in feedback:
            get_feedback = make_clickable(feed["Feedback"], feed["link"])
            st.markdown(get_feedback, unsafe_allow_html=True)


def process_and_display_results(company_name, llm):
    juristic_id, symbol, company_news, juris_id = get_juristic_id_news(
        company_name=company_name, llm=llm
    )
    print("Symbol: ", symbol)
    if not juristic_id or any(
        phrase in juristic_id
        for phrase in ["ขออภัยค่ะ", "ไม่มีเลขทะเบียนนิติบุคคล", "ไม่พบข้อมูล", "ไม่สามารถค้นหาได้"]
    ):
        st.error(f"ไม่พบข้อมูลของบริษัท {company_name}")
        st.warning("กรุณาระบุชื่อบริษัทใหม่อีกครั้ง")
        return

    slowly_display_text(
        f"ผลการค้นหาเลขนิติบุคคลของ {company_name} คือ {juristic_id}", delay=0.009
    )

    fin_data, data, url_fin = get_financial_data(juristic_id=juristic_id, symbol=symbol)

    officers = fin_data["assetProfile"].get("companyOfficers", [])
    officer_names = [officer["name"] for officer in officers if "name" in officer]
    officer_title = [officer["title"] for officer in officers if "title" in officer]
    company_officers = pd.DataFrame(officer_names, officer_title).reset_index()

    (
        formatted_company_details_analysis,
        formatted_financial_analysis,
    ) = run_analysis_in_parallel(
        llm, company_name, data, fin_data, company_news, company_officers
    )

    st.subheader("ผลการค้นหาและวิเคราะห์ข้อมูล")
    for item in formatted_company_details_analysis:
        slowly_display_text(item, delay=0.001)

    display_financial_analysis(formatted_financial_analysis)
    display_references(company_news, url_fin)
    display_feedback()

    st.session_state.analysis_done = True


def main():
    st.set_page_config(
        page_title="AI E.R.A. for Analyzing the Company's Financial",
        page_icon="images/AIU.jpeg",
        layout="wide",
    )

    setup_sidebar()
    setup_main_content()

    (
        aws_access_key_id,
        aws_secret_access_key,
        aws_session_token,
        api_key,
        cse_id,
    ) = setup_credentials()

    if (
        not aws_access_key_id
        or not aws_secret_access_key
        or not aws_session_token
        or not api_key
        or not cse_id
    ):
        st.warning("กรุณากรอก AWS credentials หรือ Google API Key หรือ CSE_ID ให้ครบถ้วน")
        return

    try:
        bedrock_runtime = initialize_bedrock_client(
            aws_access_key_id, aws_secret_access_key, aws_session_token
        )
        llm = BedrockChat(model_id=MODEL_ID, client=bedrock_runtime)
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อกับ AWS Bedrock: {e}")
        return

    set_environment_variables(
        aws_access_key_id, aws_secret_access_key, aws_session_token, api_key, cse_id
    )

    company_name = st.text_input(
        "กรุณาระบุชื่อบริษัทที่ต้องการค้นหา",
        value=st.session_state.company_name,
        key=st.session_state.input_key,
    )

    st.markdown(
        "****เฉพาะลูกค้าประเภท **<u>บุคคลธรรมดา</u>** เท่านั้น***", unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    if col1.button("เคลียร์ข้อมูล"):
        clear_data()

    st.session_state.company_name = company_name

    if col2.button("ค้นหาและวิเคราะห์ข้อมูล"):
        if not company_name.strip():
            st.error("กรุณาระบุชื่อบริษัท")
        else:
            with st.spinner("กำลังวิเคราะห์... กรุณารอสักครู่นะคะ"):
                try:
                    start_time = time.time()
                    process_and_display_results(company_name, llm)
                    end_time = time.time()
                    st.success(
                        f"การวิเคราะห์เสร็จสิ้น ใช้เวลาทั้งหมด {end_time - start_time:.2f} วินาที"
                    )
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการดึงข้อมูล: {e}")
                    st.warning("กรุณาระบุชื่อบริษัทใหม่อีกครั้ง")

    # if st.session_state.analysis_done:
    #     if st.button("ดาวน์โหลดผลการวิเคราะห์ (PDF)"):
    #         try:
    #             pdf = create_pdf_report(company_name, st.session_state.analysis_results)
    #             st.download_button(
    #                 label="คลิกเพื่อดาวน์โหลด PDF",
    #                 data=pdf,
    #                 file_name=f"{company_name}_analysis.pdf",
    #                 mime="application/pdf",
    #             )
    #         except Exception as e:
    #             st.error(f"เกิดข้อผิดพลาดในการสร้าง PDF: {e}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### เวอร์ชันแอปพลิเคชัน")
    st.sidebar.info("AI E.R.A. v1.0.0")
    st.sidebar.markdown("© 2024 AI-UNDERWRITING [DDO-Krungsri Auto]")


if __name__ == "__main__":
    main()
