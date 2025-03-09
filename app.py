import asyncio
import logging
import nest_asyncio
import pandas as pd
import re
import streamlit as st
from datetime import datetime, timedelta
from crawl4ai import AsyncWebCrawler
from googlesearch import search
import time

nest_asyncio.apply()  # แก้ปัญหา asyncio ใน Jupyter Notebook
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# ฟังก์ชันค้นหาข่าวย้อนหลัง 12 เดือน
async def search_news(comp_name):
    start_time = time.time()
    one_year_ago = datetime.now() - timedelta(days=365)
    query = f"ข่าวเกี่ยวกับ {comp_name} after:{one_year_ago.strftime('%Y-%m-%d')}"
    news_results = {}

    try:
        news_query = search(query, num_results=10, advanced=True)
        for i, result in enumerate(news_query):
            news_results[f"news_{i}"] = {
                "title": result.title,
                "url": result.url,
                "snippet": result.description,
            }
    except Exception as e:
        logging.error(f"Error searching news: {e}")

    print(f"\n Running time process Search News: {time.time() - start_time}")
    return news_results


# ฟังก์ชันสำหรับการ scrape ข้อมูลจาก URL
async def scrape_data(url):
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            return result.markdown
    except Exception as e:
        logging.error(f"Error occurred while scraping: {e}", exc_info=True)
        return None


async def main(comp_name):
    # สร้าง URL สำหรับการค้นหา
    url = f"https://data.creden.co/search?q={comp_name}&type_search=keyword"

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        print(f"URL: {url}")

        # ดึงข้อความผลลัพธ์ที่เกี่ยวข้อง
        extracted_text = result.markdown
        print(f"Extracted Text: {extracted_text}")

        start_idx = extracted_text.find("คำที่ค้นหา :")
        match = re.search(r"(##### ผลลัพธ์การค้นหา|ตัวช่วยค้นหา|\* <<)", extracted_text)
        end_idx = match.start() if match else len(extracted_text)

        print(f"Start Index: {start_idx}")
        print(f"End Index: {end_idx}")

        if start_idx != -1 and end_idx != -1:
            extracted_text2 = (
                extracted_text[start_idx:end_idx].replace("</", "").replace(">", "")
            )
        else:
            extracted_text2 = "ไม่พบข้อความที่ต้องการ"

        print(f"Extracted Text2: {extracted_text2}")

        # ค้นหาจำนวนผลลัพธ์ทั้งหมดจากข้อความ
        total_results_match = re.search(r"ผลลัพธ์การค้นหา (\d+) รายการ", extracted_text)
        total_results = int(total_results_match.group(1)) if total_results_match else 0

        # แสดงข้อความที่เหมาะสม
        if total_results > 10:
            summary_message = f"ผลลัพธ์การค้นหาทั้งหมด {total_results} รายการ แต่ในที่นี้จะนำมาแสดงเพียง 10 รายการแรกเท่านั้น ดังนี้"
            st.sidebar.write(summary_message)
            # แสดงผลข้อความที่ต้องการ
            st.sidebar.write(extracted_text2)
        else:
            summary_message = f"ผลลัพธ์การค้นหาทั้งหมด {total_results} รายการ ดังนี้"
            st.sidebar.write(summary_message)
            st.sidebar.write(extracted_text2)

        # ดึงข้อมูลตัวเลือกและ URL
        options = []
        urls = []
        for line in extracted_text.splitlines():
            if "### [" in line and "](" in line:
                option_text = line.split("[")[1].split("]")[0]
                url_text = (
                    line.split("(")[1].split(")")[0].replace("</", "").replace(">", "")
                )
                options.append(option_text)
                urls.append(url_text)

        if options:
            selected_option = st.selectbox(
                "โปรดเลือกบริษัทเพื่อค้นหาข้อมูลและข่าวเพิ่มเติม",
                ["โปรดเลือกบริษัทเพื่อค้นหาข้อมูลและข่าวเพิ่มเติม"] + options,
            )

            if selected_option != "โปรดเลือกบริษัทเพื่อค้นหาข้อมูลและข่าวเพิ่มเติม":
                selected_index = options.index(selected_option)
                print(f"Selected Index: {selected_index}")
                selected_url = urls[selected_index]

                scraped_data = await scrape_data(selected_url)
                scraped_data = scraped_data.replace("|", "")
                print(f"Scraped Data : \n{scraped_data}")
                if scraped_data:
                    # สร้าง dictionary สำหรับข้อมูลที่ต้องการ
                    company_info = {
                        "ชื่อนิติบุคคล": None,
                        "ชื่อนิติบุคคลภาษาอังกฤษ": None,
                        "เลขทะเบียน": None,
                        "วันที่จดทะเบียน": None,
                        "สถานภาพกิจการ": None,
                        "วันที่เลิก": None,
                        "ประเภทธุรกิจ": None,
                        "ทุนจดทะเบียน": None,
                        "มูลค่าบริษัท": None,
                        "ขนาดธุรกิจ": None,
                        "หมวดธุรกิจ": None,
                        "กลุ่มธุรกิจ": None,
                        "วัตถุประสงค์": None,
                        "ที่อยู่": None,
                        "ข้อมูลอัปเดตเมื่อ": None,
                    }

                    # ตรวจหาข้อมูลที่ต้องการจาก scraped_data
                    if "ข้อมูลทั่วไปของ" in scraped_data:
                        company_info["ชื่อนิติบุคคล"] = (
                            scraped_data.split("ข้อมูลทั่วไปของ")[1].split("\n")[0].strip()
                        )
                    if "ชื่อนิติบุคคล" in scraped_data:
                        company_info["ชื่อนิติบุคคลภาษาอังกฤษ"] = (
                            scraped_data.split("ชื่อนิติบุคคล")[1].split("\n")[0].strip()
                        )
                    if "เลขทะเบียนนิติบุคคล" in scraped_data:
                        company_info["เลขทะเบียน"] = (
                            scraped_data.split("เลขทะเบียนนิติบุคคล")[1]
                            .split("\n")[0]
                            .strip()
                        )
                    if "วันเดือนปีที่จดทะเบียน" in scraped_data:
                        company_info["วันที่จดทะเบียน"] = (
                            scraped_data.split("วันเดือนปีที่จดทะเบียน")[1]
                            .split("\n")[0]
                            .strip()
                        )
                    if "สถานภาพกิจการ" in scraped_data:
                        company_info["สถานภาพกิจการ"] = (
                            scraped_data.split("สถานภาพกิจการ")[1].split("\n")[0].strip()
                        )
                    if "วันที่เลิก" in scraped_data:
                        company_info["วันที่เลิก"] = (
                            scraped_data.split("วันที่เลิก")[1].split("\n")[0].strip()
                        )
                    if "ประเภทธุรกิจ" in scraped_data:
                        company_info["ประเภทธุรกิจ"] = (
                            scraped_data.split("ประเภทธุรกิจ")[1].split("\n")[0].strip()
                        )
                    if "ทุนจดทะเบียนปัจจุบัน (บาท)" in scraped_data:
                        company_info["ทุนจดทะเบียน"] = (
                            scraped_data.split("ทุนจดทะเบียนปัจจุบัน (บาท)")[1]
                            .split("\n")[0]
                            .strip()
                        )
                    if "มูลค่าบริษัท" in scraped_data:
                        company_info["มูลค่าบริษัท"] = (
                            scraped_data.split("มูลค่าบริษัท")[1].split("\n")[0].strip()
                        )
                    if "ขนาดธุรกิจ" in scraped_data:
                        # ดึงเฉพาะคำสุดท้ายจาก ขนาดธุรกิจ
                        size_business = (
                            scraped_data.split("ขนาดธุรกิจ")[1].split("\n")[0].strip()
                        )
                        company_info["ขนาดธุรกิจ"] = size_business.split()[-1]
                    if "หมวดธุรกิจ (A-U)" in scraped_data:
                        # ดึงข้อความใน [ ] จาก หมวดธุรกิจ
                        match = re.search(
                            r"\[(.*?)\]", scraped_data.split("หมวดธุรกิจ (A-U)")[1]
                        )
                        if match:
                            company_info["หมวดธุรกิจ"] = match.group(1).strip()
                    if "กลุ่มธุรกิจ (TSIC)" in scraped_data:
                        # ดึงข้อความใน [ ] จาก กลุ่มธุรกิจ
                        match = re.search(
                            r"\[(.*?)\]", scraped_data.split("กลุ่มธุรกิจ (TSIC)")[1]
                        )
                        if match:
                            company_info["กลุ่มธุรกิจ"] = match.group(1).strip()
                    if "วัตถุประสงค์" in scraped_data:
                        company_info["วัตถุประสงค์"] = (
                            scraped_data.split("วัตถุประสงค์")[1].split("\n")[0].strip()
                        )
                    if "###  ที่อยู่ " in scraped_data:
                        company_info["ที่อยู่"] = (
                            scraped_data.split("###  ที่อยู่")[1].split("\n")[1].strip()
                        )
                    if "* - ข้อมูลอัพเดทเมื่อ" in scraped_data:
                        company_info["ข้อมูลอัปเดตเมื่อ"] = (
                            scraped_data.split("* - ข้อมูลอัพเดทเมื่อ")[1]
                            .split("\n")[0]
                            .strip()
                            .replace(" คลิกเพื่ออัพเดท", "")
                        )
                    # สร้าง DataFrame ในรูปแบบ long form
                    long_form_data = []
                    for key, value in company_info.items():
                        long_form_data.append({"หัวข้อ": key, "ข้อมูล": value})

                    # แปลงข้อมูลเป็น DataFrame ไม่ต้องเอา index มาแสดง
                    df = pd.DataFrame(long_form_data).set_index("หัวข้อ")
                    # df = pd.DataFrame(long_form_data)
                    st.write(df)

                    # ค้นหาข่าวย้อนหลัง 12 เดือน
                    news_results = await search_news(comp_name=company_info["ชื่อนิติบุคคล"])

                    # แสดงข่าวใน Streamlit
                    if news_results:
                        st.write("### ข่าวที่เกี่ยวข้อง:")
                        st.write(company_info["ชื่อนิติบุคคล"])
                        for key, news in news_results.items():
                            st.write(f"🔹 **{news['title']}**")  # แสดงหัวข้อข่าว
                            st.write(f"📌 {news['snippet']}")  # แสดงคำอธิบาย
                            st.write(f"[อ่านข่าว]({news['url']})")  # แสดงลิงก์ไปยังข่าว
                            st.write("---")
                    else:
                        st.write("❌ ไม่พบข่าวที่เกี่ยวข้อง")
                else:
                    st.write("ไม่สามารถดึงข้อมูลได้")
        else:
            st.write("ไม่พบตัวเลือกใดๆ")


# ฟังก์ชันรีเฟรชหน้าจอ
def clear_all():
    """เคลียร์ข้อมูลและรีเฟรชหน้าใหม่"""
    st.session_state.clear()  # เคลียร์ค่าทุกอย่างใน session_state
    # st.rerun()  # รีเฟรชหน้าจอใหม่


# สร้าง UI
if __name__ == "__main__":
    st.title("🔍 ระบบค้นหาข้อมูลบริษัท")
    comp_name = st.text_input("กรอกชื่อบริษัทหรือเลขนิติบุคคลที่ต้องการค้นหา")

    col1, col2 = st.columns([1, 1])

    with col1:
        search_button = st.button("ค้นหา")

    with col2:
        clear_button = st.button("เคลียร์ข้อมูล", on_click=clear_all)

    # ทำงานเมื่อกด Enter หรือกดปุ่ม "ค้นหา"
    if search_button or comp_name:
        asyncio.run(main(comp_name))
    if clear_button:
        #     clear_all()
        st.session_state.clear()
    # if comp_name:
    #     asyncio.run(main(comp_name))
