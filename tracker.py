import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import streamlit as st
import json

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

def get_sheet():
    try:
        creds_dict = json.loads(st.secrets["GOOGLE_CREDS"])
        creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        client = gspread.authorize(creds)
        sheet = client.open("rag-chatbot-logs").sheet1
        return sheet
    except Exception as e:
        print(f"Sheet connection error: {e}")
        return None

def log_chat(user_message, bot_response, pdf_active):
    try:
        sheet = get_sheet()
        if sheet:
            sheet.append_row([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                user_message,
                bot_response,
                "Yes" if pdf_active else "No"
            ])
    except Exception as e:
        print(f"Logging error: {e}")

def get_visitor_count():
    try:
        sheet = get_sheet()
        if sheet:
            # Count rows minus header
            return max(0, len(sheet.get_all_values()) - 1)
        return 0
    except:
        return 0