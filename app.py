import os
import json
import pandas as pd
import streamlit as st
import gspread
from openai import OpenAI
from google.oauth2.service_account import Credentials
from datetime import datetime, timezone, timedelta
from streamlit_cookies_manager import CookieManager
from dotenv import load_dotenv

# --- Page Config ---
st.set_page_config(page_title="AI Diet Tracker", page_icon="🥗", layout="wide")

# --- Load Environment Variables & Secrets ---
# This part handles both local development (with .env) and Streamlit Cloud deployment (with st.secrets)
if os.path.exists('.env'):
    load_dotenv()
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    SPREADSHEET_ID = os.getenv('SPREADSHEET_ID')
    GOOGLE_SA_FILE = os.getenv('GOOGLE_SA_FILE')
else:
    OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
    SPREADSHEET_ID = st.secrets['SPREADSHEET_ID']
    # For deployment, the JSON content is stored directly in secrets
    GOOGLE_SA_JSON = st.secrets["gcp_service_account"]

# --- USER MANAGEMENT & COOKIE MANAGER ---
USER_LIST = ["Suyash", "Divyanshi"]
cookies = CookieManager()

# --- GSheets & OpenAI Client Initialization (Cached) ---
@st.cache_resource
def initialize_gspread_client():
    """Initializes and returns gspread client, handling both local and deployed environments."""
    try:
        if 'GOOGLE_SA_JSON' in globals():
            creds_dict = GOOGLE_SA_JSON
        else:
            creds_dict = json.load(open(GOOGLE_SA_FILE))
            
        SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
        creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        gc = gspread.authorize(creds)
        spreadsheet = gc.open_by_key(SPREADSHEET_ID)
        return {'logs': spreadsheet.worksheet('Logs'), 'settings': spreadsheet.worksheet('Settings')}
    except Exception as e:
        st.error(f"Failed to connect to Google Sheets: {e}"); return None

@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)

# --- Data Fetching ---
@st.cache_data(ttl=60)
def fetch_all_data(_clients):
    """Fetches all data, caching for 60 seconds. Converts Date column for proper filtering."""
    try:
        logs_df = pd.DataFrame(_clients['logs'].get_all_records())
        settings_df = pd.DataFrame(_clients['settings'].get_all_records())
        if not logs_df.empty and 'Date' in logs_df.columns:
            logs_df['Date'] = pd.to_datetime(logs_df['Date'])
        return logs_df, settings_df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame(), pd.DataFrame()

# --- Helper Functions ---
def get_calorie_goal(settings_df, user):
    """Gets a user's calorie goal from the settings DataFrame."""
    if settings_df.empty or 'User' not in settings_df.columns: return 2000
    user_settings = settings_df[(settings_df['User'] == user) & (settings_df['Setting'] == 'Calorie Goal')]
    if not user_settings.empty: return int(user_settings.iloc[0]['Value'])
    return 2000

def set_calorie_goal(settings_ws, user, new_goal):
    """Robustly updates or creates a user's calorie goal setting."""
    try:
        cell = settings_ws.find(user, in_column=1)
        if cell:
            settings_ws.update_cell(cell.row, 3, new_goal)
        else:
            settings_ws.append_row([user, 'Calorie Goal', new_goal])
        return True
    except Exception as e:
        st.error(f"Could not save calorie goal: {e}")
        return False

def get_daily_logs(logs_df, user, selected_date):
    """Filters the main logs DataFrame for a specific user and date."""
    if logs_df.empty or 'User' not in logs_df.columns: return pd.DataFrame()
    user_df = logs_df[logs_df['User'] == user]
    daily_df = user_df[user_df['Date'].dt.date == selected_date].copy()
    for col in ['calories_kcal', 'protein_g', 'carbs_g', 'fat_g', 'fiber_g']:
        if col in daily_df.columns:
            daily_df[col] = pd.to_numeric(daily_df[col], errors='coerce').fillna(0)
    return daily_df

def log_to_google_sheet(logs_ws, new_log_entry):
    """Appends a new row (as a list) to the Google Sheet."""
    try:
        logs_ws.append_row(new_log_entry, value_input_option='USER_ENTERED')
        return True
    except Exception as e:
        st.error(f"Error saving to Google Sheets: {e}"); return False

def get_nutrition_data(user_prompt):
    """Calls OpenAI API with a detailed and robust prompt."""
    client = get_openai_client()
    SYSTEM_PROMPT = (
        "You are a nutrition parser. Given a meal, return a single, valid JSON object and nothing else. "
        "The JSON must contain: 'meal' (string), 'items' (list of objects), and 'totals' (an object). "
        "Each item in the 'items' list must have: 'name' (string), 'quantity_text' (string), 'calories_kcal' (float), "
        "'protein_g' (float), 'carbs_g' (float), 'fat_g' (float), and 'fiber_g' (float). "
        "The 'totals' object must sum all numeric values from the items. "
        "Round all numbers to one decimal place. Use Indian cuisine defaults for estimates. "
        "Example output format: "
        '{\n'
        '  "meal": "Breakfast",\n'
        '  "items": [\n'
        '    {"name": "Dosa", "quantity_text": "2", "calories_kcal": 210.0, "protein_g": 5.2, "carbs_g": 38.0, "fat_g": 4.0, "fiber_g": 2.2},\n'
        '    {"name": "Sambar", "quantity_text": "1 cup", "calories_kcal": 150.5, "protein_g": 8.0, "carbs_g": 20.0, "fat_g": 4.5, "fiber_g": 5.0}\n'
        '  ],\n'
        '  "totals": {\n'
        '    "calories_kcal": 360.5,\n'
        '    "protein_g": 13.2,\n'
        '    "carbs_g": 58.0,\n'
        '    "fat_g": 8.5,\n'
        '    "fiber_g": 7.2\n'
        '  }\n'
        '}'
    )
    response = client.chat.completions.create(model="gpt-4o-mini", messages=[{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': user_prompt}], temperature=0.0, max_tokens=700, response_format={"type": "json_object"})
    return response.choices[0].message.content

def display_weekly_trend(logs_df, user, end_date):
    """Calculates and displays the user's calorie trend for the 7 days ending on end_date."""
    st.subheader("Weekly Calorie Trend")
    if logs_df.empty or 'Date' not in logs_df.columns:
        st.info("Log some meals to see your weekly trend."); return
    user_df = logs_df[logs_df['User'] == user]
    if user_df.empty:
        st.info("No meals logged for this user yet."); return
    start_date = end_date - timedelta(days=6)
    weekly_df = user_df[(user_df['Date'].dt.date >= start_date) & (user_df['Date'].dt.date <= end_date)]
    if weekly_df.empty:
        st.info("No meals logged in the selected 7-day period."); return
    daily_calories = weekly_df.groupby(weekly_df['Date'].dt.date)['calories_kcal'].sum()
    st.line_chart(daily_calories)
    avg_calories = daily_calories.mean()
    st.caption(f"Average daily calories for this period: **{avg_calories:.0f} kcal**")

# --- Streamlit UI ---
st.title("🥗 AI Diet Tracker")

if cookies.ready():
    clients = initialize_gspread_client()
    if not clients: st.stop()

    # --- Initialize Session State (The App's Memory) ---
    if 'user' not in st.session_state:
        last_user = cookies.get('last_user')
        st.session_state.user = last_user if last_user in USER_LIST else USER_LIST[0]
    if 'logs_df' not in st.session_state or 'settings_df' not in st.session_state:
        st.session_state.logs_df, st.session_state.settings_df = fetch_all_data(clients)
    if 'calorie_goal' not in st.session_state:
        st.session_state.calorie_goal = get_calorie_goal(st.session_state.settings_df, st.session_state.user)

    def on_user_change():
        exp_date = datetime.now() + timedelta(days=365)
        cookies['last_user'] = (st.session_state.user, {'expires_at': exp_date.isoformat()})
        st.session_state.calorie_goal = get_calorie_goal(st.session_state.settings_df, st.session_state.user)

    with st.sidebar:
        st.header("User Profile")
        st.radio("Select User:", USER_LIST, key='user', horizontal=True, on_change=on_user_change)
        st.write(f"Tracking for: **{st.session_state.user}**")
        if st.button("Refresh Data from Sheet"):
            st.cache_data.clear()
            st.session_state.pop('logs_df', None) # Force re-fetch
            st.session_state.pop('settings_df', None)
            st.rerun()
            
    col1, col2 = st.columns([1, 2])
    with col1:
        st.header("Date")
        selected_date = st.date_input("Select a date", datetime.today().date(), key="date_selector")
        
        with st.form("settings_form"):
            st.subheader("Your Calorie Goal")
            new_goal = st.number_input("Daily Goal (kcal)", min_value=0, step=50, value=st.session_state.calorie_goal, key="goal_input")
            if st.form_submit_button("Save Goal"):
                st.session_state.calorie_goal = new_goal
                if set_calorie_goal(clients['settings'], st.session_state.user, new_goal):
                    st.success(f"Goal saved as {new_goal} kcal!")
                    st.cache_data.clear()
                st.rerun()

        st.markdown("---")
        st.subheader("Daily Summary")
        daily_df = get_daily_logs(st.session_state.logs_df, st.session_state.user, selected_date)
        
        total_calories = daily_df['calories_kcal'].sum() if 'calories_kcal' in daily_df.columns else 0
        total_protein = daily_df['protein_g'].sum() if 'protein_g' in daily_df.columns else 0
        total_carbs = daily_df['carbs_g'].sum() if 'carbs_g' in daily_df.columns else 0
        total_fat = daily_df['fat_g'].sum() if 'fat_g' in daily_df.columns else 0
        total_fiber = daily_df['fiber_g'].sum() if 'fiber_g' in daily_df.columns else 0
        
        st.metric("Total Calories Today", f"{total_calories:.1f} kcal", delta_color="inverse" if total_calories > st.session_state.calorie_goal else "normal")
        calorie_progress = min(total_calories / st.session_state.calorie_goal, 1.0) if st.session_state.calorie_goal > 0 else 0
        st.progress(calorie_progress, text=f"{int(calorie_progress * 100)}% of {st.session_state.calorie_goal} kcal goal")
        
        st.markdown(f"**Protein:** {total_protein:.1f}g")
        st.markdown(f"**Carbs:** {total_carbs:.1f}g")
        st.markdown(f"**Fat:** {total_fat:.1f}g")
        st.markdown(f"**Fiber:** {total_fiber:.1f}g")

        st.markdown("---")
        if not daily_df.empty:
            st.subheader("Today's Meals")
            for index, row in daily_df.iterrows():
                st.write(f"**{row.get('Meal', 'N/A')}**: {row.get('items_text', 'N/A')} ({row.get('calories_kcal', 0):.1f} kcal)")
        else: 
            st.info("No meals logged for this date yet.")
            
        st.markdown("---")
        display_weekly_trend(st.session_state.logs_df, st.session_state.user, selected_date)

    with col2:
        st.header(f"Log Meals for {selected_date.strftime('%A, %B %d, %Y')}")
        meal_sections = {
            "Breakfast": {"emoji": "☀️", "placeholder": "Example:\n2 idli\n1 cup sambar"},
            "Lunch": {"emoji": "🍜", "placeholder": "Example:\n1 bowl rice\n1 cup chicken curry"},
            "Dinner": {"emoji": "🌙", "placeholder": "Example:\n2 roti\n1 bowl vegetable sabzi"}
        }
        for meal_name, details in meal_sections.items():
            st.subheader(f"{details['emoji']} {meal_name}")
            with st.form(f"{meal_name.lower()}_form", clear_on_submit=True):
                items_input = st.text_area(f"Log {meal_name}", height=100, placeholder=details['placeholder'], key=f"{meal_name.lower()}_input", label_visibility="collapsed")
                if st.form_submit_button(f"Log {meal_name}"):
                    if items_input.strip():
                        with st.spinner(f"Analyzing {meal_name}..."):
                            items_list = [line.strip() for line in items_input.split("\n") if line.strip()]
                            user_prompt = f"Meal: {meal_name}\nItems:\n" + "\n".join(f"- {it}" for it in items_list)
                            raw_response = get_nutrition_data(user_prompt)
                            parsed_data = json.loads(raw_response)
                            totals = parsed_data.get('totals', {})
                            
                            new_row_dict = {
                                'User': st.session_state.user, 'Timestamp': datetime.now(timezone.utc).isoformat(),
                                'Date': selected_date.strftime('%Y-%m-%d'), 'Meal': meal_name,
                                'items_text': "; ".join(items_list),
                                'calories_kcal': float(totals.get('calories_kcal', 0)), 'protein_g': float(totals.get('protein_g', 0)),
                                'carbs_g': float(totals.get('carbs_g', 0)), 'fat_g': float(totals.get('fat_g', 0)),
                                'fiber_g': float(totals.get('fiber_g', 0)), 'json_data': json.dumps(parsed_data)
                            }
                            
                            new_row_df = pd.DataFrame([new_row_dict])
                            # Ensure date column is of the correct type before concatenating
                            new_row_df['Date'] = pd.to_datetime(new_row_df['Date'])
                            st.session_state.logs_df = pd.concat([st.session_state.logs_df, new_row_df], ignore_index=True)
                            
                            log_to_google_sheet(clients['logs'], list(new_row_dict.values()))
                            
                            st.success(f"{meal_name} logged!")
                            st.rerun()
                    else: 
                        st.error(f"Please enter items for {meal_name}.")