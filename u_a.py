import os
import sqlite3
import bcrypt
import validators
import streamlit as st
import time
import logging
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from yt_dlp import YoutubeDL
from langchain.schema import Document
from fpdf import FPDF
from datetime import datetime
from gtts import gTTS
import tempfile
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, quote
import base64
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import parse_qs, urlparse
import json
from mindmap_utils import add_semantic_mindmap_section

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit app configuration - MUST BE FIRST
st.set_page_config(page_title="Enhanced Content Summarizer", page_icon="🌟")

# Database setup
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL)''')
    conn.commit()
    conn.close()

# Function to hash passwords
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# Function to check password
def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode(), hashed_password.encode())

# Function to register a user
def register_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

# Function to authenticate a user
def authenticate_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    if user and verify_password(password, user[0]):
        return True
    return False

# Add this near the beginning of your code where you define other functions
def logout_user():
    """Reset authentication state to log out the user"""
    for key in list(st.session_state.keys()):
        # Keep dark mode preference but clear everything else
        if key != "dark_mode":
            del st.session_state[key]
    
    # Reinitialize the authentication state
    st.session_state["authenticated"] = False
    st.session_state["username"] = ""
    st.session_state["messages"] = []
# Initialize database
init_db()

# Session state for authentication and theme
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["username"] = ""

# Initialize theme state
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

# Light bulb icons for toggle button (Base64 encoded for consistency)
def get_bulb_icon(is_on=True):
    if is_on:
        return "💡"  # Light bulb on emoji
    else:
        return "🔅"  # Dim button emoji

# CSS for dark mode with summary text color fix
def load_css():
    if st.session_state["dark_mode"]:
        dark_mode_css = """
        <style>
            .stApp {
                background-color: #121212;
                color: #f0f0f0;
            }
            .stTextInput, .stSelectbox, .stTextArea {
                background-color: #2b2b2b !important;
                color: white!important;
            }
            .stSelectbox label {
                color: white !important;
            }
            .stTextInput label {
                color: white !important;
            }
            .stButton button {
                background-color: #ADD8E6;
                color: white;
            }
            .stSidebar {
                background-color: #1e1e1e;
            }
            .css-145kmo2 {
                background-color: #1e1e1e;
            }
            .stTabs [data-baseweb="tab-list"] {
                background-color: #1e1e1e;
            }
            .stTabs [data-baseweb="tab"] {
                color: #f0f0f0;
            }
            .stHeader {
                background-color: #1e1e1e;
            }
            .stMarkdown {
                color: #f0f0f0;
            }
            div[data-testid="stChatMessage"] {
                background-color: #2b2b2b;
                color: #f0f0f0;
            }
            .stAlert > div {
                color: #f0f0f0 !important;
            }
            .stSuccess > div {
                background-color: #1e3a2f !important;
                color: white !important;
            }
            .stInfo > div {
                background-color: #1c3c5a !important;
                color: white !important;
            }
            .stWarning > div {
                background-color: #5a4a1c !important;
                color: white !important;
            }
            .stError > div {
                background-color: #5a1c1c !important;
                color: white !important;
            }
            .stCode {
                background-color: #2b2b2b;
            }
            .stChatMessage {
                background-color: #2b2b2b;
            }
            .stDownloadButton button {
                background-color: #ADD8E6;
                color: white;
            }
            .css-zt5igj {
                background-color: transparent;
            }
            /* Custom style for the theme toggle button */
            .theme-toggle-btn {
                display: flex;
                align-items: center;
                justify-content: center;
                background-color: #2b2b2b;
                color: #f0f0f0;
                border: none;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                font-size: 20px;
                cursor: pointer;
                transition: all 0.3s ease;
                padding: 0;
            }
            .theme-toggle-btn:hover {
                background-color: #3b3b3b;
                transform: scale(1.1);
            }
            .theme-toggle-icon {
                font-size: 24px;
            }
            /* Glowing Chat Box */
            .stChatInputContainer {
                border: 2px solid #ADD8E6;
                border-radius: 8px;
                box-shadow: 0 0 15px #ADD8E6, 0 0 20px #ADD8E6, 0 0 25px #ADD8E6;
                padding: 5px;
                animation: glow 1.5s ease-in-out infinite alternate;
            }

            @keyframes glow {
                from {
                    box-shadow: 0 0 10px #ADD8E6, 0 0 15px #ADD8E6;
                }
                to {
                    box-shadow: 0 0 20px #ADD8E6, 0 0 25px #ADD8E6, 0 0 30px #ADD8E6;
                }
            }

        /* Enhance the chat message area as well */
        div[data-testid="stChatMessage"] {
            border-left: 3px solid #ADD8E6;
            border-radius: 4px;
            transition: all 0.3s ease;
        }

        div[data-testid="stChatMessage"]:hover {
            box-shadow: 0 0 10px #ADD8E6;
        }
        .chat-header {
            text-shadow: 0 0 10px #ADD8E6, 0 0 15px #ADD8E6;
        }
        </style>
        """
        st.markdown(dark_mode_css, unsafe_allow_html=True)
    else:
        # Light mode specific styles if needed
        light_mode_css = """
        <style>
            /* Custom style for the theme toggle button in light mode */
            .theme-toggle-btn {
                display: flex;
                align-items: center;
                justify-content: center;
                background-color: #f0f0f0;
                color: #333333;
                border: none;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                font-size: 20px;
                cursor: pointer;
                transition: all 0.3s ease;
                padding: 0;
            }
            .theme-toggle-btn:hover {
                background-color: #e0e0e0;
                transform: scale(1.1);
            }
            .theme-toggle-icon {
                font-size: 24px;
            }
            /* Glowing Chat Box - Light Mode */
            .stChatInputContainer {
                border: 2px solid #ADD8E6;
                border-radius: 8px;
                box-shadow: 0 0 15px rgba(76, 175, 80, 0.6), 0 0 20px rgba(76, 175, 80, 0.4), 0 0 25px rgba(76, 175, 80, 0.2);
                padding: 5px;
                animation: glowLight 1.5s ease-in-out infinite alternate;
            }

            @keyframes glowLight {
                from {
                    box-shadow: 0 0 10px rgba(76, 175, 80, 0.4), 0 0 15px rgba(76, 175, 80, 0.2);
                }
                to {
                    box-shadow: 0 0 20px rgba(76, 175, 80, 0.6), 0 0 25px rgba(76, 175, 80, 0.4), 0 0 30px rgba(76, 175, 80, 0.2);
                }
            }

            /* Enhance the chat message area as well - Light Mode */
            div[data-testid="stChatMessage"] {
                border-left: 3px solid #ADD8E6;
                border-radius: 4px;
                transition: all 0.3s ease;
            }

            div[data-testid="stChatMessage"]:hover {
                box-shadow: 0 0 10px rgba(76, 175, 80, 0.6);
            }
            .chat-header {
                text-shadow: 0 0 10px #ADD8E6, 0 0 15px #ADD8E6;
            }
        </style>
        """
        st.markdown(light_mode_css, unsafe_allow_html=True)

# Custom theme toggle button
def theme_toggle_button():
    icon = get_bulb_icon(not st.session_state["dark_mode"])
    
    html_button = f"""
    <button class="theme-toggle-btn" onclick="document.getElementById('dark_mode_toggle').click();">
        <span class="theme-toggle-icon">{icon}</span>
    </button>
    """
    st.markdown(html_button, unsafe_allow_html=True)
    
    # Hidden button that will be clicked by the custom button
    # Using a container to minimize button visibility
    container = st.container()
    with container:
        st.markdown('<div style="height: 0.1px;"></div>', unsafe_allow_html=True)
        clicked = st.sidebar.button("Toggle Theme", key="dark_mode_toggle", help="Switch between dark and light mode")
    
    if clicked:
        st.session_state["dark_mode"] = not st.session_state["dark_mode"]
        st.rerun()
def inject_custom_css():
    """Inject custom CSS with more direct element targeting"""
    st.markdown("""
    <style>
    .stChatInputContainer, .stChatInput, div[data-baseweb="input"] {
        border: 2px solid #ADD8E6 !important;
        border-radius: 8px !important;
        box-shadow: 0 0 10px #ADD8E6, 0 0 15px #ADD8E6 !important;
        animation: chatGlow 2s ease-in-out infinite alternate !important;
    }
    
    @keyframes chatGlow {
        from {
            box-shadow: 0 0 5px #ADD8E6, 0 0 10px #ADD8E6 !important;
        }
        to {
            box-shadow: 0 0 10px #ADD8E6, 0 0 15px #ADD8E6, 0 0 20px #ADD8E6 !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)
# Load CSS based on theme
load_css()
inject_custom_css()

# Authentication UI
if not st.session_state["authenticated"]:
    st.title("Login / Register to Access Summarizer")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login")
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            if authenticate_user(login_username, login_password):
                st.session_state["authenticated"] = True
                st.session_state["username"] = login_username
                st.success("Login successful! Redirecting...")
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    with tab2:
        st.subheader("Register")
        register_username = st.text_input("New Username", key="register_username")
        register_password = st.text_input("New Password", type="password", key="register_password")
        if st.button("Register"):
            if register_user(register_username, register_password):
                st.success("Registration successful! You can now log in.")
            else:
                st.error("Username already exists. Try another.")

# Main app when authenticated
else:
    st.success(f"Welcome, {st.session_state['username']}!")
    
    # Theme toggle in sidebar using custom button
    st.sidebar.title("App Settings")
    theme_toggle_button()
    # Theme toggle in sidebar using custom button

# Add logout button to sidebar
    if st.sidebar.button("Logout"):
        logout_user()
        st.rerun()

# Summarizer App Content Starts Here
# -------------------------------
   # Summarizer App Content Starts Here
    # -------------------------------
    
    # Supported languages dictionary with their codes
    LANGUAGES = {
        "English": "en",
        "Hindi": "hi",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Chinese": "zh",
        "Japanese": "ja",
        "Korean": "ko",
        "Russian": "ru",
        "Arabic": "ar",
        "Bengali": "bn",
        "Tamil": "ta",
        "Telugu": "te"
    }

    # Summary length options
    SUMMARY_LENGTHS = {
        "Short (150 words)": 150,
        "Medium (250 words)": 250,
        "Long (300 words)": 300
    }
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "summary" not in st.session_state:
        st.session_state.summary = ""
    if "url" not in st.session_state:
        st.session_state.url = ""
    if "content_title" not in st.session_state:
        st.session_state.content_title = ""
    if "summary_generated" not in st.session_state:
        st.session_state.summary_generated = False
    if "selected_language" not in st.session_state:
        st.session_state.selected_language = "English"
    if "selected_length" not in st.session_state:
        st.session_state.selected_length = "Medium (250 words)"
    if "audio_file" not in st.session_state:
        st.session_state.audio_file = None
    if "pdf_bytes" not in st.session_state:
        st.session_state.pdf_bytes = None
    st.title("Multi-Source Content Summarizer")
    st.write("Summarize content from YouTube videos or websites in your preferred language and length.")

    # Sidebar content
    st.sidebar.title("About This App")
    st.sidebar.info(
        "This app uses LangChain and Mixtral-8x7B model from Groq API to provide customizable summaries "
        "of both YouTube videos and website content in multiple languages."
    )
    st.sidebar.title("How To Use:")
    st.write("1. Enter any URL (YouTube video or website) you wish to summarize.")
    st.write("2. Select your preferred language and summary length.")
    st.write("3. Click Summarize to get a detailed summary.")
    st.write("4. Listen to the audio version of the summary.")
    st.write("5. Download the summary as PDF if needed.")
    st.write("6. Share the summary directly via WhatsApp.")
    st.write("7. Ask follow-up questions using the chatbot!")

    # Create two columns for language and length selection
    col1, col2 = st.columns(2)

    with col1:
        selected_language = st.selectbox(
            "Select Summary Language:",
            options=list(LANGUAGES.keys()),
            index=list(LANGUAGES.keys()).index(st.session_state.selected_language)
        )

    with col2:
        selected_length = st.selectbox(
            "Select Summary Length:",
            options=list(SUMMARY_LENGTHS.keys()),
            index=list(LANGUAGES.keys()).index(st.session_state.selected_language)
        )

    # Text input for URL
    url_input = st.text_input("Enter the URL:", 

                         value=st.session_state.url,

                         placeholder="https://example.com or YouTube URL")

    if url_input != st.session_state.url:

        st.session_state.url = url_input

    def is_youtube_url(url: str) -> bool:
        """Check if the URL is a YouTube video URL."""
        youtube_domains = ['youtube.com', 'youtu.be']
        parsed_url = urlparse(url)
        return any(domain in parsed_url.netloc for domain in youtube_domains)
    def extract_youtube_video_id(url: str) -> str: 
        """Extract the video ID from a YouTube URL."""
        parsed_url = urlparse(url)
    
        if parsed_url.netloc == 'youtu.be':
            return parsed_url.path[1:]
        
        if parsed_url.netloc in ('www.youtube.com', 'youtube.com'):
            if parsed_url.path == '/watch':
                return parse_qs(parsed_url.query)['v'][0]
            elif parsed_url.path.startswith('/embed/'):
                return parsed_url.path.split('/')[2]
            elif parsed_url.path.startswith('/v/'):
                return parsed_url.path.split('/')[2]
        
        # Could not extract ID
            return None
    def get_youtube_video_details(url: str) -> dict:
        """Get title, thumbnail, and channel details for a YouTube video."""
        try:
            video_id = extract_youtube_video_id(url)
            
            if not video_id:
                return {
                    "title": "Unknown Title",
                    "thumbnail": None,
                    "channel": "Unknown Channel",
                    "success": False
                }
            
            # Use yt_dlp to get video info
            ydl_opts = {
                'quiet': True,
                'skip_download': True,
                'no_warnings': True,
                'youtube_include_dash_manifest': False
            }
            
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                return {
                    "title": info.get("title", "Unknown Title"),
                    "thumbnail": info.get("thumbnail", None),
                    "channel": info.get("uploader", "Unknown Channel"),
                    "channel_url": info.get("uploader_url", None),
                    "duration": info.get("duration", 0),
                    "view_count": info.get("view_count", 0),
                    "upload_date": info.get("upload_date", "Unknown"),
                    "success": True
                }
        
        except Exception as e:
            logger.error(f"Error getting YouTube details: {str(e)}", exc_info=True)
            return {
                "title": "Error retrieving video details",
                "thumbnail": None,
                "channel": "Unknown",
                "success": False
            }

    def display_youtube_video_info(url: str):
        """Display YouTube video information in the Streamlit UI."""
    st.subheader("Video Information")
    
    with st.spinner("Retrieving video details..."):
        video_details = get_youtube_video_details(url_input)
        
        if video_details["success"]:
            # Create columns for layout
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Display thumbnail
                if video_details["thumbnail"]:
                    st.image(video_details["thumbnail"], use_container_width=True)
                else:
                    st.info("Thumbnail not available")
            
            with col2:
                # Display video details
                st.markdown(f"**Title:** {video_details['title']}")
                st.markdown(f"**Channel:** {video_details['channel']}")
                
                # Display additional details if available
                if video_details.get("view_count"):
                    st.markdown(f"**Views:** {video_details['view_count']:,}")
                
                if video_details.get("duration"):
                    minutes = video_details['duration'] // 60
                    seconds = video_details['duration'] % 60
                    st.markdown(f"**Duration:** {minutes} min {seconds} sec")
                
                if video_details.get("upload_date"):
                    date = video_details["upload_date"]
                    try:
                        # Format date if it's in YYYYMMDD format
                        formatted_date = f"{date[0:4]}-{date[4:6]}-{date[6:8]}"
                        st.markdown(f"**Upload Date:** {formatted_date}")
                    except:
                        st.markdown(f"**Upload Date:** {date}")
                
                # Add link to channel
                if video_details.get("channel_url"):
                    st.markdown(f"[Visit Channel]({video_details['channel_url']})")
            
            # Add a divider
            st.markdown("---")
        
    # Show URL type if URL is entered
    if url_input:
        url_type = "YouTube video" if is_youtube_url(url_input) else "website"
        st.info(f"Detected URL type: {url_type}")

    # Initialize LLM
    llm = ChatGroq(
        model="mixtral-8x7b-32768",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    # Summary prompt template
    SUMMARY_PROMPT = """
    Create a comprehensive summary of the following content directly in {language}.
    The summary should be exactly {word_count} words long.

    Content: {text}

    Requirements:
    1. Generate the summary DIRECTLY in {language}
    2. Make it exactly {word_count} words
    3. Maintain natural and fluent language
    4. Use appropriate script for the language (e.g., Devanagari for Hindi)
    5. Focus on key points and main ideas
    """

    prompt = PromptTemplate(template=SUMMARY_PROMPT, input_variables=["text", "language", "word_count"])

    # IMPROVED AUDIO GENERATION FUNCTION WITH BETTER ERROR HANDLING
    def create_audio(text: str, language_code: str) -> str:
        """Create audio file from text using gTTS with enhanced error handling."""
        try:
            logger.info(f"Starting audio generation for language: {language_code}")
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, "summary_audio.mp3")
            
            # Add a timeout parameter and retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"TTS attempt {attempt+1}/{max_retries}")
                    
                    # Configure a session with appropriate timeouts
                    session = requests.Session()
                    session.mount('https://', requests.adapters.HTTPAdapter(
                        max_retries=3,
                        pool_connections=1,
                        pool_maxsize=1
                    ))
                    
                    # Use the session for gTTS
                    tts = gTTS(text=text, lang=language_code, slow=False)
                    tts.save(temp_path)
                    
                    # Verify file was created
                    if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                        logger.info(f"Audio file created successfully: {temp_path}")
                        return temp_path
                    else:
                        logger.warning("Audio file was created but is empty or doesn't exist")
                        
                except requests.exceptions.RequestException as req_err:
                    logger.warning(f"Request error on attempt {attempt+1}: {req_err}")
                    if attempt < max_retries - 1:
                        wait_time = 2 * (attempt + 1)  # Exponential backoff
                        logger.info(f"Waiting {wait_time} seconds before retrying...")
                        time.sleep(wait_time)
                    else:
                        raise req_err
                        
            logger.error("All TTS attempts failed")
            return None
            
        except requests.exceptions.ConnectionError as conn_err:
            logger.error(f"Connection error: {conn_err}")
            st.error("Failed to connect to Google TTS service. Please check your internet connection.")
        except requests.exceptions.Timeout as timeout_err:
            logger.error(f"Timeout error: {timeout_err}")
            st.error("Connection to Google TTS service timed out. Please try again later.")
        except Exception as e:
            logger.error(f"General error in create_audio: {str(e)}", exc_info=True)
            st.error(f"Error generating audio: {str(e)}")
        
        return None

    # FALLBACK OFFLINE TTS FUNCTION
    def create_audio_offline(text: str) -> str:
        """Create audio file using pyttsx3 (offline TTS) as a fallback."""
        try:
            # First check if pyttsx3 is installed
            try:
                import pyttsx3
            except ImportError:
                st.warning("Offline TTS requires pyttsx3. Installing...")
                os.system("pip install pyttsx3")
                import pyttsx3
                
            logger.info("Generating audio using offline TTS engine")
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, "summary_audio.mp3")
            
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)  # Speed
            
            # Save to file
            engine.save_to_file(text, temp_path)
            engine.runAndWait()
            
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                logger.info("Offline audio generation successful")
                return temp_path
            else:
                logger.warning("Offline audio file was not created properly")
                return None
                
        except Exception as e:
            logger.error(f"Error in offline TTS: {str(e)}", exc_info=True)
            st.error(f"Error generating offline audio: {str(e)}")
            return None

    # COMBINED AUDIO GENERATION WITH FALLBACK
    def create_audio_with_fallback(text: str, language_code: str) -> str:
        """Try online TTS first, then fall back to offline TTS if needed."""
        # Try online gTTS first
        audio_path = create_audio(text, language_code)
        if audio_path:
            return audio_path
            
        # If online TTS fails and language is English, try offline TTS
        if language_code == "en":
            st.warning("Online TTS failed. Trying offline TTS instead...")
            return create_audio_offline(text)
        else:
            st.warning("Online TTS failed. Offline TTS only supports English.")
            # Try English TTS as last resort if original language fails
            if language_code != "en":
                st.info("Attempting English TTS as fallback...")
                return create_audio(text, "en")
            
        return None

    def load_youtube_content(url: str) -> str:
        """Extract YouTube content as text using yt_dlp."""
        try:
            ydl_opts = {'format': 'bestaudio/best', 'quiet': True}
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info.get("title", "Video")
                description = info.get("description", "No description available.")
                return f"{title}\n\n{description}"
        except Exception as e:
            logger.error(f"YouTube extraction error: {str(e)}", exc_info=True)
            raise Exception(f"Error extracting YouTube content: {str(e)}")

    def extract_website_content(url: str) -> str:
        """Extract main content from a website using BeautifulSoup."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            # Use a session with timeout
            session = requests.Session()
            response = session.get(url, headers=headers, timeout=15, verify=True)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'iframe']):
                element.decompose()
            
            # Extract title
            title = soup.title.string if soup.title else ""
            
            # Extract main content
            main_content = []
            
            # Check for article or main content
            content_elements = soup.find_all(['article', 'main', 'div.content', 'div.post'])
            if content_elements:
                for element in content_elements:
                    main_content.append(element.get_text(strip=True, separator=' '))
            else:
                # Fallback to paragraphs if no main content container found
                paragraphs = soup.find_all('p')
                main_content = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50]
            
            return f"{title}\n\n{' '.join(main_content)}"
        
        except requests.exceptions.SSLError:
            # Try again without SSL verification as fallback
            logger.warning(f"SSL Error for {url}, retrying without verification")
            response = requests.get(url, headers=headers, verify=False, timeout=15)
            # Continue with the same processing as above
            soup = BeautifulSoup(response.text, 'html.parser')
            # Rest of the processing (same as above)
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'iframe']):
                element.decompose()
            title = soup.title.string if soup.title else ""
            main_content = []
            content_elements = soup.find_all(['article', 'main', 'div.content', 'div.post'])
            if content_elements:
                for element in content_elements:
                    main_content.append(element.get_text(strip=True, separator=' '))
            else:
                paragraphs = soup.find_all('p')
                main_content = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50]
            return f"{title}\n\n{' '.join(main_content)}"
            
        except Exception as e:
            logger.error(f"Website extraction error: {str(e)}", exc_info=True)
            raise Exception(f"Error extracting website content: {str(e)}")

    def get_content(url: str) -> list[Document]:
        """Load content from URL based on its type."""
        try:
            if is_youtube_url(url):
                text_content = load_youtube_content(url)
            else:
                text_content = extract_website_content(url)
            
            # Clean the text content
            text_content = re.sub(r'\s+', ' ', text_content).strip()
            text_content = re.sub(r'[^\w\s.,!?-]', '', text_content)
            
            return [Document(page_content=text_content)]
        
        except Exception as e:
            logger.error(f"Error in get_content: {str(e)}", exc_info=True)
            raise Exception(f"Error loading content: {str(e)}")

    def count_words(text: str) -> int:
        """Count words in text, handling multiple languages."""
        text = re.sub(r'http\S+|www.\S+', '', text)
        words = [word for word in text.split() if word.strip()]
        return len(words)

    def create_pdf(summary: str, url: str, language: str, length: str) -> bytes:
        """Create a PDF document containing the summary."""
        pdf = FPDF()
        pdf.add_page()
        
        # Add header
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Content Summary Report', ln=True, align='C')
        
        # Add metadata
        pdf.set_font('Arial', '', 12)
        pdf.line(10, 30, 200, 30)
        pdf.ln(15)
        
        # Add summary details
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Summary Details:', ln=True)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f'Source URL: {url}', ln=True)
        pdf.cell(0, 10, f'Language: {language}', ln=True)
        pdf.cell(0, 10, f'Summary Length: {length}', ln=True)
        pdf.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=True)
        
        # Add summary content
        pdf.ln(10)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Summary:', ln=True)
        pdf.set_font('Arial', '', 12)
        
        pdf.multi_cell(0, 10, summary)
        
        return pdf.output(dest='S').encode('latin1')

    def get_whatsapp_share_link(title, summary, source_url):
        """Generate WhatsApp sharing link."""
        # Create a share text
        whatsapp_text = f"📚 Summary of: {title}\n\n{summary}\n\nOriginal content: {source_url}"
        whatsapp_link = f"https://wa.me/?text={quote(whatsapp_text)}"
        
        return whatsapp_link

    def setup_whatsapp_sharing_ui(title, summary, url):
        """Set up the WhatsApp sharing UI section"""
        st.subheader("Share Summary:")
        
        # Generate WhatsApp sharing link
        whatsapp_link = get_whatsapp_share_link(
            title=title,
            summary=summary,
            source_url=url
        )
        
        
        st.markdown("### Share to WhatsApp")
        st.markdown("This will share the complete summary with the original link.")
        whatsapp_preview = f"""
        📱 WhatsApp Preview:
        
        📚 Summary of: {title}
        
        {summary[:150]}...
        
        Original content: {url}
        
        """
        st.markdown(whatsapp_preview)
        st.markdown(f"[![Share on WhatsApp](https://img.shields.io/badge/Share_on-WhatsApp-25D366?style=for-the-badge&logo=whatsapp&logoColor=white)]({whatsapp_link})")
        st.markdown("---")
            
        # Add copy functionality for complete summary
        st.subheader("Copy Full Summary")
        full_share_text = f"""📚 Summary of: {title}

    {summary}

    Original content: {url}

    Generated by Multi-Source Content Summarizer"""
        
        st.code(full_share_text, language="markdown")
        st.button("📋 Copy to Clipboard", 
                on_click=lambda: st.write("Summary copied to clipboard!"),
                help="Copy the full summary to paste manually into any platform")

    # Initialize session state for the chat

    # Summarization Process
    if st.button("Summarize"):
        if not url_input.strip():
            st.error("Please provide a URL to proceed.")
        elif not validators.url(url_input):
            st.error("Please enter a valid URL (YouTube or website).")
        else:
            try:

                if is_youtube_url(url_input):
                    display_youtube_video_info(url_input)
                word_count = SUMMARY_LENGTHS[selected_length]
                with st.spinner(f"Creating {selected_length.lower()} summary in {selected_language}..."):
                    docs = get_content(url_input)
                    
                    chain = load_summarize_chain(
                        llm, 
                        chain_type="stuff",
                        prompt=prompt.partial(
                            language=selected_language,
                            word_count=word_count
                        )
                    )
                    summary = chain.run(docs)
                    # Store summary and metadata in session state
                    st.session_state["summary"] = summary
                    st.session_state["url"] = url_input
                    st.session_state["summary_generated"] = True
                    st.session_state.selected_language = selected_language
                    st.session_state.selected_length = selected_length
                    
                    # Extract title for sharing
                    if is_youtube_url(url_input):
                        try:
                            with YoutubeDL({'quiet': True}) as ydl:
                                info = ydl.extract_info(url_input, download=False)
                                content_title = info.get("title", "YouTube Content")
                        except:
                            content_title = "YouTube Content"
                    else:
                        try:
                            response = requests.get(url_input, timeout=5)
                            soup = BeautifulSoup(response.text, 'html.parser')
                            content_title = soup.title.string if soup.title else "Web Content"
                        except:
                            content_title = "Web Content"
                    
                    st.session_state.content_title = content_title
                    #Generate audio for the summary
                    with st.spinner("Generating audio..."):
                            audio_file = create_audio(summary, LANGUAGES[selected_language])
                            if audio_file:
                                st.session_state.audio_file = audio_file           
                        # Create PDF bytes
                    pdf_bytes = create_pdf(
                            summary=summary,
                            url=url_input,
                            language=selected_language,
                            length=selected_length
                        )
                    st.session_state.pdf_bytes = pdf_bytes
            except Exception as e:
                st.exception(f"An error occurred: {e}")
# Always display summary if it exists in session state
    if st.session_state.summary_generated:
        st.subheader(f"Summary in {st.session_state.selected_language}:")
        st.success(st.session_state.summary)
        actual_word_count = count_words(st.session_state.summary)
        st.info(f"Word count: {actual_word_count} words")
    # Display audio player if audio file exists

    if st.session_state.audio_file:
        st.subheader("Listen to Summary:")
        st.audio(st.session_state.audio_file, format='audio/mp3')
    # Display PDF download button if PDF bytes exist
    if st.session_state.pdf_bytes:
        st.download_button(
            label="Download Summary as PDF",
            data=st.session_state.pdf_bytes,
            file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            key="pdf_download"
        )
    # Display sharing UI
        setup_whatsapp_sharing_ui(
        title=st.session_state.content_title,
        summary=st.session_state.summary,
        url=st.session_state.url,
        )
        current_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        add_semantic_mindmap_section(
        summary_text=st.session_state["summary"],
        dark_mode=st.session_state["dark_mode"],
        timestamp=current_timestamp
    )
    # Chat Interface
    st.markdown("<h3 class='chat-header'>Chat with AI</h3>", unsafe_allow_html=True)
    st.write(f"Ask any questions about the summary! (Responses will be in {st.session_state.selected_language})")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# User input for chatbot
    if user_input := st.chat_input("Ask a question..."):
    # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        context = f"""Based on this summary:\n\n{st.session_state.summary}\n\n
        Please provide the answer in {st.session_state.selected_language}.\n\n"""
        chat_prompt = f"{context}Answer this question: {user_input}"
        with st.spinner("Thinking..."):
            response = llm.invoke(chat_prompt).content
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
    st.sidebar.header("Features Available")
    st.sidebar.write("""
    - Summarize YouTube videos and websites
    - Video Information display
    - Multi-language support
    - Text-to-Speech capability
    - Customizable summary length
    - PDF download option
    - WhatsApp sharing
    - Full summary copying capability
    - Interactive chat interface
    - Dark/Light mode toggle with bulb icon
    """)
    st.sidebar.markdown("---")
    st.sidebar.write("Developed with ❤️ by BATCH E17")
