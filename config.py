'''import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# App Configuration
APP_TITLE = "🎓 Study Notes Maker"
APP_SUBTITLE = "Transform your study materials into smart notes, flashcards, and quizzes!"

# File Upload Settings
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB
ALLOWED_EXTENSIONS = {
    'pdf': ['.pdf'],
    'audio': ['.mp3', '.wav', '.m4a', '.flac'],
    'video': ['.mp4', '.avi', '.mov', '.mkv']
}

# AI Generation Settings
DEFAULT_MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 2000
TEMPERATURE = 0.7

# UI Configuration
THEME_CONFIG = {
    "primary_color": "#FF6B6B",
    "background_color": "#FFFFFF",
    "secondary_background_color": "#F0F2F6",
    "text_color": "#262730"
}'''

# config.py
APP_TITLE = "Study Notes Maker"
APP_SUBTITLE = "Transform your study materials into smart notes!"
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB
ALLOWED_EXTENSIONS = ['pdf', 'mp3', 'wav', 'm4a', 'flac', 'mp4', 'avi', 'mov', 'mkv']