'''import os
import tempfile
import fitz  # PyMuPDF
import whisper
import streamlit as st
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import io

class FileProcessor:
    def __init__(self):
        self.whisper_model = None
    
    def load_whisper_model(self):
        """Load Whisper model with caching"""
        if self.whisper_model is None:
            with st.spinner("Loading Whisper model (this may take a moment)..."):
                self.whisper_model = whisper.load_model("base")
        return self.whisper_model
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF file"""
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Extract text using PyMuPDF
            doc = fitz.open(tmp_file_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            
            doc.close()
            os.unlink(tmp_file_path)  # Clean up temp file
            
            return text.strip()
        
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return None
    
    def transcribe_audio(self, audio_file):
        """Transcribe audio file using Whisper"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Load Whisper model
            model = self.load_whisper_model()
            
            # Transcribe
            with st.spinner("Transcribing audio... This may take a few minutes."):
                result = model.transcribe(tmp_file_path)
            
            os.unlink(tmp_file_path)  # Clean up
            return result["text"]
        
        except Exception as e:
            st.error(f"Error transcribing audio: {str(e)}")
            return None
    
    def extract_audio_from_video(self, video_file):
        """Extract audio from video file and transcribe"""
        try:
            # Create temporary files
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
                tmp_video.write(video_file.getvalue())
                tmp_video_path = tmp_video.name
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                tmp_audio_path = tmp_audio.name
            
            # Extract audio from video
            with st.spinner("Extracting audio from video..."):
                video = VideoFileClip(tmp_video_path)
                audio = video.audio
                audio.write_audiofile(tmp_audio_path, verbose=False, logger=None)
                video.close()
                audio.close()
            
            # Create audio file-like object for transcription
            with open(tmp_audio_path, 'rb') as audio_file:
                audio_bytes = io.BytesIO(audio_file.read())
                audio_bytes.name = "extracted_audio.wav"
            
            # Transcribe the extracted audio
            text = self.transcribe_audio(audio_bytes)
            
            # Clean up temp files
            os.unlink(tmp_video_path)
            os.unlink(tmp_audio_path)
            
            return text
        
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            return None
    
    def get_file_type(self, file):
        """Determine file type based on extension"""
        file_extension = os.path.splitext(file.name)[1].lower()
        
        if file_extension in ['.pdf']:
            return 'pdf'
        elif file_extension in ['.mp3', '.wav', '.m4a', '.flac']:
            return 'audio'
        elif file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
            return 'video'
        else:
            return 'unknown'
    
    def process_file(self, uploaded_file):
        """Process uploaded file and return extracted text"""
        if uploaded_file is None:
            return None
        
        file_type = self.get_file_type(uploaded_file)
        
        if file_type == 'pdf':
            return self.extract_text_from_pdf(uploaded_file)
        elif file_type == 'audio':
            return self.transcribe_audio(uploaded_file)
        elif file_type == 'video':
            return self.extract_audio_from_video(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload PDF, audio, or video files.")
            return None'''

# utils/file_processors.py
import streamlit as st
import tempfile
import os
import fitz  # PyMuPDF
import whisper
from moviepy.editor import VideoFileClip
from pydub import AudioSegment

class FileProcessor:
    def __init__(self):
        self.whisper_model = None
    
    def load_whisper_model(self):
        if self.whisper_model is None:
            self.whisper_model = whisper.load_model("base")
        return self.whisper_model
    
    def process_file(self, uploaded_file):
        """Process uploaded file and extract text"""
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        try:
            if file_extension == 'pdf':
                return self.extract_text_from_pdf(uploaded_file)
            elif file_extension in ['mp3', 'wav', 'm4a', 'flac']:
                return self.transcribe_audio(uploaded_file)
            elif file_extension in ['mp4', 'avi', 'mov', 'mkv']:
                return self.transcribe_video(uploaded_file)
            else:
                st.error(f"Unsupported file type: {file_extension}")
                return None
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return None
    
    def extract_text_from_pdf(self, uploaded_file):
        """Extract text from PDF file"""
        text = ""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        try:
            doc = fitz.open(tmp_file_path)
            for page in doc:
                text += page.get_text()
            doc.close()
        finally:
            os.unlink(tmp_file_path)
        
        return text.strip()
    
    def transcribe_audio(self, uploaded_file):
        """Transcribe audio file to text"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        try:
            model = self.load_whisper_model()
            result = model.transcribe(tmp_file_path)
            return result["text"]
        finally:
            os.unlink(tmp_file_path)
    
    def transcribe_video(self, uploaded_file):
        """Extract audio from video and transcribe"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        audio_path = tmp_file_path.replace(tmp_file_path.split('.')[-1], 'wav')
        
        try:
            # Extract audio from video
            video = VideoFileClip(tmp_file_path)
            video.audio.write_audiofile(audio_path, verbose=False, logger=None)
            video.close()
            
            # Transcribe audio
            model = self.load_whisper_model()
            result = model.transcribe(audio_path)
            return result["text"]
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            if os.path.exists(audio_path):
                os.unlink(audio_path)