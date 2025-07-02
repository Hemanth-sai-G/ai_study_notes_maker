import streamlit as st
import os
import time
from datetime import datetime
import json

# Import our custom modules
from utils.file_processor import FileProcessor
from utils.ai_generator import AIGenerator
from utils.pdf_exporter import PDFExporter
from config import APP_TITLE, APP_SUBTITLE, MAX_FILE_SIZE, ALLOWED_EXTENSIONS

# Page configuration
st.set_page_config(
    page_title="Study Notes Maker",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    try:
        with open("assets/styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # CSS file is optional

load_css()

# Initialize session state
def init_session_state():
    if 'processed_text' not in st.session_state:
        st.session_state.processed_text = None
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    if 'flashcards' not in st.session_state:
        st.session_state.flashcards = None
    if 'quiz' not in st.session_state:
        st.session_state.quiz = None
    if 'processing_step' not in st.session_state:
        st.session_state.processing_step = 0

init_session_state()

# Initialize processors
@st.cache_resource
def get_processors():
    return FileProcessor(), AIGenerator(), PDFExporter()

file_processor, ai_generator, exporter = get_processors()

# Header
def render_header():
    st.markdown("""
    <div class="header-container">
        <div class="header-title">🎓 Study Notes Maker</div>
        <div class="header-subtitle">Transform your study materials into smart notes, flashcards, and quizzes!</div>
    </div>
    """, unsafe_allow_html=True)

# Sidebar
def render_sidebar():
    with st.sidebar:
        st.markdown("## 📚 Features")
        
        features = [
            "📄 PDF Text Extraction",
            "🎵 Audio Transcription", 
            "🎬 Video Processing",
            "📝 AI-Powered Summaries",
            "🗂️ Smart Flashcards",
            "📋 Interactive Quizzes",
            "💾 Export Options"
        ]
        
        for feature in features:
            st.markdown(f"- {feature}")
        
        st.markdown("---")
        st.markdown("## 🔧 Settings")
        
        # Model selection
        model_choice = st.selectbox(
            "AI Model",
            ["gpt-3.5-turbo", "gpt-4"],
            help="Choose the AI model for content generation"
        )
        
        # Number of flashcards
        num_flashcards = st.slider(
            "Number of Flashcards",
            min_value=5,
            max_value=20,
            value=10,
            help="How many flashcards to generate"
        )
        
        # Number of quiz questions
        num_quiz = st.slider(
            "Quiz Questions",
            min_value=3,
            max_value=15,
            value=5,
            help="How many quiz questions to generate"
        )
        
        return model_choice, num_flashcards, num_quiz

# File upload section
def render_file_upload():
    st.markdown("## 📁 Upload Your Study Material")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'mp3', 'wav', 'm4a', 'flac', 'mp4', 'avi', 'mov', 'mkv'],
            help="Upload PDF, audio, or video files"
        )
    
    with col2:
        if uploaded_file:
            file_details = {
                "filename": uploaded_file.name,
                "filetype": uploaded_file.type,
                "filesize": uploaded_file.size
            }
            
            st.json(file_details)
    
    return uploaded_file

# Processing section
def render_processing(uploaded_file, model_choice, num_flashcards, num_quiz):
    if uploaded_file is None:
        st.info("👆 Please upload a file to get started!")
        return
    
    if st.button("🚀 Generate Study Materials", type="primary"):
        with st.spinner("Processing your file..."):
            # Step 1: Extract text
            st.session_state.processing_step = 1
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("📄 Extracting text from file...")
            progress_bar.progress(20)
            
            text = file_processor.process_file(uploaded_file)
            
            if text:
                st.session_state.processed_text = text
                progress_bar.progress(40)
                
                # Step 2: Generate summary
                status_text.text("📝 Generating summary...")
                summary = ai_generator.generate_summary(text)
                if summary:
                    st.session_state.summary = summary
                progress_bar.progress(60)
                
                # Step 3: Generate flashcards
                status_text.text("🗂️ Creating flashcards...")
                flashcards = ai_generator.generate_flashcards(text, num_flashcards)
                if flashcards:
                    st.session_state.flashcards = flashcards
                progress_bar.progress(80)
                
                # Step 4: Generate quiz
                status_text.text("📋 Preparing quiz...")
                quiz = ai_generator.generate_quiz(text, num_quiz)
                if quiz:
                    st.session_state.quiz = quiz
                progress_bar.progress(100)
                
                status_text.text("✅ All done! Check out your study materials below.")
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                
                st.success("🎉 Study materials generated successfully!")
                st.balloons()
            else:
                st.error("❌ Failed to process the file. Please try again.")

# Results display section
def render_results():
    if not any([st.session_state.summary, st.session_state.flashcards, st.session_state.quiz]):
        return
    
    st.markdown("## 📊 Your Study Materials")
    
    # Create tabs for different content types
    tab1, tab2, tab3 = st.tabs(["📝 Summary", "🗂️ Flashcards", "📋 Quiz"])
    
    with tab1:
        render_summary()
    
    with tab2:
        render_flashcards()
    
    with tab3:
        render_quiz()
    
    # Export section
    render_export_section()

def render_summary():
    if st.session_state.summary:
        st.markdown("### 📝 Summary")
        st.markdown(f"""
        <div class="custom-card">
            {st.session_state.summary.replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No summary available. Generate study materials first!")

def render_flashcards():
    if st.session_state.flashcards:
        st.markdown("### 🗂️ Flashcards")
        
        # Add study mode toggle
        study_mode = st.toggle("Study Mode (Click to reveal answers)")
        
        for i, card in enumerate(st.session_state.flashcards):
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    question = card.get('question', 'No question available')
                    answer = card.get('answer', 'No answer available')
                    
                    if study_mode:
                        with st.expander(f"📌 Flashcard {i+1}: {question[:50]}..."):
                            st.markdown(f"**Question:** {question}")
                            st.markdown(f"**Answer:** {answer}")
                    else:
                        st.markdown(f"""
                        <div class="flashcard">
                            <div class="flashcard-question">Q{i+1}: {question}</div>
                            <div class="flashcard-answer">A: {answer}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"Card {i+1}")
    else:
        st.info("No flashcards available. Generate study materials first!")

def render_quiz():
    if st.session_state.quiz:
        st.markdown("### 📋 Interactive Quiz")
        
        # Initialize quiz state
        if 'quiz_answers' not in st.session_state:
            st.session_state.quiz_answers = {}
        if 'quiz_submitted' not in st.session_state:
            st.session_state.quiz_submitted = False
        
        # Quiz form
        with st.form("quiz_form"):
            for i, question in enumerate(st.session_state.quiz):
                st.markdown(f"**Question {i+1}:** {question.get('question', 'No question')}")
                
                options = question.get('options', [])
                answer_key = f"q_{i}"
                
                selected = st.radio(
                    f"Select your answer for Question {i+1}:",
                    options,
                    key=answer_key,
                    index=None
                )
                
                if selected:
                    st.session_state.quiz_answers[i] = selected
                
                st.markdown("---")
            
            submitted = st.form_submit_button("Submit Quiz", type="primary")
            
            if submitted:
                st.session_state.quiz_submitted = True
        
        # Show results if submitted
        if st.session_state.quiz_submitted:
            render_quiz_results()
    else:
        st.info("No quiz available. Generate study materials first!")

def render_quiz_results():
    st.markdown("### 🎯 Quiz Results")
    
    correct_answers = 0
    total_questions = len(st.session_state.quiz)
    
    for i, question in enumerate(st.session_state.quiz):
        user_answer = st.session_state.quiz_answers.get(i)
        correct_answer = question.get('correct_answer', 'A')
        correct_option = question.get('options', [])[ord(correct_answer) - ord('A')] if correct_answer else "Unknown"
        
        is_correct = user_answer == correct_option
        if is_correct:
            correct_answers += 1
        
        # Display question result
        status_icon = "✅" if is_correct else "❌"
        status_color = "correct-answer" if is_correct else "wrong-answer"
        
        st.markdown(f"""
        <div class="quiz-question {status_color}">
            <strong>{status_icon} Question {i+1}:</strong> {question.get('question', 'No question')}<br>
            <strong>Your Answer:</strong> {user_answer or 'Not answered'}<br>
            <strong>Correct Answer:</strong> {correct_option}<br>
            <strong>Explanation:</strong> {question.get('explanation', 'No explanation available')}
        </div>
        """, unsafe_allow_html=True)
    
    # Overall score
    score_percentage = (correct_answers / total_questions) * 100
    st.markdown(f"""
    <div class="metric-container">
        <h3>Final Score: {correct_answers}/{total_questions} ({score_percentage:.1f}%)</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance message
    if score_percentage >= 80:
        st.success("🎉 Excellent work! You've mastered this material!")
    elif score_percentage >= 60:
        st.warning("👍 Good job! Review the missed questions and try again.")
    else:
        st.error("📚 Keep studying! Review the material and retake the quiz.")

def render_export_section():
    st.markdown("## 💾 Export Your Study Materials")
    
    col1, col2, col3 = st.columns(3)
    
    # Text export
    with col1:
        if st.button("📄 Download as Text", type="secondary"):
            text_content = exporter.create_text_export(
                st.session_state.summary,
                st.session_state.flashcards,
                st.session_state.quiz
            )
            
            st.download_button(
                label="📥 Download TXT",
                data=text_content,
                file_name=f"study_notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    # PDF export
    with col2:
        if st.button("📑 Download as PDF", type="secondary"):
            pdf_content = exporter.create_pdf_export(
                st.session_state.summary,
                st.session_state.flashcards,
                st.session_state.quiz
            )
            
            if pdf_content:
                st.download_button(
                    label="📥 Download PDF",
                    data=pdf_content,
                    file_name=f"study_notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
    
    # Flashcard PDF export
    with col3:
        if st.session_state.flashcards and st.button("🗂️ Flashcard PDF", type="secondary"):
            flashcard_pdf = exporter.create_flashcard_pdf(st.session_state.flashcards)
            
            if flashcard_pdf:
                st.download_button(
                    label="📥 Download Flashcards",
                    data=flashcard_pdf,
                    file_name=f"flashcards_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )

# Main app
def main():
    render_header()
    
    # Get sidebar settings
    model_choice, num_flashcards, num_quiz = render_sidebar()
    
    # Main content
    uploaded_file = render_file_upload()
    
    st.markdown("---")
    
    render_processing(uploaded_file, model_choice, num_flashcards, num_quiz)
    
    st.markdown("---")
    
    render_results()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>Made with ❤️ using Streamlit and OpenAI</p>
        <p>🎓 Happy Studying! 🎓</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()