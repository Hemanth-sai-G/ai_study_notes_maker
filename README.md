# 🎓 Study Notes Maker

A powerful GenAI-powered web application that transforms your study materials (PDFs, audio, video) into comprehensive summaries, flashcards, and quizzes!

## ✨ Features

- 📄 **PDF Text Extraction** - Extract text from PDF documents
- 🎵 **Audio Transcription** - Convert audio files to text using Whisper
- 🎬 **Video Processing** - Extract and transcribe audio from video files
- 📝 **AI-Powered Summaries** - Generate concise, comprehensive summaries
- 🗂️ **Smart Flashcards** - Create Q&A flashcards for active recall
- 📋 **Interactive Quizzes** - Generate multiple-choice quizzes with explanations
- 💾 **Export Options** - Download as TXT or PDF formats
- 🎨 **Beautiful UI** - Modern, responsive interface built with Streamlit

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd study-notes-maker
pip install -r requirements.txt
```

### 2. Get OpenAI API Key
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Create an account and get your API key
3. Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_api_key_here
```

### 3. Run the App
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
study-notes-maker/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── config.py             # Configuration settings
├── utils/
│   ├── file_processors.py # File processing utilities
│   ├── ai_generators.py   # AI content generation
│   └── exporters.py       # Export functionality
├── assets/
│   └── styles.css        # Custom CSS styles
└── temp/                 # Temporary file storage
```

## 🎯 How to Use

1. **Upload File**: Choose a PDF, audio, or video file
2. **Configure Settings**: Select AI model and number of flashcards/quiz questions
3. **Generate**: Click "Generate Study Materials" and wait for processing
4. **Study**: Review summaries, flashcards, and take the quiz
5. **Export**: Download your study materials as TXT or PDF

## 📋 Supported File Types

- **PDFs**: `.pdf`
- **Audio**: `.mp3`, `.wav`, `.m4a`, `.flac`
- **Video**: `.mp4`, `.avi`, `.mov`, `.mkv`

## ⚙️ Configuration

Edit `config.py` to customize:
- OpenAI model settings
- File size limits
- UI themes and colors
- Generation parameters

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AI**: OpenAI GPT-3.5/GPT-4
- **Audio Processing**: OpenAI Whisper
- **PDF Processing**: PyMuPDF
- **Video Processing**: MoviePy
- **Export**: FPDF2

## 🚀 Deployment Options

### Option 1: Streamlit Cloud
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add your OpenAI API key in secrets

### Option 2: Hugging Face Spaces
1. Create a new Space on Hugging Face
2. Upload your files
3. Set your API key in Space settings

### Option 3: Local Network
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

## 🔧 Troubleshooting

### Common Issues

**"OpenAI API key not found"**
- Make sure your `.env` file contains `OPENAI_API_KEY=your_key`
- Check that python-dotenv is installed

**"Whisper model loading failed"**
- Ensure you have sufficient disk space (models are ~1GB)
- Check your internet connection for first-time download

**"PDF processing error"**
- Make sure the PDF is not password-protected
- Try with a different PDF file

**"Video processing slow"**
- Large video files take longer to process
- Consider using shorter clips for testing

## 📈 Performance Tips

- Use GPT-3.5-turbo for faster, cheaper generation
- Keep file sizes reasonable (<50MB for best performance)
- For long content, consider splitting into smaller chunks

## 🎨 Customization

### Themes
Edit `assets/styles.css` to customize colors, fonts, and layout

### AI Prompts
Modify prompts in `utils/ai_generators.py` to change output style

### Features
Add new file types by extending `utils/file_processors.py`

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 🆘 Support

If you encounter issues:
1. Check the troubleshooting section
2. Review your API key setup
3. Create an issue on GitHub

---

**Happy Studying! 🎓**