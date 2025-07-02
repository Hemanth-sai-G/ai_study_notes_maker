# AI Study Notes Maker 📚🤖

An intelligent web application that transforms various file formats into comprehensive study notes using advanced AI and Natural Language Processing technologies. Built with Streamlit and powered by transformer models.

## 🌟 Features

### 📄 Multi-Format File Support
- **PDF Documents** - Extract and process PDF content
- **Video Files** - Extract audio and convert to text
- **Audio Files** - Direct audio-to-text conversion
- **Text Files** - Process plain text documents

### 🧠 AI-Powered Content Generation
- **Intelligent Summarization** - Create concise summaries from lengthy documents
- **Quiz Generation** - Automatically generate questions and answers
- **Flashcard Creation** - Generate interactive flashcards for quick revision
- **Study Notes** - Transform content into structured study materials

### 🎯 Advanced Processing
- **NLP Integration** - Uses NLTK and spaCy for advanced text processing
- **Transformer Models** - Leverages state-of-the-art AI models for content understanding
- **Multi-modal Processing** - Handles text, audio, and video inputs
- **Custom File Processing** - Modular architecture for different file types

## 🚀 Technology Stack

- **Frontend**: Streamlit
- **AI/ML**: Transformers, PyTorch, NLTK, spaCy
- **File Processing**: PyMuPDF, python-docx, MoviePy, PyDub
- **Document Generation**: ReportLab
- **Backend**: Python 3.8+

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Internet connection (for downloading AI models)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-study-notes-maker.git
   cd ai-study-notes-maker
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download required language models**
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Download NLTK data** (will be prompted during first run)
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

## 🚀 Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Upload your files** using the file uploader interface

4. **Select processing options**:
   - Choose summary length
   - Select quiz difficulty
   - Configure flashcard settings

5. **Generate study materials** and download your personalized notes

## 📁 Project Structure

```
ai-study-notes-maker/
│
├── app.py 
|__config.py                # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
│
├── utils/                # Utility modules
│   ├── file_processor.py # File processing functions
│   ├── ai_generator.py   # AI content generation
│   └── pdf_exporter.py   # PDF export functionality
|   |__ __init__.py
│
├── config.py             # Configuration settings
└── .gitignore            # Git ignore rules
```

## 🎯 Key Components

### File Processor
- Handles multiple file formats
- Extracts text from various sources
- Preprocesses content for AI processing

### AI Generator
- Utilizes transformer models for content generation
- Creates summaries, quizzes, and flashcards
- Implements advanced NLP techniques

### PDF Exporter
- Generates professional study material PDFs
- Customizable formatting and styling
- Supports various document layouts

## 📊 Supported File Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| PDF | `.pdf` | Portable Document Format |
| Word | `.docx` | Microsoft Word Document |
| Video | `.mp4`, `.avi`, `.mov` | Video files (audio extraction) |
| Audio | `.mp3`, `.wav`, `.m4a` | Audio files |
| Text | `.txt` | Plain text files |

## 🔧 Configuration

The application can be configured through the `config.py` file:

- **APP_TITLE**: Application title
- **APP_SUBTITLE**: Application subtitle
- **MAX_FILE_SIZE**: Maximum file upload size
- **ALLOWED_EXTENSIONS**: Supported file extensions

## 🎨 Screenshots

*Add screenshots of your application here*

1. **Main Interface** - File upload and processing options
2. **Study Notes Generation** - AI-generated content display
3. **Quiz Interface** - Interactive quiz functionality
4. **PDF Export** - Generated study materials

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face Transformers** for providing state-of-the-art NLP models
- **Streamlit** for the amazing web app framework
- **spaCy** and **NLTK** for natural language processing capabilities
- **ReportLab** for PDF generation functionality

## 📞 Contact

**Your Name** - your.email@example.com

Project Link: [https://github.com/yourusername/ai-study-notes-maker](https://github.com/yourusername/ai-study-notes-maker)

## 🔮 Future Enhancements

- [ ] Support for more file formats
- [ ] Multi-language support
- [ ] Cloud storage integration
- [ ] Mobile-responsive design
- [ ] User authentication system
- [ ] Advanced quiz customization
- [ ] Export to multiple formats

---

⭐ **Star this repository if you found it helpful!** ⭐