# 🏥 MediAid AI

An intelligent medical assistant powered by RAG (Retrieval-Augmented Generation) technology, combining FAISS vector search with OpenAI's language models to provide accurate medical information.

## ✨ Features

- 🔍 **Intelligent Medical Search**: Query medical databases with natural language
- 🤖 **AI-Powered Responses**: Get comprehensive answers using OpenAI GPT models
- 📊 **Vector Search**: Fast similarity search through medical documents using FAISS
- 🌐 **Web Interface**: User-friendly Streamlit web application
- 📚 **Medical Database**: Curated medical data from CDC and WHO sources
- 💬 **Chat Interface**: Interactive conversation with medical AI assistant

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/anumohan10/Mediaid-AI.git
cd Mediaid-AI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up OpenAI API key
copy .env.example .env
# Edit .env and add your OpenAI API key

# 4. Build the medical database (first time only)
python scripts/build_faiss_index.py

# 5. Test the setup
python test_rag_simple.py

# 6. Run the web application
streamlit run streamlit_app.py
```

## 📋 Requirements

- Python 3.8+
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- ~2GB RAM for FAISS index
- Internet connection for OpenAI API calls

## 🔑 Setup OpenAI API Key

You need an OpenAI API key to use this application. Choose one method:

**Method 1: Environment Variable**
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key-here"

# Linux/Mac
export OPENAI_API_KEY="your-api-key-here"
```

**Method 2: .env File**
```bash
# Copy example and edit
copy .env.example .env
# Edit .env file and add your API key
```

## 🖥️ Usage

### Web Interface (Recommended)
```bash
streamlit run streamlit_app.py
```
- Open browser to `http://localhost:8501`
- Click "🔍 Search" to access chat interface
- Ask medical questions and get AI-powered responses

### Command Line Demo
```bash
python medical_rag_demo.py
```

### API Usage
```python
from utils.rag import MedicalRAG

rag = MedicalRAG("rag_data/medical_embeddings.index")
result = rag.query("What are symptoms of diabetes?")
print(result['response'])
```

## 📁 Project Structure

```
├── streamlit_app.py          # Main web interface
├── medical_rag_demo.py       # Command-line demo  
├── test_rag_simple.py        # Setup validation
├── requirements.txt          # Dependencies
├── .env.example             # Environment template
├── config/                  # Configuration files
├── utils/                   # Core utilities
├── scripts/                 # Setup and data scripts
├── data/                    # Medical data sources
└── rag_data/               # Generated vector database
```

## 🧪 Testing

Validate your setup:
```bash
python test_rag_simple.py
```

## 📖 Detailed Setup

For comprehensive setup instructions, see [SETUP.md](SETUP.md)

## ⚠️ Important Notes

- **API Costs**: OpenAI usage incurs costs - monitor your usage
- **Privacy**: Never commit API keys or `.env` files to version control  
- **Medical Disclaimer**: This is for informational purposes only - always consult healthcare professionals

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- 📚 [Setup Guide](SETUP.md)
- 🐛 [Report Issues](https://github.com/anumohan10/Mediaid-AI/issues)
- 💬 [Discussions](https://github.com/anumohan10/Mediaid-AI/discussions)

---

**⭐ If you find this project helpful, please give it a star!**