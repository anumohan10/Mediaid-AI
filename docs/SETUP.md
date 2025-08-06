# 🏥 MediAid AI - Setup Guide

A powerful medical AI assistant using RAG (Retrieval-Augmented Generation) with FAISS vector search and OpenAI integration.

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/anumohan10/Mediaid-AI.git
cd Mediaid-AI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up OpenAI API Key

**Option A: Environment Variable (Recommended)**
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-openai-api-key-here"

# Windows Command Prompt
set OPENAI_API_KEY=your-openai-api-key-here

# Linux/Mac
export OPENAI_API_KEY="your-openai-api-key-here"
```

**Option B: Create .env File**
```bash
# Copy the example file
copy .env.example .env

# Edit .env and add your API key
# OPENAI_API_KEY=your-openai-api-key-here
```

**Option C: Edit Config File**
Edit `config/openai_config.py` and add your API key directly (less secure).

### 4. Build FAISS Index (First Time Only)
```bash
python scripts/build_faiss_index.py
```

### 5. Test Your Setup
```bash
python test_rag_simple.py
```

### 6. Run the Application
```bash
# Web Interface (Recommended)
streamlit run streamlit_app.py

# Or Command Line Demo
python medical_rag_demo.py
```

## 🔑 Getting OpenAI API Key

1. Visit [OpenAI API](https://platform.openai.com/)
2. Sign up or log in to your account
3. Go to API Keys section
4. Create a new API key
5. Copy the key and follow setup steps above

**Note**: You'll need to add billing information to OpenAI for API usage.

## 📁 Project Structure

```
Mediaid-AI/
├── streamlit_app.py          # Main web interface
├── medical_rag_demo.py       # Command-line demo
├── test_rag_simple.py        # Setup validation
├── requirements.txt          # Python dependencies
├── .env.example             # Environment template
├── config/
│   ├── openai_config.py     # OpenAI configuration
│   └── pinecone_config.py   # Pinecone configuration
├── utils/
│   ├── rag.py               # RAG system core
│   ├── faiss_utils.py       # Vector search utilities
│   └── parser.py            # Text processing
├── scripts/
│   ├── build_faiss_index.py # Index building
│   └── extract_data.py      # Data extraction
└── data/                    # Medical data (CDC/WHO)
```

## 🧪 Testing

Run comprehensive tests:
```bash
# Test system setup
python test_rag_simple.py

# Test specific components
python test_faiss.py
python test_openai.py
```

## 🌐 Usage Examples

### Web Interface
1. Start the app: `streamlit run streamlit_app.py`
2. Open browser to `http://localhost:8501`
3. Click "🔍 Search" to access chat interface
4. Ask medical questions and get AI responses

### Command Line
```bash
python medical_rag_demo.py
```

### API Usage
```python
from utils.rag import MedicalRAG

# Initialize RAG system
rag = MedicalRAG("rag_data/medical_embeddings.index")

# Query the system
result = rag.query("What are symptoms of diabetes?")
print(result['response'])
```

## ⚠️ Important Notes

- **API Costs**: OpenAI API usage incurs costs. Monitor your usage.
- **Data Privacy**: Never commit `.env` files or API keys to version control.
- **System Requirements**: Requires Python 3.8+ and ~2GB RAM for FAISS index.

## 🔧 Troubleshooting

### Common Issues

**"No module named 'openai'"**
```bash
pip install openai
```

**"FAISS index not found"**
```bash
python scripts/build_faiss_index.py
```

**"Invalid API key"**
- Verify your OpenAI API key is correct
- Check if billing is set up on OpenAI account
- Ensure environment variable is set properly

**"Streamlit duplicate element error"**
- Restart the Streamlit app
- Clear browser cache

### Getting Help

1. Check the test script output: `python test_rag_simple.py`
2. Review error messages carefully
3. Ensure all dependencies are installed
4. Verify OpenAI API key is valid and has credit

## 🚀 Next Steps

After successful setup:
- Explore the web interface features
- Try different medical queries
- Customize the RAG system for specific use cases
- Add new data sources to the vector database

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Need help?** Open an issue on GitHub or check the troubleshooting section above.
