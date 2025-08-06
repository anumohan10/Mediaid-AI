# 🏥 MediAid AI - Project Structure

This document outlines the complete project structure and organization of the MediAid AI medical document analysis system.

## 📁 Project Organization

```
📁 Mediaid-AI-4/
├── 🚀 Main Applications
│   ├── streamlit_app.py          # Main Streamlit web app with chat and document upload
│   └── app.py                    # FastAPI web service for API access
│
├── 📊 Data & Models  
│   ├── rag_data/                 # FAISS index & embeddings (5,489 medical documents)
│   │   ├── medical_embeddings.index
│   │   ├── medical_embeddings_metadata.json
│   │   ├── cdc_chunks.json
│   │   └── who_chunks.json
│   ├── data/                     # Raw medical data
│   │   ├── cdc_data.json
│   │   ├── who_data.json
│   │   ├── cdc_urls.json
│   │   └── who_urls.json
│   └── cleaned/                  # Processed data
│       ├── cdc_data_cleaned.json
│       └── who_data_cleaned.json
│
├── ⚙️ Core Systems
│   ├── config/                   # Configuration files
│   │   ├── openai_config.py
│   │   └── pinecone_config.py
│   ├── utils/                    # Utilities (OCR, RAG, etc.)
│   │   ├── rag.py               # RAG system implementation
│   │   ├── faiss_utils.py       # FAISS vector store utilities
│   │   ├── ocr_utils.py         # OCR processing for documents
│   │   ├── parser.py            # Data parsing utilities
│   │   ├── predict.py           # ML prediction utilities
│   │   └── explain.py           # AI explanation utilities
│   └── scripts/                  # Data processing scripts
│       ├── build_faiss_index.py
│       ├── chunk_texts.py
│       ├── embed_chunks.py
│       ├── extract_data.py
│       └── various cleaning scripts
│
├── 🧪 Testing & Demo
│   ├── tests/                   # Test files
│   │   ├── test_rag_simple.py   # Main system test
│   │   └── test_rag_system.py   # Comprehensive test
│   ├── demos/                   # Demo applications
│   │   ├── medical_rag_demo.py  # Interactive user demo
│   │   ├── demo_search.py       # Search examples
│   │   ├── simple_rag_demo.py   # Simple RAG demonstration
│   │   └── medical_search.py    # Medical search demo
│   └── examples/                # Code examples
│       └── faiss_example.py     # FAISS usage examples
│
├── 📚 Documentation
│   ├── README.md                # Main project documentation
│   ├── docs/
│   │   └── SETUP.md            # Detailed setup instructions
│   └── LICENSE                  # Project license
│
└── 🔧 Configuration
    ├── requirements.txt          # Python dependencies
    ├── setup.py                  # Package setup configuration
    ├── .gitignore               # Git ignore rules
    ├── .env.example             # Environment variables template
    └── setup_openai.bat         # Windows setup script
```

## 🚀 Quick Start

1. **Setup**: Follow `docs/SETUP.md` for detailed setup instructions
2. **Test**: Run `python tests/test_rag_simple.py` to verify setup
3. **Demo**: Run `python demos/medical_rag_demo.py` for interactive demo
4. **Web App**: Run `streamlit run streamlit_app.py` for full web interface

## 📋 Key Features

- **🤖 AI-Powered Chat**: Conversational medical information with context
- **📄 Document Upload**: OCR processing of medical documents (prescriptions, lab reports)
- **🔍 Smart Search**: Vector search through 5,489 medical documents
- **🩺 Medical Analysis**: Extract medications, diagnoses, lab values from uploads
- **📱 Web Interface**: User-friendly Streamlit web application
- **🔌 API Access**: FastAPI service for programmatic access

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **AI**: OpenAI GPT models
- **Vector DB**: FAISS
- **OCR**: EasyOCR + Tesseract
- **Data**: CDC & WHO medical databases
- **Language**: Python 3.13+

## 📞 Support

For setup issues or questions, refer to:
- `docs/SETUP.md` - Comprehensive setup guide
- `tests/test_rag_simple.py` - System verification
- `demos/` - Working examples
