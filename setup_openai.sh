#!/bin/bash
# Cross-platform OpenAI API key setup for MediAid-AI (macOS/Linux)

echo "==============================================="
echo "    OPENAI API KEY SETUP FOR MEDIAID-AI"
echo "==============================================="
echo ""

# TODO: Replace 'your-api-key-here' with your actual API key
export OPENAI_API_KEY="your-api-key-here"

echo "✓ OpenAI API key set for this session."
echo ""
echo "💡 To make this permanent, add to your shell profile:"
echo "   echo 'export OPENAI_API_KEY=\"your-actual-key\"' >> ~/.bashrc"
echo "   # or ~/.zshrc for Zsh users"
echo ""
echo "🚀 Available commands after setting your key:"
echo "   python medical_rag_demo.py          - Full RAG demo"
echo "   python app.py                       - FastAPI web service"
echo "   python test_rag_simple.py           - Test without full setup"
echo ""
