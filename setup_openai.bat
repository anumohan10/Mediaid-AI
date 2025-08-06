@echo off
REM Set your OpenAI API key for the current session
REM Replace 'your-api-key-here' with your actual OpenAI API key

@echo off
REM Set your OpenAI API key for the current session
REM Replace 'your-api-key-here' with your actual OpenAI API key

echo ===============================================
echo    OPENAI API KEY SETUP FOR MEDIAID-AI
echo ===============================================
echo.

REM TODO: Replace 'your-api-key-here' with your actual API key
set OPENAI_API_KEY=your-api-key-here

echo ✓ OpenAI API key set for this session.
echo.
echo 💡 To make this permanent, you can:
echo    1. Edit this file and replace 'your-api-key-here'
echo    2. Or add to Windows Environment Variables:
echo       OPENAI_API_KEY=your-actual-key
echo.
echo 🚀 Available commands after setting your key:
echo    python medical_rag_demo.py          - Full RAG demo
echo    python app.py                       - FastAPI web service  
echo    python test_rag_simple.py           - Test without full setup
echo.

pause
