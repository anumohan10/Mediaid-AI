#!/usr/bin/env python3
"""
MediAid AI Setup Script
Helps new users set up the project quickly.
"""

import os
import sys
import subprocess
import shutil

def print_header():
    print("🏥 MediAid AI - Setup Script")
    print("=" * 50)
    print()

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_requirements():
    """Install required Python packages."""
    print("\n📦 Installing Python packages...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print("✅ Python packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        print("   Try running: pip install -r requirements.txt")
        return False

def setup_env_file():
    """Create .env file from example."""
    print("\n🔑 Setting up environment configuration...")
    
    if os.path.exists('.env'):
        print("⚠️  .env file already exists")
        return True
    
    if os.path.exists('.env.example'):
        try:
            shutil.copy('.env.example', '.env')
            print("✅ Created .env file from template")
            print("   📝 Please edit .env and add your OpenAI API key")
            return True
        except Exception as e:
            print(f"❌ Error creating .env file: {e}")
            return False
    else:
        print("❌ .env.example file not found")
        return False

def check_openai_key():
    """Check if OpenAI API key is configured."""
    print("\n🔍 Checking OpenAI API key...")
    
    # Check environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    
    # Check .env file
    if not api_key and os.path.exists('.env'):
        try:
            with open('.env', 'r') as f:
                for line in f:
                    if line.startswith('OPENAI_API_KEY='):
                        api_key = line.split('=', 1)[1].strip()
                        break
        except Exception:
            pass
    
    if api_key and api_key != 'your-openai-api-key-here':
        print("✅ OpenAI API key found")
        return True
    else:
        print("⚠️  OpenAI API key not configured")
        print("   Get your API key from: https://platform.openai.com/api-keys")
        print("   Then either:")
        print("   • Set environment variable: OPENAI_API_KEY=your-key")
        print("   • Edit .env file and add your key")
        return False

def build_faiss_index():
    """Build FAISS index if it doesn't exist."""
    print("\n🗂️  Checking FAISS index...")
    
    index_path = "rag_data/medical_embeddings.index"
    if os.path.exists(index_path):
        print("✅ FAISS index already exists")
        return True
    
    print("🔧 Building FAISS index (this may take a few minutes)...")
    try:
        subprocess.run([sys.executable, "scripts/build_faiss_index.py"], 
                      check=True, capture_output=True, text=True)
        print("✅ FAISS index built successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error building FAISS index: {e}")
        print("   Try running manually: python scripts/build_faiss_index.py")
        return False

def test_setup():
    """Test the complete setup."""
    print("\n🧪 Testing setup...")
    
    try:
        subprocess.run([sys.executable, "test_rag_simple.py"], 
                      check=True, capture_output=True, text=True)
        print("✅ Setup test passed!")
        return True
    except subprocess.CalledProcessError as e:
        print("⚠️  Setup test failed")
        print("   Run manually for details: python test_rag_simple.py")
        return False

def show_next_steps():
    """Show what to do next."""
    print("\n🚀 Next Steps:")
    print("   1. Ensure OpenAI API key is configured")
    print("   2. Test your setup: python test_rag_simple.py")
    print("   3. Run the web app: streamlit run streamlit_app.py")
    print("   4. Or try the demo: python medical_rag_demo.py")
    print()
    print("📚 For detailed instructions, see: SETUP.md")
    print("🆘 Need help? Check: https://github.com/anumohan10/Mediaid-AI")

def main():
    """Main setup function."""
    print_header()
    
    # Step 1: Check Python version
    if not check_python_version():
        return
    
    # Step 2: Install requirements
    if not install_requirements():
        print("\n❌ Setup failed at package installation")
        return
    
    # Step 3: Setup .env file
    setup_env_file()
    
    # Step 4: Check OpenAI key
    has_key = check_openai_key()
    
    # Step 5: Build FAISS index (if we have API key)
    if has_key:
        build_faiss_index()
        
        # Step 6: Test setup
        test_setup()
    
    # Show next steps
    show_next_steps()
    
    if has_key:
        print("\n🎉 Setup complete! Your MediAid AI is ready to use.")
    else:
        print("\n⚠️  Setup incomplete: Please configure your OpenAI API key.")

if __name__ == "__main__":
    main()
