import os
from typing import Optional

# OpenAI Configuration for MediAid-AI

def get_openai_api_key() -> Optional[str]:
    """
    Get OpenAI API key from multiple sources in order of preference:
    1. Environment variable OPENAI_API_KEY
    2. Hardcoded value below (for development only)
    3. Return None if not found
    """
    
    # Method 1: Environment variable (recommended)
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        return api_key
    
    # Method 2: Hardcode here (NOT recommended for production)
    # Uncomment and replace with your actual key for quick testing:
    # HARDCODED_API_KEY = "your-api-key-here"
    # return HARDCODED_API_KEY
    
    # Method 3: Not found
    return None

def validate_openai_setup() -> bool:
    """Check if OpenAI API key is properly configured."""
    api_key = get_openai_api_key()
    
    if not api_key:
        print("❌ OpenAI API key not found!")
        print("\n💡 How to set it up:")
        print("   Option 1 (Recommended): Set environment variable")
        print("     Windows: $env:OPENAI_API_KEY='your-key-here'")
        print("     Or run: setup_openai.bat")
        print()
        print("   Option 2: Edit config/openai_config.py")
        print("     Uncomment and set HARDCODED_API_KEY")
        print()
        return False
    
    if api_key == "your-api-key-here":
        print("❌ Please replace 'your-api-key-here' with your actual OpenAI API key")
        return False
    
    print("✅ OpenAI API key configured successfully!")
    return True

# OpenAI Model Settings
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 1000
TEMPERATURE = 0.1

if __name__ == "__main__":
    # Test the configuration
    print("🔧 Testing OpenAI Configuration...")
    validate_openai_setup()