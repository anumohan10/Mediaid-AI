#!/usr/bin/env python3
"""
Cross-platform utilities for MediAid-AI
Handles platform-specific configurations and paths
"""

import os
import platform
from typing import Dict, List, Optional

def get_platform_info() -> Dict[str, str]:
    """Get current platform information."""
    system = platform.system().lower()
    return {
        'system': system,
        'is_windows': system == 'windows',
        'is_macos': system == 'darwin',
        'is_linux': system == 'linux',
        'platform_name': {
            'windows': 'Windows',
            'darwin': 'macOS', 
            'linux': 'Linux'
        }.get(system, 'Unknown')
    }

def get_setup_script_name() -> str:
    """Get the appropriate setup script for current platform."""
    if get_platform_info()['is_windows']:
        return 'setup_openai.bat'
    else:
        return 'setup_openai.sh'

def get_env_set_command(var_name: str, var_value: str) -> str:
    """Get platform-specific environment variable set command."""
    if get_platform_info()['is_windows']:
        return f'$env:{var_name}="{var_value}"'  # PowerShell
    else:
        return f'export {var_name}="{var_value}"'  # Bash/Zsh

def normalize_path(path: str) -> str:
    """Normalize path for current platform."""
    return os.path.normpath(path)

def get_tesseract_default_paths() -> List[str]:
    """Get default Tesseract installation paths for current platform."""
    platform_info = get_platform_info()
    
    if platform_info['is_windows']:
        return [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        ]
    elif platform_info['is_macos']:
        return [
            "/opt/homebrew/bin/tesseract",  # Apple Silicon
            "/usr/local/bin/tesseract",     # Intel Mac
            "/opt/local/bin/tesseract"      # MacPorts
        ]
    else:  # Linux
        return [
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract"
        ]

if __name__ == "__main__":
    # Demo
    info = get_platform_info()
    print(f"Platform: {info['platform_name']}")
    print(f"Setup script: {get_setup_script_name()}")
    print(f"Tesseract paths: {get_tesseract_default_paths()}")
