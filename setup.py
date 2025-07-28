#!/usr/bin/env python3
"""
Setup script for CLIP Image Moderation System
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.13+"""
    if sys.version_info < (3, 13):
        print("❌ Python 3.13+ is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def create_env_file():
    """Create .env file from template if it doesn't exist"""
    if os.path.exists('.env'):
        print("✅ .env file already exists")
        return True
    
    if os.path.exists('.env.example'):
        shutil.copy('.env.example', '.env')
        print("✅ Created .env file from template")
        print("⚠️  Please edit .env file with your Reddit API credentials")
        return True
    else:
        print("❌ .env.example not found")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'torch', 'transformers', 'streamlit', 'praw', 
        'PIL', 'numpy', 'pandas', 'requests'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_model_files():
    """Check if model files exist"""
    model_path = Path("clip-checkpoints/checkpoint-660/")
    if model_path.exists():
        print("✅ Model checkpoints found")
        return True
    else:
        print("⚠️  Model checkpoints not found")
        print("Run: python train.py")
        return False

def main():
    print("🔧 CLIP Image Moderation System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create environment file
    if not create_env_file():
        return False
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Check model files
    check_model_files()
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Edit .env file with your Reddit API credentials")
    print("2. Run: python train.py (if model not found)")
    print("3. Run: streamlit run streamlit_app.py")
    
    return True

if __name__ == "__main__":
    main() 