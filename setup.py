#!/usr/bin/env python3
"""
Setup script for SpeakServe AI project.
This script helps users configure their environment and get the project ready to run.
"""

import os
import sys
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        "fastapi",
        "uvicorn",
        "openai",
        "whisper",
        "numpy",
        "python-dotenv"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install dependencies with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def setup_environment():
    """Set up environment variables"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_example.exists():
        print("❌ Error: .env.example file not found")
        return False
    
    if env_file.exists():
        print("📝 .env file already exists")
        response = input("   Do you want to overwrite it? (y/N): ").strip().lower()
        if response != 'y':
            print("   Keeping existing .env file")
            return True
    
    # Copy .env.example to .env
    try:
        shutil.copy(env_example, env_file)
        print("✅ Created .env file from .env.example")
        print("📝 Please edit .env file and add your API keys:")
        print("   - OPENAI_API_KEY: Your OpenAI API key")
        print("   - Other variables as needed")
        return True
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False

def validate_environment():
    """Validate that environment variables are set correctly"""
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        return False
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n📝 Please edit .env file and add these variables")
        return False
    
    print("✅ Environment variables are configured")
    return True

def create_directories():
    """Create necessary directories"""
    directories = ["logs", "audio", "temp"]
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(exist_ok=True)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"📁 Directory exists: {directory}")

def show_next_steps():
    """Show next steps for the user"""
    print("\n" + "=" * 60)
    print("🎉 Setup Complete! Next steps:")
    print("=" * 60)
    print("1. 📝 Edit .env file with your API keys")
    print("2. 🚀 Start the server:")
    print("   python main.py")
    print("3. 🧪 Test the AgentSession:")
    print("   python example_usage.py interactive")
    print("4. 📚 Read documentation:")
    print("   README_AgentSession.md")
    print("\n🔗 Useful commands:")
    print("   - Start server: python main.py")
    print("   - Test agent: python example_usage.py")
    print("   - Interactive mode: python example_usage.py interactive")
    print("   - View logs: tail -f logs/app.log")

def main():
    """Main setup function"""
    print("🚀 SpeakServe AI Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Setup environment
    if not setup_environment():
        return False
    
    # Create directories
    create_directories()
    
    # Validate environment (optional, since user might not have filled .env yet)
    print("\n📋 Environment validation:")
    if validate_environment():
        print("✅ All checks passed!")
    else:
        print("⚠️  Please complete environment setup")
    
    # Show next steps
    show_next_steps()
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1) 