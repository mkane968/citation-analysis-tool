import subprocess
import sys
import os

def install_requirements():
    """Install required packages from requirements.txt"""
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("Package installation complete!")

def download_nltk_data():
    """Download required NLTK data"""
    print("Downloading NLTK data...")
    import nltk
    nltk.download('punkt')
    print("NLTK data download complete!")

def main():
    """Run the setup process"""
    print("Setting up Citation Analysis Tool...")
    
    # Install required packages
    install_requirements()
    
    # Download NLTK data
    download_nltk_data()
    
    print("\nSetup complete! You can now run the application with:")
    print("python app_public.py")
    print("\nOpen your browser and go to: http://127.0.0.1:5000/")

if __name__ == "__main__":
    main()
