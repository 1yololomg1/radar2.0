#!/usr/bin/env python3
"""
Launch script for CONFIRM Data Converter
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'openpyxl', 
        'sqlalchemy', 'plotly', 'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
        print("Dependencies installed successfully!")

def main():
    """Main function to launch the application."""
    print("Starting CONFIRM Data Converter...")
    
    # Check and install dependencies
    check_dependencies()
    
    # Launch Streamlit app
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 
            'confirm_data_converter.py',
            '--server.port', '8501',
            '--server.address', 'localhost'
        ])
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Error launching application: {e}")

if __name__ == "__main__":
    main()