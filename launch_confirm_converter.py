#!/usr/bin/env python3
"""
CONFIRM Data Converter Launcher
Easy launcher for the CONFIRM Data Converter application
"""

import sys
import os
from pathlib import Path

def main():
    """Launch the CONFIRM Data Converter"""
    try:
        # Add current directory to Python path
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        
        # Import and run the application
        from confirm_data_converter import CONFIRMDataConverter
        
        print("Starting CONFIRM Data Converter...")
        print("Transform raw client data into contingency tables for statistical validation")
        print("-" * 70)
        
        app = CONFIRMDataConverter()
        app.run()
        
    except ImportError as e:
        print(f"Error: Missing required dependencies - {e}")
        print("Please install required packages:")
        print("sudo apt install python3-pandas python3-numpy python3-scipy python3-matplotlib python3-seaborn python3-openpyxl python3-sklearn python3-tk")
        sys.exit(1)
        
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()