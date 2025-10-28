#!/usr/bin/env python3
"""
DABBLE SCRAPER ENVIRONMENT SETUP
================================

This script sets up the complete environment for the Dabble betting history scraper.
It handles installation of dependencies, configuration of Appium server, and 
creation of necessary configuration files.

SETUP PROCESS:
1. Install Python dependencies from requirements.txt
2. Check for and install Appium server and dependencies
3. Create default configuration files
4. Set up Google Sheets authentication
5. Validate Android device connection
6. Create directory structure and logging setup

REQUIREMENTS:
- Python 3.8+ with pip
- Node.js and npm (for Appium server)
- Android SDK and ADB tools
- Android device with USB debugging enabled
- Google Cloud service account for Sheets API
"""

import os
import sys
import json
import subprocess
import platform
from pathlib import Path
from typing import Dict, List, Tuple


class EnvironmentSetup:
    """Handles complete environment setup for Dabble scraper"""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.setup_results = {
            'python_deps': False,
            'appium_server': False,
            'android_tools': False,
            'config_files': False,
            'google_auth': False,
            'device_connection': False
        }
    
    def run_setup(self) -> bool:
        """Execute complete setup process"""
        print("="*60)
        print("DABBLE SCRAPER ENVIRONMENT SETUP")
        print("="*60)
        
        steps = [
            ("Installing Python dependencies", self.install_python_dependencies),
            ("Setting up Appium server", self.setup_appium),
            ("Checking Android tools", self.check_android_tools),
            ("Creating configuration files", self.create_config_files),
            ("Setting up Google Sheets authentication", self.setup_google_auth),
            ("Testing device connection", self.test_device_connection)
        ]
        
        for step_name, step_func in steps:
            print(f"\n{step_name}...")
            try:
                success = step_func()
                status = "✓ SUCCESS" if success else "✗ FAILED"
                print(f"  {status}")
            except Exception as e:
                print(f"  ✗ ERROR: {e}")
                success = False
            
            if not success:
                print(f"\nSetup failed at step: {step_name}")
                self.print_troubleshooting_guide()
                return False
        
        print("\n" + "="*60)
        print("SETUP COMPLETED SUCCESSFULLY!")
        print("="*60)
        self.print_next_steps()
        return True
    
    def install_python_dependencies(self) -> bool:
        """Install required Python packages"""
        try:
            # Check if requirements.txt exists
            if not os.path.exists('requirements.txt'):
                print("  Warning: requirements.txt not found, creating minimal version")
                self.create_minimal_requirements()
            
            # Install dependencies
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.setup_results['python_deps'] = True
                print("  Installed Python dependencies successfully")
                return True
            else:
                print(f"  Error installing dependencies: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"  Exception during pip install: {e}")
            return False
    
    def create_minimal_requirements(self):
        """Create minimal requirements.txt if missing"""
        minimal_reqs = [
            "Appium-Python-Client>=3.0.0",
            "selenium>=4.0.0", 
            "gspread>=5.0.0",
            "google-auth>=2.0.0",
            "APScheduler>=3.0.0"
        ]
        
        with open('requirements.txt', 'w') as f:
            f.write('\n'.join(minimal_reqs))
    
    def setup_appium(self) -> bool:
        """Set up Appium server and dependencies"""
        try:
            # Check if Node.js is installed
            node_result = subprocess.run(['node', '--version'], 
                                       capture_output=True, text=True)
            if node_result.returncode != 0:
                print("  Node.js not found. Please install Node.js first.")
                print("  Download from: https://nodejs.org/")
                return False
            
            print(f"  Found Node.js: {node_result.stdout.strip()}")
            
            # Check if Appium is installed
            appium_result = subprocess.run(['appium', '--version'], 
                                         capture_output=True, text=True)
            
            if appium_result.returncode != 0:
                print("  Installing Appium server...")
                install_result = subprocess.run(['npm', 'install', '-g', 'appium'], 
                                              capture_output=True, text=True)
                if install_result.returncode != 0:
                    print(f"  Failed to install Appium: {install_result.stderr}")
                    return False
            
            # Install UiAutomator2 driver
            print("  Installing UiAutomator2 driver...")
            driver_result = subprocess.run(['appium', 'driver', 'install', 'uiautomator2'], 
                                         capture_output=True, text=True)
            
            self.setup_results['appium_server'] = True
            print("  Appium server setup completed")
            return True
            
        except Exception as e:
            print(f"  Exception during Appium setup: {e}")
            return False
    
    def check_android_tools(self) -> bool:
        """Check Android SDK and ADB tools"""
        try:
            # Check ADB
            adb_result = subprocess.run(['adb', 'version'], 
                                      capture_output=True, text=True)
            if adb_result.returncode != 0:
                print("  ADB not found. Please install Android SDK Platform Tools.")
                print("  Download from: https://developer.android.com/studio/releases/platform-tools")
                return False
            
            print(f"  Found ADB")
            
            # Check connected devices
            devices_result = subprocess.run(['adb', 'devices'], 
                                          capture_output=True, text=True)
            
            if devices_result.returncode == 0:
                devices_output = devices_result.stdout
                if 'device' in devices_output and len(devices_output.split('\n')) > 2:
                    print("  Android device(s) detected")
                else:
                    print("  No Android devices connected")
                    print("  Please connect your Android device with USB debugging enabled")
            
            self.setup_results['android_tools'] = True
            return True
            
        except Exception as e:
            print(f"  Exception checking Android tools: {e}")
            return False
    
    def create_config_files(self) -> bool:
        """Create necessary configuration files"""
        try:
            # Create main config file
            config = {
                "appium": {
                    "server_url": "http://localhost:4723/wd/hub",
                    "desired_capabilities": {
                        "platformName": "Android",
                        "deviceName": "Android Device",
                        "appPackage": "com.dabble.betting",
                        "appActivity": ".MainActivity",
                        "automationName": "UiAutomator2",
                        "noReset": True,
                        "fullReset": False
                    }
                },
                "google_sheets": {
                    "credentials_path": "google_credentials.json",
                    "spreadsheet_name": "Dabble Betting History"
                },
                "notifications": {
                    "email": {
                        "enabled": False,
                        "smtp_server": "smtp.gmail.com",
                        "smtp_port": 587,
                        "sender_email": "your-email@gmail.com",
                        "sender_password": "your-app-password",
                        "recipient_email": "recipient@gmail.com"
                    }
                },
                "scraping": {
                    "timeout": 30,
                    "retry_attempts": 3,
                    "delay_between_users": 2,
                    "screenshot_on_error": True
                }
            }
            
            with open('config.json', 'w') as f:
                json.dump(config, f, indent=2)
            
            # Create users file
            users_data = {
                "users": [
                    "Durag",
                    "Reggie", 
                    "RayWitDaLocks",
                    "SportsAlmanac"
                ],
                "last_check": {},
                "stats": {}
            }
            
            with open('users.json', 'w') as f:
                json.dump(users_data, f, indent=2)
            
            # Create directories
            os.makedirs('logs', exist_ok=True)
            os.makedirs('screenshots', exist_ok=True)
            os.makedirs('backups', exist_ok=True)
            
            self.setup_results['config_files'] = True
            print("  Created configuration files and directories")
            return True
            
        except Exception as e:
            print(f"  Exception creating config files: {e}")
            return False
    
    def setup_google_auth(self) -> bool:
        """Set up Google Sheets authentication"""
        try:
            if os.path.exists('google_credentials.json'):
                print("  Google credentials file already exists")
                self.setup_results['google_auth'] = True
                return True
            
            print("  Google credentials not found.")
            print("  Please follow these steps to set up Google Sheets access:")
            print("  1. Go to https://console.cloud.google.com/")
            print("  2. Create a new project or select existing one")
            print("  3. Enable Google Sheets API and Google Drive API")
            print("  4. Create a Service Account")
            print("  5. Download the JSON credentials file")
            print("  6. Rename it to 'google_credentials.json' in this directory")
            
            # Create template file
            template = {
                "type": "service_account",
                "project_id": "your-project-id",
                "private_key_id": "your-private-key-id",
                "private_key": "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n",
                "client_email": "your-service-account@your-project.iam.gserviceaccount.com",
                "client_id": "your-client-id",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40your-project.iam.gserviceaccount.com"
            }
            
            with open('google_credentials_template.json', 'w') as f:
                json.dump(template, f, indent=2)
            
            print("  Created google_credentials_template.json as reference")
            
            # For now, mark as completed if user confirms they'll set it up
            response = input("  Have you already set up Google Sheets credentials? (y/n): ")
            if response.lower() == 'y':
                self.setup_results['google_auth'] = True
                return True
            else:
                print("  Please set up Google credentials before running the scraper")
                return False
                
        except Exception as e:
            print(f"  Exception during Google auth setup: {e}")
            return False
    
    def test_device_connection(self) -> bool:
        """Test Android device connection"""
        try:
            # List connected devices
            result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                connected_devices = [line for line in lines if 'device' in line and 'offline' not in line]
                
                if connected_devices:
                    print(f"  Found {len(connected_devices)} connected device(s)")
                    
                    # Test basic ADB command
                    test_result = subprocess.run(['adb', 'shell', 'echo', 'test'], 
                                               capture_output=True, text=True)
                    if test_result.returncode == 0:
                        print("  Device communication test successful")
                        self.setup_results['device_connection'] = True
                        return True
                    else:
                        print("  Device communication test failed")
                        return False
                else:
                    print("  No devices connected or devices offline")
                    print("  Please ensure USB debugging is enabled and device is connected")
                    return False
            else:
                print("  Failed to check device connection")
                return False
                
        except Exception as e:
            print(f"  Exception testing device connection: {e}")
            return False
    
    def print_troubleshooting_guide(self):
        """Print troubleshooting information"""
        print("\n" + "="*60)
        print("TROUBLESHOOTING GUIDE")
        print("="*60)
        
        print("\n1. PYTHON DEPENDENCIES:")
        print("   - Ensure Python 3.8+ is installed")
        print("   - Try: pip install --upgrade pip")
        print("   - For permission issues: pip install --user")
        
        print("\n2. APPIUM SERVER:")
        print("   - Install Node.js from https://nodejs.org/")
        print("   - Install Appium: npm install -g appium")
        print("   - Install driver: appium driver install uiautomator2")
        
        print("\n3. ANDROID TOOLS:")
        print("   - Download Android SDK Platform Tools")
        print("   - Add ADB to system PATH")
        print("   - Enable USB debugging on Android device")
        print("   - Trust computer on device when prompted")
        
        print("\n4. GOOGLE SHEETS:")
        print("   - Create Google Cloud project")
        print("   - Enable Sheets and Drive APIs")
        print("   - Create service account and download JSON")
        print("   - Share spreadsheet with service account email")
        
        print("\n5. DEVICE CONNECTION:")
        print("   - Check USB cable and connection")
        print("   - Try different USB port")
        print("   - Restart ADB: adb kill-server && adb start-server")
        print("   - Check device authorization: adb devices")
    
    def print_next_steps(self):
        """Print next steps after successful setup"""
        print("\nNEXT STEPS:")
        print("1. Update config.json with correct Dabble app package name")
        print("2. Set up Google Sheets credentials (google_credentials.json)")
        print("3. Test manual scraping: python dabble_scraper.py --manual")
        print("4. Add/remove users: python dabble_scraper.py --add-user USERNAME")
        print("5. Start scheduled scraping: python dabble_scraper.py --schedule")
        
        print("\nIMPORTANT NOTES:")
        print("- Make sure Dabble app is installed on connected device")
        print("- Verify app package name in config.json")
        print("- Test with one user first before adding all users")
        print("- Check logs/dabble_scraper.log for detailed information")


def main():
    """Main setup function"""
    setup = EnvironmentSetup()
    
    print("This script will set up the Dabble scraper environment.")
    print("Make sure you have:")
    print("- Python 3.8+ installed")
    print("- Android device with USB debugging enabled")
    print("- Internet connection for downloading dependencies")
    
    response = input("\nProceed with setup? (y/n): ")
    if response.lower() != 'y':
        print("Setup cancelled.")
        return
    
    success = setup.run_setup()
    
    if success:
        print("\nSetup completed! You can now run the Dabble scraper.")
    else:
        print("\nSetup failed. Please check the troubleshooting guide above.")
        sys.exit(1)


if __name__ == "__main__":
    main()