#!/usr/bin/env python3
"""
DABBLE APP INSPECTOR
===================

This utility helps identify the correct UI elements and app package information
for the Dabble betting app. It's essential for configuring the scraper with
the correct element selectors and app details.

FUNCTIONALITY:
- Connect to Android device and inspect installed apps
- Find Dabble app package name and activities
- Capture and analyze UI elements from app screens
- Generate element selectors for scraping automation
- Export app structure and element mappings

USAGE:
1. Connect Android device with Dabble app installed
2. Run this script to inspect app structure
3. Navigate through app manually while script captures elements
4. Use generated selectors in main scraper configuration
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from appium import webdriver
from appium.webdriver.common.appiumby import AppiumBy
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException


class DabbleAppInspector:
    """Utility for inspecting Dabble app structure and UI elements"""
    
    def __init__(self):
        self.driver = None
        self.app_info = {}
        self.element_mappings = {}
        self.screenshots_dir = Path("inspection_screenshots")
        self.screenshots_dir.mkdir(exist_ok=True)
    
    def connect_to_device(self) -> bool:
        """Connect to Android device for inspection"""
        try:
            # Basic capabilities for inspection
            desired_caps = {
                'platformName': 'Android',
                'deviceName': 'Android Device',
                'automationName': 'UiAutomator2',
                'noReset': True,
                'fullReset': False
            }
            
            self.driver = webdriver.Remote(
                'http://localhost:4723/wd/hub',
                desired_caps
            )
            
            print("✓ Connected to Android device")
            return True
            
        except Exception as e:
            print(f"✗ Failed to connect to device: {e}")
            return False
    
    def find_dabble_app(self) -> Optional[Dict]:
        """Find Dabble app package information"""
        try:
            print("\nSearching for Dabble app...")
            
            # Get list of installed packages
            packages = self.driver.execute_script(
                'mobile: shell', 
                {'command': 'pm list packages | grep -i dabble'}
            )
            
            if packages:
                print(f"Found potential Dabble packages: {packages}")
                
                # Try common variations
                potential_packages = [
                    'com.dabble.app',
                    'com.dabble.betting',
                    'com.dabble.sports',
                    'au.com.dabble',
                    'com.dabblebet.app'
                ]
                
                for package in potential_packages:
                    try:
                        # Try to get app info
                        app_info = self.driver.execute_script(
                            'mobile: shell',
                            {'command': f'dumpsys package {package}'}
                        )
                        
                        if app_info and 'Activities:' in app_info:
                            print(f"✓ Found Dabble app: {package}")
                            self.app_info['package'] = package
                            self.parse_app_activities(app_info)
                            return self.app_info
                            
                    except Exception:
                        continue
            
            # Manual search through all apps
            print("Searching through all installed apps...")
            all_packages = self.driver.execute_script(
                'mobile: shell',
                {'command': 'pm list packages'}
            )
            
            if all_packages:
                for line in all_packages.split('\n'):
                    if 'package:' in line:
                        package = line.replace('package:', '').strip()
                        if any(keyword in package.lower() for keyword in ['bet', 'sport', 'gambl']):
                            print(f"Found betting-related app: {package}")
            
            return None
            
        except Exception as e:
            print(f"Error searching for Dabble app: {e}")
            return None
    
    def parse_app_activities(self, app_info: str):
        """Parse app activities from dumpsys output"""
        try:
            activities = []
            in_activities_section = False
            
            for line in app_info.split('\n'):
                line = line.strip()
                
                if 'Activities:' in line:
                    in_activities_section = True
                    continue
                
                if in_activities_section:
                    if line.startswith('Activity #'):
                        # Extract activity name
                        if ' ' in line:
                            activity_part = line.split(' ', 2)
                            if len(activity_part) > 2:
                                activity_name = activity_part[2].split(' ')[0]
                                activities.append(activity_name)
                    elif line == '' or not line.startswith(' '):
                        break
            
            self.app_info['activities'] = activities
            if activities:
                self.app_info['main_activity'] = activities[0]
                print(f"Found {len(activities)} activities")
                
        except Exception as e:
            print(f"Error parsing activities: {e}")
    
    def launch_dabble_app(self, package_name: str) -> bool:
        """Launch Dabble app for inspection"""
        try:
            # Update capabilities with found package
            self.driver.quit()
            
            desired_caps = {
                'platformName': 'Android',
                'deviceName': 'Android Device', 
                'appPackage': package_name,
                'appActivity': self.app_info.get('main_activity', '.MainActivity'),
                'automationName': 'UiAutomator2',
                'noReset': True
            }
            
            self.driver = webdriver.Remote(
                'http://localhost:4723/wd/hub',
                desired_caps
            )
            
            print(f"✓ Launched Dabble app: {package_name}")
            time.sleep(3)  # Wait for app to load
            return True
            
        except Exception as e:
            print(f"✗ Failed to launch app: {e}")
            return False
    
    def capture_current_screen(self, screen_name: str = None) -> str:
        """Capture screenshot and page source"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screen_name = screen_name or f"screen_{timestamp}"
            
            # Take screenshot
            screenshot_path = self.screenshots_dir / f"{screen_name}.png"
            self.driver.save_screenshot(str(screenshot_path))
            
            # Save page source
            source_path = self.screenshots_dir / f"{screen_name}_source.xml"
            with open(source_path, 'w', encoding='utf-8') as f:
                f.write(self.driver.page_source)
            
            print(f"✓ Captured screen: {screen_name}")
            return screen_name
            
        except Exception as e:
            print(f"Error capturing screen: {e}")
            return ""
    
    def analyze_current_elements(self) -> Dict:
        """Analyze UI elements on current screen"""
        try:
            elements_info = {
                'text_elements': [],
                'clickable_elements': [],
                'input_elements': [],
                'navigation_elements': []
            }
            
            # Find all text elements
            text_elements = self.driver.find_elements(AppiumBy.CLASS_NAME, "android.widget.TextView")
            for element in text_elements:
                try:
                    text = element.text.strip()
                    if text:
                        element_info = {
                            'text': text,
                            'resource_id': element.get_attribute('resource-id'),
                            'class': element.get_attribute('class'),
                            'clickable': element.get_attribute('clickable'),
                            'bounds': element.get_attribute('bounds')
                        }
                        elements_info['text_elements'].append(element_info)
                except Exception:
                    continue
            
            # Find clickable elements
            clickable_elements = self.driver.find_elements(AppiumBy.XPATH, "//*[@clickable='true']")
            for element in clickable_elements:
                try:
                    element_info = {
                        'text': element.text.strip() if element.text else '',
                        'resource_id': element.get_attribute('resource-id'),
                        'class': element.get_attribute('class'),
                        'content_desc': element.get_attribute('content-desc'),
                        'bounds': element.get_attribute('bounds')
                    }
                    elements_info['clickable_elements'].append(element_info)
                except Exception:
                    continue
            
            # Find input elements
            input_elements = self.driver.find_elements(AppiumBy.CLASS_NAME, "android.widget.EditText")
            for element in input_elements:
                try:
                    element_info = {
                        'hint': element.get_attribute('hint'),
                        'text': element.get_attribute('text'),
                        'resource_id': element.get_attribute('resource-id'),
                        'bounds': element.get_attribute('bounds')
                    }
                    elements_info['input_elements'].append(element_info)
                except Exception:
                    continue
            
            return elements_info
            
        except Exception as e:
            print(f"Error analyzing elements: {e}")
            return {}
    
    def find_user_profile_elements(self) -> Dict:
        """Look for user profile related elements"""
        try:
            profile_elements = {}
            
            # Look for common profile indicators
            profile_keywords = ['profile', 'user', 'account', 'last 10', 'history', 'bets']
            
            for keyword in profile_keywords:
                # Search by text
                try:
                    elements = self.driver.find_elements(
                        AppiumBy.XPATH, 
                        f"//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{keyword}')]"
                    )
                    if elements:
                        profile_elements[keyword] = []
                        for element in elements:
                            profile_elements[keyword].append({
                                'text': element.text,
                                'resource_id': element.get_attribute('resource-id'),
                                'class': element.get_attribute('class'),
                                'clickable': element.get_attribute('clickable')
                            })
                except Exception:
                    continue
            
            return profile_elements
            
        except Exception as e:
            print(f"Error finding profile elements: {e}")
            return {}
    
    def interactive_inspection(self):
        """Interactive mode for manual app inspection"""
        print("\n" + "="*60)
        print("INTERACTIVE INSPECTION MODE")
        print("="*60)
        print("Commands:")
        print("  'capture' - Take screenshot and analyze current screen")
        print("  'elements' - List all elements on current screen")
        print("  'profile' - Search for profile-related elements")
        print("  'text <search>' - Search for elements containing text")
        print("  'click <resource_id>' - Click element by resource ID")
        print("  'back' - Press back button")
        print("  'home' - Press home button")
        print("  'quit' - Exit inspection mode")
        print("\nNavigate through the app manually and use commands to inspect elements.")
        
        screen_count = 0
        
        while True:
            try:
                command = input("\nInspection> ").strip().lower()
                
                if command == 'quit':
                    break
                
                elif command == 'capture':
                    screen_count += 1
                    screen_name = self.capture_current_screen(f"manual_{screen_count}")
                    elements = self.analyze_current_elements()
                    
                    print(f"Found {len(elements['text_elements'])} text elements")
                    print(f"Found {len(elements['clickable_elements'])} clickable elements")
                    
                    # Save elements info
                    elements_path = self.screenshots_dir / f"{screen_name}_elements.json"
                    with open(elements_path, 'w') as f:
                        json.dump(elements, f, indent=2)
                
                elif command == 'elements':
                    elements = self.analyze_current_elements()
                    
                    print("\nText Elements:")
                    for i, elem in enumerate(elements['text_elements'][:10]):  # Show first 10
                        print(f"  {i+1}. '{elem['text']}' (ID: {elem['resource_id']})")
                    
                    print("\nClickable Elements:")
                    for i, elem in enumerate(elements['clickable_elements'][:10]):
                        text = elem['text'] or elem['content_desc'] or 'No text'
                        print(f"  {i+1}. '{text}' (ID: {elem['resource_id']})")
                
                elif command == 'profile':
                    profile_elements = self.find_user_profile_elements()
                    
                    for keyword, elements in profile_elements.items():
                        if elements:
                            print(f"\n{keyword.upper()} elements:")
                            for elem in elements:
                                print(f"  - '{elem['text']}' (ID: {elem['resource_id']})")
                
                elif command.startswith('text '):
                    search_text = command[5:]
                    try:
                        elements = self.driver.find_elements(
                            AppiumBy.XPATH,
                            f"//*[contains(text(), '{search_text}')]"
                        )
                        
                        print(f"\nFound {len(elements)} elements containing '{search_text}':")
                        for elem in elements:
                            print(f"  - '{elem.text}' (ID: {elem.get_attribute('resource-id')})")
                    
                    except Exception as e:
                        print(f"Search error: {e}")
                
                elif command.startswith('click '):
                    resource_id = command[6:]
                    try:
                        element = self.driver.find_element(AppiumBy.ID, resource_id)
                        element.click()
                        print(f"Clicked element: {resource_id}")
                        time.sleep(1)
                    
                    except Exception as e:
                        print(f"Click error: {e}")
                
                elif command == 'back':
                    self.driver.back()
                    print("Pressed back button")
                    time.sleep(1)
                
                elif command == 'home':
                    self.driver.press_keycode(3)  # Home key
                    print("Pressed home button")
                    time.sleep(1)
                
                else:
                    print("Unknown command. Type 'quit' to exit.")
            
            except KeyboardInterrupt:
                print("\nExiting inspection mode...")
                break
            except Exception as e:
                print(f"Command error: {e}")
    
    def generate_config_suggestions(self) -> Dict:
        """Generate configuration suggestions based on inspection"""
        try:
            suggestions = {
                'app_package': self.app_info.get('package', 'com.dabble.app'),
                'app_activity': self.app_info.get('main_activity', '.MainActivity'),
                'element_selectors': {},
                'navigation_flow': [],
                'recommendations': []
            }
            
            # Analyze captured screens for common patterns
            for screenshot_file in self.screenshots_dir.glob("*_elements.json"):
                try:
                    with open(screenshot_file, 'r') as f:
                        elements = json.load(f)
                    
                    # Look for profile-related elements
                    for elem in elements.get('text_elements', []):
                        text = elem['text'].lower()
                        if any(keyword in text for keyword in ['last 10', 'profile', 'history']):
                            suggestions['element_selectors']['profile_tab'] = {
                                'text': elem['text'],
                                'resource_id': elem['resource_id'],
                                'selector_type': 'id' if elem['resource_id'] else 'text'
                            }
                    
                    # Look for user search elements
                    for elem in elements.get('input_elements', []):
                        if elem.get('hint') and 'search' in elem['hint'].lower():
                            suggestions['element_selectors']['search_field'] = {
                                'resource_id': elem['resource_id'],
                                'hint': elem['hint']
                            }
                
                except Exception:
                    continue
            
            # Add recommendations
            suggestions['recommendations'] = [
                "Test app navigation manually first",
                "Verify element selectors work consistently", 
                "Check if app requires login or permissions",
                "Test with different screen sizes/orientations",
                "Validate bet text parsing with real data"
            ]
            
            return suggestions
            
        except Exception as e:
            print(f"Error generating suggestions: {e}")
            return {}
    
    def save_inspection_results(self):
        """Save all inspection results to JSON file"""
        try:
            results = {
                'timestamp': datetime.now().isoformat(),
                'app_info': self.app_info,
                'element_mappings': self.element_mappings,
                'config_suggestions': self.generate_config_suggestions()
            }
            
            results_path = Path("dabble_app_inspection_results.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n✓ Saved inspection results to: {results_path}")
            
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def disconnect(self):
        """Disconnect from device"""
        if self.driver:
            try:
                self.driver.quit()
                print("✓ Disconnected from device")
            except Exception as e:
                print(f"Error disconnecting: {e}")


def main():
    """Main inspection function"""
    print("="*60)
    print("DABBLE APP INSPECTOR")
    print("="*60)
    print("This tool helps identify UI elements and app structure for scraping.")
    print("Make sure:")
    print("- Android device is connected with USB debugging")
    print("- Appium server is running (appium)")
    print("- Dabble app is installed on the device")
    
    inspector = DabbleAppInspector()
    
    try:
        # Connect to device
        if not inspector.connect_to_device():
            return
        
        # Find Dabble app
        app_info = inspector.find_dabble_app()
        if not app_info:
            print("\n✗ Could not find Dabble app automatically")
            package_name = input("Enter Dabble app package name manually: ").strip()
            if package_name:
                inspector.app_info['package'] = package_name
            else:
                print("Cannot proceed without app package name")
                return
        
        # Launch app
        if inspector.launch_dabble_app(inspector.app_info['package']):
            # Initial screen capture
            inspector.capture_current_screen("initial_screen")
            
            # Start interactive inspection
            inspector.interactive_inspection()
        
        # Save results
        inspector.save_inspection_results()
        
        print("\n" + "="*60)
        print("INSPECTION COMPLETE")
        print("="*60)
        print("Check the following files:")
        print("- dabble_app_inspection_results.json (configuration suggestions)")
        print("- inspection_screenshots/ (screenshots and element data)")
        print("\nUse the results to update config.json with correct app package and element selectors.")
    
    except KeyboardInterrupt:
        print("\nInspection cancelled by user")
    
    finally:
        inspector.disconnect()


if __name__ == "__main__":
    main()