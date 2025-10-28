#!/usr/bin/env python3
"""
DABBLE BETTING HISTORY SCRAPER
==============================

SYSTEM ARCHITECTURE:
- DabbleScraper: Main orchestration class handling app automation
- BetParser: Text parsing engine for extracting bet data from screen text
- GoogleSheetsManager: Handles all Google Sheets operations and data storage
- UserManager: Manages user lists and tracking state
- ScheduleManager: Handles automation scheduling and triggers
- DuplicateDetector: Prevents duplicate bet entries

MAIN CLASSES:
- DabbleScraper: Android app automation using Appium WebDriver
- BetParser: Regex-based text parsing for bet leg extraction
- GoogleSheetsManager: Google Sheets API integration with authentication
- UserManager: User profile management and state tracking
- ScheduleManager: APScheduler-based automation scheduling

DATA FLOW:
1. Load user list and last check timestamps
2. For each user: Navigate to profile → "Last 10" tab
3. Extract screen text and parse bet legs
4. Filter out duplicates based on date/user/bet content
5. Append new bets to Google Sheets
6. Update user tracking state and timestamps
7. Send completion notifications

KEY FUNCTIONS:
- scrape_user_bets(): Main scraping logic for individual users
- parse_bet_text(): Extract structured data from raw screen text
- detect_duplicates(): Compare new bets against existing data
- update_sheets(): Append new data to Google Sheets
- schedule_daily_run(): Set up automated daily execution
"""

import os
import re
import json
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

# Android automation
from appium import webdriver
from appium.webdriver.common.appiumby import AppiumBy
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Google Sheets integration
import gspread
from google.oauth2.service_account import Credentials

# Scheduling
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

# Notifications
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


@dataclass
class BetLeg:
    """Represents a single bet leg with all extracted data"""
    date: str
    username: str
    sport: str
    player: str
    stat_type: str
    predicted: str  # OVER/UNDER
    line: float
    result: str  # WON/LOST/PENDING
    correct: bool = None  # Will be calculated based on result
    
    def __post_init__(self):
        """Calculate correct field based on result"""
        if self.result == "WON":
            self.correct = True
        elif self.result == "LOST":
            self.correct = False
        else:  # PENDING
            self.correct = None


class BetParser:
    """Text parsing engine for extracting bet data from screen captures"""
    
    def __init__(self):
        self.setup_patterns()
        
    def setup_patterns(self):
        """Initialize regex patterns for different bet formats"""
        # Pattern for player and stat extraction
        self.player_pattern = re.compile(r'([A-Za-z\s\.]+)\s*\(([A-Z]{2,4})-([A-Z]{1,3})\)')
        
        # Pattern for stat line (e.g., "↑ MORE 47.5 Rushing Yards", "↓ LESS 2.5 Touchdowns")
        self.stat_pattern = re.compile(r'[↑↓]\s*(MORE|LESS)\s*(\d+\.?\d*)\s*(.+)')
        
        # Pattern for game matchup (e.g., "WAS @ KC", "LAL vs BOS")
        self.matchup_pattern = re.compile(r'([A-Z]{2,4})\s*[@vs]\s*([A-Z]{2,4})')
        
        # Pattern for result
        self.result_pattern = re.compile(r'\b(WON|LOST|PENDING)\b', re.IGNORECASE)
        
        # Sport detection patterns
        self.sport_patterns = {
            'NFL': re.compile(r'\b(Rush|Receiving|Passing|Touchdowns?|TD|Yards?|YDS)\b', re.IGNORECASE),
            'NBA': re.compile(r'\b(Points?|PTS|Rebounds?|REB|Assists?|AST|Steals?|STL|Blocks?|BLK)\b', re.IGNORECASE),
            'NHL': re.compile(r'\b(Goals?|Assists?|Points?|Shots?|Saves?|Hits?)\b', re.IGNORECASE),
            'MLB': re.compile(r'\b(Hits?|Home Runs?|HR|RBI|Runs?|Strikeouts?|K)\b', re.IGNORECASE)
        }
        
        # Stat type normalization mapping
        self.stat_mapping = {
            'rushing yards': 'Rushing_Yards',
            'receiving yards': 'Receiving_Yards', 
            'passing yards': 'Passing_Yards',
            'touchdowns': 'Touchdowns',
            'points': 'Points',
            'rebounds': 'Rebounds',
            'assists': 'Assists',
            'steals': 'Steals',
            'blocks': 'Blocks',
            'goals': 'Goals',
            'shots': 'Shots',
            'saves': 'Saves',
            'hits': 'Hits',
            'home runs': 'Home_Runs',
            'rbi': 'RBI',
            'runs': 'Runs',
            'strikeouts': 'Strikeouts'
        }
    
    def parse_bet_text(self, text: str, username: str, date: str) -> List[BetLeg]:
        """
        Parse raw text from screen capture into structured bet legs
        
        Args:
            text: Raw text extracted from screen
            username: Username who placed the bet
            date: Date when bet was placed
            
        Returns:
            List of BetLeg objects representing individual bet legs
        """
        bet_legs = []
        lines = text.strip().split('\n')
        
        current_bet = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Try to extract player info
            player_match = self.player_pattern.search(line)
            if player_match:
                current_bet['player'] = player_match.group(1).strip()
                current_bet['team'] = player_match.group(2)
                current_bet['position'] = player_match.group(3)
                continue
            
            # Try to extract stat line
            stat_match = self.stat_pattern.search(line)
            if stat_match:
                current_bet['predicted'] = 'OVER' if stat_match.group(1) == 'MORE' else 'UNDER'
                current_bet['line'] = float(stat_match.group(2))
                stat_text = stat_match.group(3).strip().lower()
                current_bet['stat_type'] = self.normalize_stat_type(stat_text)
                current_bet['sport'] = self.detect_sport(stat_text)
                continue
            
            # Try to extract result
            result_match = self.result_pattern.search(line)
            if result_match:
                current_bet['result'] = result_match.group(1).upper()
                
                # If we have all required fields, create BetLeg
                if self.is_complete_bet(current_bet):
                    bet_leg = BetLeg(
                        date=date,
                        username=username,
                        sport=current_bet.get('sport', 'UNKNOWN'),
                        player=current_bet.get('player', 'UNKNOWN'),
                        stat_type=current_bet.get('stat_type', 'UNKNOWN'),
                        predicted=current_bet.get('predicted', 'UNKNOWN'),
                        line=current_bet.get('line', 0.0),
                        result=current_bet.get('result', 'PENDING')
                    )
                    bet_legs.append(bet_leg)
                
                # Reset for next bet
                current_bet = {}
        
        return bet_legs
    
    def normalize_stat_type(self, stat_text: str) -> str:
        """Normalize stat type text to standard format"""
        stat_text = stat_text.lower().strip()
        
        # Direct mapping
        if stat_text in self.stat_mapping:
            return self.stat_mapping[stat_text]
        
        # Fuzzy matching for variations
        for key, value in self.stat_mapping.items():
            if key in stat_text or stat_text in key:
                return value
        
        # Default fallback
        return stat_text.replace(' ', '_').title()
    
    def detect_sport(self, stat_text: str) -> str:
        """Detect sport based on stat type"""
        for sport, pattern in self.sport_patterns.items():
            if pattern.search(stat_text):
                return sport
        return 'UNKNOWN'
    
    def is_complete_bet(self, bet_data: Dict) -> bool:
        """Check if bet data contains all required fields"""
        required_fields = ['player', 'stat_type', 'predicted', 'line', 'result']
        return all(field in bet_data for field in required_fields)


class GoogleSheetsManager:
    """Handles all Google Sheets operations and data storage"""
    
    def __init__(self, credentials_path: str, spreadsheet_name: str):
        self.credentials_path = credentials_path
        self.spreadsheet_name = spreadsheet_name
        self.client = None
        self.sheet = None
        self.setup_connection()
    
    def setup_connection(self):
        """Initialize Google Sheets connection"""
        try:
            # Define the scope
            scope = [
                'https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive'
            ]
            
            # Load credentials
            creds = Credentials.from_service_account_file(
                self.credentials_path, 
                scopes=scope
            )
            
            # Initialize client
            self.client = gspread.authorize(creds)
            
            # Open or create spreadsheet
            try:
                self.sheet = self.client.open(self.spreadsheet_name).sheet1
                logging.info(f"Connected to existing spreadsheet: {self.spreadsheet_name}")
            except gspread.SpreadsheetNotFound:
                # Create new spreadsheet
                self.sheet = self.client.create(self.spreadsheet_name).sheet1
                self.setup_headers()
                logging.info(f"Created new spreadsheet: {self.spreadsheet_name}")
                
        except Exception as e:
            logging.error(f"Failed to setup Google Sheets connection: {e}")
            raise
    
    def setup_headers(self):
        """Set up column headers in the spreadsheet"""
        headers = [
            'Date', 'User', 'Sport', 'Player', 'Stat_Type', 
            'Predicted', 'Line', 'Result', 'Correct', 'Timestamp'
        ]
        self.sheet.insert_row(headers, 1)
        logging.info("Set up spreadsheet headers")
    
    def append_bets(self, bet_legs: List[BetLeg]) -> int:
        """
        Append new bet legs to the spreadsheet
        
        Args:
            bet_legs: List of BetLeg objects to append
            
        Returns:
            Number of rows added
        """
        if not bet_legs:
            return 0
        
        try:
            # Prepare rows for insertion
            rows = []
            timestamp = datetime.now().isoformat()
            
            for bet in bet_legs:
                row = [
                    bet.date,
                    bet.username,
                    bet.sport,
                    bet.player,
                    bet.stat_type,
                    bet.predicted,
                    bet.line,
                    bet.result,
                    bet.correct,
                    timestamp
                ]
                rows.append(row)
            
            # Append all rows at once for efficiency
            self.sheet.append_rows(rows)
            
            logging.info(f"Successfully appended {len(rows)} bet legs to spreadsheet")
            return len(rows)
            
        except Exception as e:
            logging.error(f"Failed to append bets to spreadsheet: {e}")
            raise
    
    def get_existing_bets(self, username: str = None, days_back: int = 30) -> List[Dict]:
        """
        Retrieve existing bets for duplicate detection
        
        Args:
            username: Filter by specific username (optional)
            days_back: Number of days to look back for duplicates
            
        Returns:
            List of existing bet records
        """
        try:
            # Get all records
            records = self.sheet.get_all_records()
            
            # Filter by date range
            cutoff_date = datetime.now() - timedelta(days=days_back)
            filtered_records = []
            
            for record in records:
                try:
                    record_date = datetime.fromisoformat(record.get('Date', ''))
                    if record_date >= cutoff_date:
                        if username is None or record.get('User') == username:
                            filtered_records.append(record)
                except (ValueError, TypeError):
                    # Skip records with invalid dates
                    continue
            
            return filtered_records
            
        except Exception as e:
            logging.error(f"Failed to retrieve existing bets: {e}")
            return []


class DuplicateDetector:
    """Prevents duplicate bet entries using multiple matching strategies"""
    
    def __init__(self, sheets_manager: GoogleSheetsManager):
        self.sheets_manager = sheets_manager
    
    def filter_duplicates(self, new_bets: List[BetLeg], username: str) -> List[BetLeg]:
        """
        Filter out duplicate bets from new bet list
        
        Args:
            new_bets: List of newly scraped bet legs
            username: Username to check duplicates for
            
        Returns:
            List of bet legs that are not duplicates
        """
        if not new_bets:
            return []
        
        # Get existing bets for comparison
        existing_bets = self.sheets_manager.get_existing_bets(username=username)
        
        # Convert existing bets to comparable format
        existing_signatures = set()
        for bet in existing_bets:
            signature = self.create_bet_signature(bet)
            existing_signatures.add(signature)
        
        # Filter new bets
        unique_bets = []
        for bet in new_bets:
            signature = self.create_bet_signature(asdict(bet))
            if signature not in existing_signatures:
                unique_bets.append(bet)
            else:
                logging.info(f"Filtered duplicate bet: {bet.player} {bet.stat_type} {bet.line}")
        
        logging.info(f"Filtered {len(new_bets) - len(unique_bets)} duplicates, {len(unique_bets)} unique bets remain")
        return unique_bets
    
    def create_bet_signature(self, bet_data: Dict) -> str:
        """
        Create a unique signature for a bet to detect duplicates
        
        Args:
            bet_data: Dictionary containing bet information
            
        Returns:
            String signature for duplicate detection
        """
        # Use key fields that should be unique for each bet
        signature_parts = [
            str(bet_data.get('Date', '')),
            str(bet_data.get('User', '')),
            str(bet_data.get('Player', '')),
            str(bet_data.get('Stat_Type', '')),
            str(bet_data.get('Predicted', '')),
            str(bet_data.get('Line', ''))
        ]
        
        return '|'.join(signature_parts).lower()


class UserManager:
    """Manages user lists and tracking state"""
    
    def __init__(self, users_file: str = 'users.json'):
        self.users_file = users_file
        self.users_data = self.load_users()
    
    def load_users(self) -> Dict:
        """Load user data from JSON file"""
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logging.warning(f"Invalid JSON in {self.users_file}, starting fresh")
        
        # Default user data structure
        return {
            'users': [
                'Durag',
                'Reggie', 
                'RayWitDaLocks',
                'SportsAlmanac'
            ],
            'last_check': {},
            'stats': {}
        }
    
    def save_users(self):
        """Save user data to JSON file"""
        try:
            with open(self.users_file, 'w') as f:
                json.dump(self.users_data, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save user data: {e}")
    
    def get_users(self) -> List[str]:
        """Get list of users to track"""
        return self.users_data.get('users', [])
    
    def add_user(self, username: str):
        """Add a new user to track"""
        if username not in self.users_data['users']:
            self.users_data['users'].append(username)
            self.save_users()
            logging.info(f"Added new user: {username}")
    
    def remove_user(self, username: str):
        """Remove a user from tracking"""
        if username in self.users_data['users']:
            self.users_data['users'].remove(username)
            self.save_users()
            logging.info(f"Removed user: {username}")
    
    def update_last_check(self, username: str):
        """Update last check timestamp for a user"""
        self.users_data['last_check'][username] = datetime.now().isoformat()
        self.save_users()
    
    def get_last_check(self, username: str) -> Optional[datetime]:
        """Get last check timestamp for a user"""
        timestamp_str = self.users_data['last_check'].get(username)
        if timestamp_str:
            try:
                return datetime.fromisoformat(timestamp_str)
            except ValueError:
                pass
        return None
    
    def update_stats(self, username: str, new_bets_count: int):
        """Update statistics for a user"""
        if username not in self.users_data['stats']:
            self.users_data['stats'][username] = {
                'total_bets': 0,
                'last_scrape_count': 0,
                'last_scrape_time': None
            }
        
        self.users_data['stats'][username]['total_bets'] += new_bets_count
        self.users_data['stats'][username]['last_scrape_count'] = new_bets_count
        self.users_data['stats'][username]['last_scrape_time'] = datetime.now().isoformat()
        self.save_users()


class DabbleScraper:
    """Main orchestration class handling app automation"""
    
    def __init__(self, config_file: str = 'config.json'):
        self.config = self.load_config(config_file)
        self.driver = None
        self.bet_parser = BetParser()
        self.sheets_manager = GoogleSheetsManager(
            self.config['google_sheets']['credentials_path'],
            self.config['google_sheets']['spreadsheet_name']
        )
        self.duplicate_detector = DuplicateDetector(self.sheets_manager)
        self.user_manager = UserManager()
        self.setup_logging()
    
    def load_config(self, config_file: str) -> Dict:
        """Load configuration from JSON file"""
        default_config = {
            'appium': {
                'server_url': 'http://localhost:4723/wd/hub',
                'desired_capabilities': {
                    'platformName': 'Android',
                    'deviceName': 'Android Device',
                    'appPackage': 'com.dabble.app',  # Replace with actual package name
                    'appActivity': '.MainActivity',   # Replace with actual activity
                    'automationName': 'UiAutomator2'
                }
            },
            'google_sheets': {
                'credentials_path': 'google_credentials.json',
                'spreadsheet_name': 'Dabble Betting History'
            },
            'notifications': {
                'email': {
                    'enabled': False,
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'sender_email': '',
                    'sender_password': '',
                    'recipient_email': ''
                }
            },
            'scraping': {
                'timeout': 30,
                'retry_attempts': 3,
                'delay_between_users': 2
            }
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    # Merge with defaults
                    default_config.update(user_config)
            except json.JSONDecodeError:
                logging.warning(f"Invalid JSON in {config_file}, using defaults")
        else:
            # Save default config
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            logging.info(f"Created default config file: {config_file}")
        
        return default_config
    
    def setup_logging(self):
        """Configure logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('dabble_scraper.log'),
                logging.StreamHandler()
            ]
        )
    
    def connect_to_device(self):
        """Initialize Appium WebDriver connection"""
        try:
            self.driver = webdriver.Remote(
                self.config['appium']['server_url'],
                self.config['appium']['desired_capabilities']
            )
            logging.info("Successfully connected to Android device")
            return True
        except Exception as e:
            logging.error(f"Failed to connect to device: {e}")
            return False
    
    def disconnect_device(self):
        """Close Appium WebDriver connection"""
        if self.driver:
            try:
                self.driver.quit()
                logging.info("Disconnected from Android device")
            except Exception as e:
                logging.error(f"Error disconnecting from device: {e}")
    
    def navigate_to_user_profile(self, username: str) -> bool:
        """
        Navigate to specific user profile in Dabble app
        
        Args:
            username: Username to navigate to
            
        Returns:
            True if navigation successful, False otherwise
        """
        try:
            # This is a placeholder - actual implementation depends on Dabble app UI
            # You'll need to inspect the app to find the correct element locators
            
            # Example navigation flow (adjust based on actual app):
            # 1. Find search or user list
            # 2. Search for username
            # 3. Click on user profile
            
            # Wait for app to load
            WebDriverWait(self.driver, self.config['scraping']['timeout']).until(
                EC.presence_of_element_located((AppiumBy.ID, "com.dabble.app:id/main_container"))
            )
            
            # Search for user (example - adjust selectors)
            search_button = self.driver.find_element(AppiumBy.ID, "com.dabble.app:id/search_button")
            search_button.click()
            
            search_field = self.driver.find_element(AppiumBy.ID, "com.dabble.app:id/search_field")
            search_field.send_keys(username)
            
            # Click on user from results
            user_result = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((AppiumBy.XPATH, f"//android.widget.TextView[contains(@text, '{username}')]"))
            )
            user_result.click()
            
            logging.info(f"Successfully navigated to {username}'s profile")
            return True
            
        except (TimeoutException, NoSuchElementException) as e:
            logging.error(f"Failed to navigate to {username}'s profile: {e}")
            return False
    
    def navigate_to_last_10_tab(self) -> bool:
        """
        Navigate to the "Last 10" tab in user profile
        
        Returns:
            True if navigation successful, False otherwise
        """
        try:
            # Find and click "Last 10" tab (adjust selector based on actual app)
            last_10_tab = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((AppiumBy.XPATH, "//android.widget.TextView[contains(@text, 'Last 10')]"))
            )
            last_10_tab.click()
            
            # Wait for content to load
            time.sleep(2)
            
            logging.info("Successfully navigated to Last 10 tab")
            return True
            
        except (TimeoutException, NoSuchElementException) as e:
            logging.error(f"Failed to navigate to Last 10 tab: {e}")
            return False
    
    def extract_screen_text(self) -> str:
        """
        Extract all text from current screen
        
        Returns:
            Combined text from all screen elements
        """
        try:
            # Get page source and extract text elements
            page_source = self.driver.page_source
            
            # Find all text elements (adjust based on actual app structure)
            text_elements = self.driver.find_elements(AppiumBy.CLASS_NAME, "android.widget.TextView")
            
            # Combine all text
            screen_text = []
            for element in text_elements:
                try:
                    text = element.text.strip()
                    if text:
                        screen_text.append(text)
                except Exception:
                    continue
            
            combined_text = '\n'.join(screen_text)
            logging.debug(f"Extracted {len(screen_text)} text elements from screen")
            
            return combined_text
            
        except Exception as e:
            logging.error(f"Failed to extract screen text: {e}")
            return ""
    
    def scrape_user_bets(self, username: str) -> List[BetLeg]:
        """
        Main scraping logic for individual users
        
        Args:
            username: Username to scrape bets for
            
        Returns:
            List of BetLeg objects representing scraped bets
        """
        all_bets = []
        
        try:
            # Navigate to user profile
            if not self.navigate_to_user_profile(username):
                return all_bets
            
            # Navigate to Last 10 tab
            if not self.navigate_to_last_10_tab():
                return all_bets
            
            # Extract screen text
            screen_text = self.extract_screen_text()
            if not screen_text:
                logging.warning(f"No text extracted for user {username}")
                return all_bets
            
            # Parse bet data
            current_date = datetime.now().strftime('%Y-%m-%d')
            parsed_bets = self.bet_parser.parse_bet_text(screen_text, username, current_date)
            
            # Filter duplicates
            unique_bets = self.duplicate_detector.filter_duplicates(parsed_bets, username)
            
            all_bets.extend(unique_bets)
            
            logging.info(f"Scraped {len(unique_bets)} new bets for user {username}")
            
        except Exception as e:
            logging.error(f"Error scraping bets for user {username}: {e}")
        
        return all_bets
    
    def run_scraping_session(self) -> Dict:
        """
        Execute complete scraping session for all users
        
        Returns:
            Dictionary with session results and statistics
        """
        session_results = {
            'start_time': datetime.now().isoformat(),
            'users_processed': 0,
            'total_new_bets': 0,
            'user_results': {},
            'errors': []
        }
        
        try:
            # Connect to device
            if not self.connect_to_device():
                session_results['errors'].append("Failed to connect to Android device")
                return session_results
            
            # Get list of users to process
            users = self.user_manager.get_users()
            
            for username in users:
                try:
                    logging.info(f"Processing user: {username}")
                    
                    # Scrape bets for this user
                    user_bets = self.scrape_user_bets(username)
                    
                    # Save to Google Sheets
                    rows_added = 0
                    if user_bets:
                        rows_added = self.sheets_manager.append_bets(user_bets)
                    
                    # Update user tracking
                    self.user_manager.update_last_check(username)
                    self.user_manager.update_stats(username, len(user_bets))
                    
                    # Record results
                    session_results['user_results'][username] = {
                        'new_bets': len(user_bets),
                        'rows_added': rows_added,
                        'success': True
                    }
                    
                    session_results['users_processed'] += 1
                    session_results['total_new_bets'] += len(user_bets)
                    
                    # Delay between users to avoid rate limiting
                    if username != users[-1]:  # Don't delay after last user
                        time.sleep(self.config['scraping']['delay_between_users'])
                
                except Exception as e:
                    error_msg = f"Error processing user {username}: {e}"
                    logging.error(error_msg)
                    session_results['errors'].append(error_msg)
                    session_results['user_results'][username] = {
                        'new_bets': 0,
                        'rows_added': 0,
                        'success': False,
                        'error': str(e)
                    }
        
        finally:
            # Always disconnect from device
            self.disconnect_device()
        
        session_results['end_time'] = datetime.now().isoformat()
        
        # Send notification if configured
        self.send_completion_notification(session_results)
        
        return session_results
    
    def send_completion_notification(self, results: Dict):
        """Send notification when scraping session completes"""
        if not self.config['notifications']['email']['enabled']:
            return
        
        try:
            # Prepare email content
            subject = f"Dabble Scraper Results - {results['total_new_bets']} new bets"
            
            body = f"""
Dabble Scraping Session Complete

Session Summary:
- Start Time: {results['start_time']}
- End Time: {results['end_time']}
- Users Processed: {results['users_processed']}
- Total New Bets: {results['total_new_bets']}

User Results:
"""
            
            for username, user_result in results['user_results'].items():
                body += f"- {username}: {user_result['new_bets']} new bets"
                if not user_result['success']:
                    body += f" (ERROR: {user_result.get('error', 'Unknown error')})"
                body += "\n"
            
            if results['errors']:
                body += f"\nErrors:\n"
                for error in results['errors']:
                    body += f"- {error}\n"
            
            # Send email
            self.send_email(subject, body)
            
        except Exception as e:
            logging.error(f"Failed to send notification: {e}")
    
    def send_email(self, subject: str, body: str):
        """Send email notification"""
        email_config = self.config['notifications']['email']
        
        try:
            msg = MIMEMultipart()
            msg['From'] = email_config['sender_email']
            msg['To'] = email_config['recipient_email']
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['sender_email'], email_config['sender_password'])
            
            text = msg.as_string()
            server.sendmail(email_config['sender_email'], email_config['recipient_email'], text)
            server.quit()
            
            logging.info("Email notification sent successfully")
            
        except Exception as e:
            logging.error(f"Failed to send email: {e}")


class ScheduleManager:
    """Handles automation scheduling and triggers"""
    
    def __init__(self, scraper: DabbleScraper):
        self.scraper = scraper
        self.scheduler = BlockingScheduler()
    
    def schedule_daily_run(self, hour: int = 18, minute: int = 0):
        """
        Schedule daily scraping run
        
        Args:
            hour: Hour to run (24-hour format, default 18 = 6 PM)
            minute: Minute to run (default 0)
        """
        self.scheduler.add_job(
            func=self.scraper.run_scraping_session,
            trigger=CronTrigger(hour=hour, minute=minute),
            id='daily_scrape',
            name='Daily Dabble Scraping',
            replace_existing=True
        )
        
        logging.info(f"Scheduled daily scraping at {hour:02d}:{minute:02d}")
    
    def start_scheduler(self):
        """Start the scheduler (blocking operation)"""
        try:
            logging.info("Starting scheduler...")
            self.scheduler.start()
        except KeyboardInterrupt:
            logging.info("Scheduler stopped by user")
        except Exception as e:
            logging.error(f"Scheduler error: {e}")
        finally:
            self.scheduler.shutdown()
    
    def run_manual_scrape(self):
        """Run manual scraping session"""
        logging.info("Starting manual scraping session...")
        results = self.scraper.run_scraping_session()
        
        print("\n" + "="*50)
        print("SCRAPING SESSION RESULTS")
        print("="*50)
        print(f"Users Processed: {results['users_processed']}")
        print(f"Total New Bets: {results['total_new_bets']}")
        
        print("\nUser Results:")
        for username, user_result in results['user_results'].items():
            status = "✓" if user_result['success'] else "✗"
            print(f"  {status} {username}: {user_result['new_bets']} new bets")
            if not user_result['success']:
                print(f"    Error: {user_result.get('error', 'Unknown error')}")
        
        if results['errors']:
            print("\nSession Errors:")
            for error in results['errors']:
                print(f"  - {error}")
        
        print("="*50)
        
        return results


def main():
    """Main entry point for the application"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Dabble Betting History Scraper')
    parser.add_argument('--manual', action='store_true', help='Run manual scraping session')
    parser.add_argument('--schedule', action='store_true', help='Start scheduled scraping')
    parser.add_argument('--add-user', type=str, help='Add a new user to track')
    parser.add_argument('--remove-user', type=str, help='Remove a user from tracking')
    parser.add_argument('--list-users', action='store_true', help='List all tracked users')
    parser.add_argument('--config', type=str, default='config.json', help='Config file path')
    
    args = parser.parse_args()
    
    try:
        # Initialize scraper
        scraper = DabbleScraper(args.config)
        schedule_manager = ScheduleManager(scraper)
        
        if args.add_user:
            scraper.user_manager.add_user(args.add_user)
            print(f"Added user: {args.add_user}")
            
        elif args.remove_user:
            scraper.user_manager.remove_user(args.remove_user)
            print(f"Removed user: {args.remove_user}")
            
        elif args.list_users:
            users = scraper.user_manager.get_users()
            print("Tracked Users:")
            for i, user in enumerate(users, 1):
                last_check = scraper.user_manager.get_last_check(user)
                last_check_str = last_check.strftime('%Y-%m-%d %H:%M') if last_check else 'Never'
                print(f"  {i}. {user} (Last check: {last_check_str})")
                
        elif args.manual:
            schedule_manager.run_manual_scrape()
            
        elif args.schedule:
            schedule_manager.schedule_daily_run()
            schedule_manager.start_scheduler()
            
        else:
            print("No action specified. Use --help for available options.")
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logging.error(f"Application error: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()