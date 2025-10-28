#!/usr/bin/env python3
"""
DABBLE SCRAPER TEST SUITE
=========================

Comprehensive test suite for validating all components of the Dabble scraper.
Tests individual components and end-to-end functionality to ensure reliability.

TEST CATEGORIES:
1. Text Parser Tests - Validate bet text parsing accuracy
2. Google Sheets Tests - Test spreadsheet integration
3. User Management Tests - Verify user tracking functionality  
4. Duplicate Detection Tests - Ensure no duplicate entries
5. Configuration Tests - Validate config file handling
6. Android Connection Tests - Test device connectivity
7. Integration Tests - End-to-end workflow validation

USAGE:
- Run all tests: python test_scraper.py
- Run specific category: python test_scraper.py --category parser
- Run with verbose output: python test_scraper.py --verbose
- Generate test report: python test_scraper.py --report
"""

import os
import sys
import json
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dabble_scraper import (
    BetLeg, BetParser, GoogleSheetsManager, DuplicateDetector, 
    UserManager, DabbleScraper
)


class TestBetParser(unittest.TestCase):
    """Test bet text parsing functionality"""
    
    def setUp(self):
        self.parser = BetParser()
    
    def test_parse_simple_bet(self):
        """Test parsing a simple bet leg"""
        bet_text = """
        Isiah Pacheco (KC-RB)
        ↑ MORE 47.5 Rushing Yards
        WAS @ KC
        WON
        """
        
        bets = self.parser.parse_bet_text(bet_text, "TestUser", "2024-10-28")
        
        self.assertEqual(len(bets), 1)
        bet = bets[0]
        self.assertEqual(bet.player, "Isiah Pacheco")
        self.assertEqual(bet.sport, "NFL")
        self.assertEqual(bet.stat_type, "Rushing_Yards")
        self.assertEqual(bet.predicted, "OVER")
        self.assertEqual(bet.line, 47.5)
        self.assertEqual(bet.result, "WON")
        self.assertTrue(bet.correct)
    
    def test_parse_under_bet(self):
        """Test parsing an under bet"""
        bet_text = """
        Patrick Mahomes (KC-QB)
        ↓ LESS 2.5 Touchdowns
        WAS @ KC
        LOST
        """
        
        bets = self.parser.parse_bet_text(bet_text, "TestUser", "2024-10-28")
        
        self.assertEqual(len(bets), 1)
        bet = bets[0]
        self.assertEqual(bet.predicted, "UNDER")
        self.assertEqual(bet.line, 2.5)
        self.assertEqual(bet.result, "LOST")
        self.assertFalse(bet.correct)
    
    def test_parse_nba_bet(self):
        """Test parsing NBA bet"""
        bet_text = """
        LeBron James (LAL-SF)
        ↑ MORE 25.5 Points
        LAL vs BOS
        PENDING
        """
        
        bets = self.parser.parse_bet_text(bet_text, "TestUser", "2024-10-28")
        
        self.assertEqual(len(bets), 1)
        bet = bets[0]
        self.assertEqual(bet.sport, "NBA")
        self.assertEqual(bet.stat_type, "Points")
        self.assertEqual(bet.result, "PENDING")
        self.assertIsNone(bet.correct)
    
    def test_parse_multiple_bets(self):
        """Test parsing multiple bet legs from parlay"""
        bet_text = """
        Isiah Pacheco (KC-RB)
        ↑ MORE 47.5 Rushing Yards
        WAS @ KC
        WON
        
        Travis Kelce (KC-TE)
        ↑ MORE 65.5 Receiving Yards
        WAS @ KC
        WON
        """
        
        bets = self.parser.parse_bet_text(bet_text, "TestUser", "2024-10-28")
        
        self.assertEqual(len(bets), 2)
        self.assertEqual(bets[0].player, "Isiah Pacheco")
        self.assertEqual(bets[1].player, "Travis Kelce")
    
    def test_stat_type_normalization(self):
        """Test stat type normalization"""
        test_cases = [
            ("rushing yards", "Rushing_Yards"),
            ("receiving yards", "Receiving_Yards"),
            ("points", "Points"),
            ("rebounds", "Rebounds"),
            ("home runs", "Home_Runs")
        ]
        
        for input_stat, expected in test_cases:
            result = self.parser.normalize_stat_type(input_stat)
            self.assertEqual(result, expected)
    
    def test_sport_detection(self):
        """Test sport detection from stat types"""
        test_cases = [
            ("rushing yards", "NFL"),
            ("points", "NBA"),
            ("goals", "NHL"),
            ("home runs", "MLB")
        ]
        
        for stat_text, expected_sport in test_cases:
            result = self.parser.detect_sport(stat_text)
            self.assertEqual(result, expected_sport)


class TestUserManager(unittest.TestCase):
    """Test user management functionality"""
    
    def setUp(self):
        # Create temporary file for testing
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        self.temp_file.close()
        self.user_manager = UserManager(self.temp_file.name)
    
    def tearDown(self):
        # Clean up temporary file
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_add_user(self):
        """Test adding new user"""
        initial_count = len(self.user_manager.get_users())
        self.user_manager.add_user("NewUser")
        
        users = self.user_manager.get_users()
        self.assertEqual(len(users), initial_count + 1)
        self.assertIn("NewUser", users)
    
    def test_remove_user(self):
        """Test removing user"""
        self.user_manager.add_user("TempUser")
        initial_count = len(self.user_manager.get_users())
        
        self.user_manager.remove_user("TempUser")
        users = self.user_manager.get_users()
        
        self.assertEqual(len(users), initial_count - 1)
        self.assertNotIn("TempUser", users)
    
    def test_update_last_check(self):
        """Test updating last check timestamp"""
        username = "TestUser"
        self.user_manager.update_last_check(username)
        
        last_check = self.user_manager.get_last_check(username)
        self.assertIsNotNone(last_check)
        self.assertIsInstance(last_check, datetime)
    
    def test_update_stats(self):
        """Test updating user statistics"""
        username = "TestUser"
        self.user_manager.update_stats(username, 5)
        
        stats = self.user_manager.users_data['stats'][username]
        self.assertEqual(stats['total_bets'], 5)
        self.assertEqual(stats['last_scrape_count'], 5)


class TestDuplicateDetector(unittest.TestCase):
    """Test duplicate detection functionality"""
    
    def setUp(self):
        # Mock GoogleSheetsManager
        self.mock_sheets = Mock()
        self.detector = DuplicateDetector(self.mock_sheets)
    
    def test_create_bet_signature(self):
        """Test bet signature creation"""
        bet_data = {
            'Date': '2024-10-28',
            'User': 'TestUser',
            'Player': 'Test Player',
            'Stat_Type': 'Rushing_Yards',
            'Predicted': 'OVER',
            'Line': 47.5
        }
        
        signature = self.detector.create_bet_signature(bet_data)
        expected = "2024-10-28|testuser|test player|rushing_yards|over|47.5"
        self.assertEqual(signature, expected)
    
    def test_filter_duplicates(self):
        """Test filtering duplicate bets"""
        # Mock existing bets
        existing_bets = [
            {
                'Date': '2024-10-28',
                'User': 'TestUser',
                'Player': 'Player1',
                'Stat_Type': 'Rushing_Yards',
                'Predicted': 'OVER',
                'Line': 47.5
            }
        ]
        self.mock_sheets.get_existing_bets.return_value = existing_bets
        
        # Create new bets (one duplicate, one unique)
        new_bets = [
            BetLeg(
                date='2024-10-28',
                username='TestUser',
                sport='NFL',
                player='Player1',  # Duplicate
                stat_type='Rushing_Yards',
                predicted='OVER',
                line=47.5,
                result='WON'
            ),
            BetLeg(
                date='2024-10-28',
                username='TestUser',
                sport='NFL',
                player='Player2',  # Unique
                stat_type='Receiving_Yards',
                predicted='OVER',
                line=65.5,
                result='WON'
            )
        ]
        
        unique_bets = self.detector.filter_duplicates(new_bets, 'TestUser')
        
        self.assertEqual(len(unique_bets), 1)
        self.assertEqual(unique_bets[0].player, 'Player2')


class TestGoogleSheetsManager(unittest.TestCase):
    """Test Google Sheets integration (mocked)"""
    
    def setUp(self):
        # Mock gspread and credentials
        with patch('gspread.authorize'), \
             patch('google.oauth2.service_account.Credentials.from_service_account_file'):
            
            # Create temporary credentials file
            self.temp_creds = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            json.dump({'type': 'service_account'}, self.temp_creds)
            self.temp_creds.close()
            
            self.sheets_manager = GoogleSheetsManager(
                self.temp_creds.name,
                'Test Spreadsheet'
            )
            
            # Mock the sheet object
            self.sheets_manager.sheet = Mock()
    
    def tearDown(self):
        if os.path.exists(self.temp_creds.name):
            os.unlink(self.temp_creds.name)
    
    def test_append_bets(self):
        """Test appending bets to spreadsheet"""
        bet_legs = [
            BetLeg(
                date='2024-10-28',
                username='TestUser',
                sport='NFL',
                player='Test Player',
                stat_type='Rushing_Yards',
                predicted='OVER',
                line=47.5,
                result='WON'
            )
        ]
        
        # Mock successful append
        self.sheets_manager.sheet.append_rows.return_value = None
        
        result = self.sheets_manager.append_bets(bet_legs)
        
        self.assertEqual(result, 1)
        self.sheets_manager.sheet.append_rows.assert_called_once()
    
    def test_get_existing_bets(self):
        """Test retrieving existing bets"""
        # Mock spreadsheet data
        mock_records = [
            {
                'Date': '2024-10-28T10:00:00',
                'User': 'TestUser',
                'Player': 'Test Player',
                'Stat_Type': 'Rushing_Yards'
            }
        ]
        self.sheets_manager.sheet.get_all_records.return_value = mock_records
        
        result = self.sheets_manager.get_existing_bets('TestUser')
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['User'], 'TestUser')


class TestConfigurationHandling(unittest.TestCase):
    """Test configuration file handling"""
    
    def test_config_loading(self):
        """Test loading configuration from file"""
        # Create temporary config file
        temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        config_data = {
            'appium': {
                'server_url': 'http://test:4723/wd/hub'
            }
        }
        json.dump(config_data, temp_config)
        temp_config.close()
        
        try:
            with patch('dabble_scraper.DabbleScraper.__init__') as mock_init:
                mock_init.return_value = None
                scraper = DabbleScraper.__new__(DabbleScraper)
                config = scraper.load_config(temp_config.name)
                
                self.assertEqual(config['appium']['server_url'], 'http://test:4723/wd/hub')
        
        finally:
            os.unlink(temp_config.name)


class TestIntegration(unittest.TestCase):
    """Integration tests for end-to-end functionality"""
    
    @patch('dabble_scraper.webdriver.Remote')
    @patch('dabble_scraper.GoogleSheetsManager')
    def test_scraping_workflow(self, mock_sheets, mock_driver):
        """Test complete scraping workflow"""
        # Mock driver and sheets manager
        mock_driver_instance = Mock()
        mock_driver.return_value = mock_driver_instance
        
        mock_sheets_instance = Mock()
        mock_sheets.return_value = mock_sheets_instance
        
        # Create temporary config
        temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump({}, temp_config)
        temp_config.close()
        
        try:
            scraper = DabbleScraper(temp_config.name)
            
            # Mock successful connection
            scraper.connect_to_device = Mock(return_value=True)
            scraper.navigate_to_user_profile = Mock(return_value=True)
            scraper.navigate_to_last_10_tab = Mock(return_value=True)
            scraper.extract_screen_text = Mock(return_value="Test bet text")
            
            # Mock bet parsing
            mock_bet = BetLeg(
                date='2024-10-28',
                username='TestUser',
                sport='NFL',
                player='Test Player',
                stat_type='Rushing_Yards',
                predicted='OVER',
                line=47.5,
                result='WON'
            )
            scraper.bet_parser.parse_bet_text = Mock(return_value=[mock_bet])
            
            # Mock duplicate detection
            scraper.duplicate_detector.filter_duplicates = Mock(return_value=[mock_bet])
            
            # Run scraping for one user
            result = scraper.scrape_user_bets('TestUser')
            
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].player, 'Test Player')
        
        finally:
            os.unlink(temp_config.name)


def run_test_suite(category=None, verbose=False):
    """Run the test suite with optional filtering"""
    
    # Test categories mapping
    categories = {
        'parser': TestBetParser,
        'users': TestUserManager,
        'duplicates': TestDuplicateDetector,
        'sheets': TestGoogleSheetsManager,
        'config': TestConfigurationHandling,
        'integration': TestIntegration
    }
    
    # Create test suite
    suite = unittest.TestSuite()
    
    if category and category in categories:
        # Run specific category
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(categories[category]))
        print(f"Running {category} tests...")
    else:
        # Run all tests
        for test_class in categories.values():
            suite.addTest(unittest.TestLoader().loadTestsFromTestCase(test_class))
        print("Running all tests...")
    
    # Configure test runner
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    
    # Run tests
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall: {'PASS' if success else 'FAIL'}")
    
    return success


def generate_test_report():
    """Generate detailed test report"""
    print("Generating test report...")
    
    # Run tests and capture results
    import io
    import contextlib
    
    output = io.StringIO()
    
    with contextlib.redirect_stdout(output):
        success = run_test_suite(verbose=True)
    
    test_output = output.getvalue()
    
    # Generate report
    report = {
        'timestamp': datetime.now().isoformat(),
        'success': success,
        'test_output': test_output,
        'environment': {
            'python_version': sys.version,
            'platform': sys.platform
        }
    }
    
    # Save report
    report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Test report saved to: {report_file}")
    return report_file


def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Dabble Scraper Test Suite')
    parser.add_argument('--category', choices=['parser', 'users', 'duplicates', 'sheets', 'config', 'integration'],
                       help='Run specific test category')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--report', action='store_true', help='Generate test report')
    
    args = parser.parse_args()
    
    print("="*60)
    print("DABBLE SCRAPER TEST SUITE")
    print("="*60)
    
    if args.report:
        generate_test_report()
    else:
        success = run_test_suite(args.category, args.verbose)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()