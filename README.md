# Dabble Betting History Scraper

An automated data collection tool for extracting betting history from the Dabble sports betting app and organizing it into spreadsheets for statistical analysis.

## Overview

This tool automates the process of collecting betting data from user profiles in the Dabble mobile app, parsing individual bet legs, and storing the data in Google Sheets for analysis. It handles duplicate detection, user management, scheduling, and provides comprehensive error handling.

## Features

- **Automated Mobile App Scraping**: Uses Appium to interact with Android Dabble app
- **Intelligent Text Parsing**: Extracts structured data from bet text using regex patterns
- **Google Sheets Integration**: Automatically stores data in organized spreadsheets
- **Duplicate Detection**: Prevents duplicate entries using multiple matching strategies
- **User Management**: Track multiple users with individual statistics and timestamps
- **Scheduling**: Daily automated runs or manual execution
- **Multi-Sport Support**: Handles NFL, NBA, NHL, and MLB betting data
- **Error Handling**: Comprehensive logging and error recovery
- **Notifications**: Email alerts for scraping completion and errors

## System Architecture

### Core Components

- **DabbleScraper**: Main orchestration class handling app automation
- **BetParser**: Text parsing engine for extracting bet data from screen text
- **GoogleSheetsManager**: Handles all Google Sheets operations and data storage
- **UserManager**: Manages user lists and tracking state
- **DuplicateDetector**: Prevents duplicate bet entries
- **ScheduleManager**: Handles automation scheduling and triggers

### Data Flow

1. Load user list and last check timestamps
2. For each user: Navigate to profile → "Last 10" tab
3. Extract screen text and parse bet legs
4. Filter out duplicates based on date/user/bet content
5. Append new bets to Google Sheets
6. Update user tracking state and timestamps
7. Send completion notifications

## Installation

### Prerequisites

- Python 3.8 or higher
- Node.js and npm (for Appium server)
- Android SDK with ADB tools
- Android device with USB debugging enabled
- Google Cloud service account for Sheets API access

### Quick Setup

1. **Clone and install dependencies:**
```bash
git clone <repository-url>
cd dabble-scraper
python setup_environment.py
```

2. **Manual setup (if automated setup fails):**
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Appium server
npm install -g appium
appium driver install uiautomator2

# Verify Android tools
adb devices
```

3. **Configure Google Sheets:**
   - Create Google Cloud project
   - Enable Google Sheets API and Google Drive API
   - Create service account and download JSON credentials
   - Rename credentials file to `google_credentials.json`
   - Share target spreadsheet with service account email

4. **Configure Dabble app:**
```bash
# Inspect app to find correct package name and UI elements
python app_inspector.py

# Update config.json with correct app details
```

## Configuration

### Main Configuration (`config.json`)

```json
{
  "appium": {
    "server_url": "http://localhost:4723/wd/hub",
    "desired_capabilities": {
      "platformName": "Android",
      "deviceName": "Android Device",
      "appPackage": "com.dabble.betting",
      "appActivity": ".MainActivity",
      "automationName": "UiAutomator2"
    }
  },
  "google_sheets": {
    "credentials_path": "google_credentials.json",
    "spreadsheet_name": "Dabble Betting History"
  },
  "scraping": {
    "timeout": 30,
    "retry_attempts": 3,
    "delay_between_users": 2
  }
}
```

### User Management (`users.json`)

```json
{
  "users": [
    "Durag",
    "Reggie",
    "RayWitDaLocks", 
    "SportsAlmanac"
  ],
  "last_check": {},
  "stats": {}
}
```

## Usage

### Command Line Interface

```bash
# Run manual scraping session
python dabble_scraper.py --manual

# Start scheduled daily scraping (6 PM)
python dabble_scraper.py --schedule

# User management
python dabble_scraper.py --add-user "NewUser"
python dabble_scraper.py --remove-user "OldUser"
python dabble_scraper.py --list-users

# App inspection (find UI elements)
python app_inspector.py

# Run tests
python test_scraper.py
python test_scraper.py --category parser --verbose
```

### Programmatic Usage

```python
from dabble_scraper import DabbleScraper, ScheduleManager

# Initialize scraper
scraper = DabbleScraper('config.json')

# Run manual session
results = scraper.run_scraping_session()

# Set up scheduling
scheduler = ScheduleManager(scraper)
scheduler.schedule_daily_run(hour=18, minute=0)
scheduler.start_scheduler()
```

## Data Structure

### Extracted Data Fields

Each bet leg is extracted with the following fields:

| Field | Description | Example |
|-------|-------------|---------|
| Date | Date bet was placed | 2024-10-28 |
| User | Username who posted bet | Durag |
| Sport | Sport category | NFL |
| Player | Player name | Isiah Pacheco |
| Stat_Type | Type of statistic | Rushing_Yards |
| Predicted | Over/Under prediction | OVER |
| Line | Betting line value | 47.5 |
| Result | Bet outcome | WON |
| Correct | Boolean result | True |

### Example Bet Parsing

**Input Text:**
```
Isiah Pacheco (KC-RB)
↑ MORE 47.5 Rushing Yards
WAS @ KC
WON
```

**Extracted Data:**
```json
{
  "date": "2024-10-28",
  "username": "Durag", 
  "sport": "NFL",
  "player": "Isiah Pacheco",
  "stat_type": "Rushing_Yards",
  "predicted": "OVER",
  "line": 47.5,
  "result": "WON",
  "correct": true
}
```

## App Inspection

Use the app inspector to identify correct UI elements:

```bash
python app_inspector.py
```

**Interactive Commands:**
- `capture` - Take screenshot and analyze elements
- `elements` - List all UI elements on screen
- `profile` - Search for profile-related elements
- `text <search>` - Find elements containing text
- `click <resource_id>` - Click element by ID
- `back` / `home` - Navigation controls

## Testing

### Run Test Suite

```bash
# Run all tests
python test_scraper.py

# Run specific test categories
python test_scraper.py --category parser
python test_scraper.py --category sheets --verbose

# Generate test report
python test_scraper.py --report
```

### Test Categories

- **Parser Tests**: Validate bet text parsing accuracy
- **Sheets Tests**: Test Google Sheets integration
- **User Tests**: Verify user management functionality
- **Duplicate Tests**: Ensure no duplicate entries
- **Config Tests**: Validate configuration handling
- **Integration Tests**: End-to-end workflow validation

## Troubleshooting

### Common Issues

**1. Device Connection Failed**
```bash
# Check device connection
adb devices

# Restart ADB server
adb kill-server && adb start-server

# Verify USB debugging is enabled
```

**2. App Package Not Found**
```bash
# Find installed apps
adb shell pm list packages | grep -i dabble

# Use app inspector to identify correct package
python app_inspector.py
```

**3. Google Sheets Authentication**
- Verify service account credentials are correct
- Check that spreadsheet is shared with service account email
- Ensure Google Sheets and Drive APIs are enabled

**4. Element Not Found Errors**
- Use app inspector to find current element selectors
- Update config.json with correct resource IDs
- Check if app UI has changed

**5. Parsing Errors**
- Verify bet text format matches expected patterns
- Check logs for detailed parsing information
- Test with known good bet examples

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check log files:
- `dabble_scraper.log` - Main application logs
- `logs/` - Additional debug information
- `screenshots/` - Error screenshots

## Scheduling

### Daily Automation

```bash
# Start scheduled scraping (runs daily at 6 PM)
python dabble_scraper.py --schedule
```

### Custom Scheduling

```python
from dabble_scraper import ScheduleManager

scheduler = ScheduleManager(scraper)

# Custom time (e.g., 9:30 AM)
scheduler.schedule_daily_run(hour=9, minute=30)

# Multiple daily runs
scheduler.scheduler.add_job(
    func=scraper.run_scraping_session,
    trigger='cron',
    hour='6,18',  # 6 AM and 6 PM
    id='twice_daily'
)
```

### System Service (Linux)

Create systemd service for automatic startup:

```ini
# /etc/systemd/system/dabble-scraper.service
[Unit]
Description=Dabble Betting Scraper
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/scraper
ExecStart=/usr/bin/python3 dabble_scraper.py --schedule
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable dabble-scraper.service
sudo systemctl start dabble-scraper.service
```

## Security Considerations

- **Credentials**: Store Google credentials securely, never commit to version control
- **Device Access**: Ensure Android device is trusted and secure
- **Network**: Use secure connections for API access
- **Data**: Implement proper data retention and privacy policies
- **Rate Limiting**: Respect app terms of service and implement delays

## Performance Optimization

- **Parallel Processing**: Process multiple users concurrently (if permitted)
- **Caching**: Cache frequently accessed data and configurations
- **Batch Operations**: Use batch Google Sheets operations for efficiency
- **Error Recovery**: Implement smart retry logic with exponential backoff
- **Resource Management**: Properly close connections and clean up resources

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Run tests (`python test_scraper.py`)
4. Commit changes (`git commit -am 'Add new feature'`)
5. Push to branch (`git push origin feature/new-feature`)
6. Create Pull Request

## License

This project is for educational and personal use only. Ensure compliance with Dabble's terms of service and applicable laws regarding data scraping and automation.

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review log files for detailed error information
3. Run the test suite to identify component issues
4. Use the app inspector to debug UI element problems

## Changelog

### Version 1.0.0
- Initial release with core scraping functionality
- Google Sheets integration
- User management and scheduling
- Comprehensive test suite
- App inspection utilities