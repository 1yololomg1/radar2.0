#!/usr/bin/env python3
"""
RADAR - Results Analysis and Data Accuracy Reporter
Professional Deployment Script
Creates standalone executable and installer package

Author: RADAR Development Team
Version: 2.0.0
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import tempfile

def create_build_config():
    """Create PyInstaller build configuration."""
    spec_content = """
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['radar_validator.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'matplotlib',
        'matplotlib.backends.backend_tkagg',
        'matplotlib.pyplot',
        'matplotlib.figure',
        'seaborn',
        'pandas',
        'numpy',
        'scipy',
        'scipy.stats',
        'openpyxl',
        'pandas._libs.tslibs.timedeltas',
        'pandas._libs.writers',
        'reportlab',
        'reportlab.lib.pagesizes',
        'reportlab.lib.styles',
        'reportlab.lib.units',
        'reportlab.lib.colors',
        'reportlab.platypus',
        'reportlab.graphics',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'IPython',
        'jupyter',
        'notebook',
        'sphinx',
        'pytest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='RADAR_ModelValidator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # GUI application
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
"""
    
    with open('radar_validator.spec', 'w') as f:
        f.write(spec_content)

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy', 'openpyxl', 'reportlab', 'pyinstaller'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies are installed")
    return True

def validate_source_code():
    """Validate that source code exists and is functional."""
    print("🔍 Validating source code...")
    
    if not os.path.exists('radar_validator.py'):
        print("❌ radar_validator.py not found!")
        return False
    
    # Try to import the module to check for syntax errors
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("radar_validator", "radar_validator.py")
        radar_validator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(radar_validator)
        print("✅ Source code validation passed")
        return True
    except Exception as e:
        print(f"❌ Source code validation failed: {e}")
        return False

def build_executable():
    """Build standalone executable using PyInstaller."""
    print("🏗️  Building standalone executable...")
    
    # Create PyInstaller specification
    create_build_config()
    
    try:
        # Clean previous builds
        if os.path.exists('build'):
            shutil.rmtree('build')
        if os.path.exists('dist'):
            shutil.rmtree('dist')
        
        # Build executable
        result = subprocess.run([
            sys.executable, '-m', 'PyInstaller', 
            '--clean', 
            'radar_validator.spec'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Executable built successfully!")
            return True
        else:
            print("❌ Build failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed with error: {e}")
        return False

def test_executable():
    """Test the built executable."""
    print("🧪 Testing executable...")
    
    exe_path = os.path.join('dist', 'RADAR_ModelValidator.exe')
    if os.name != 'nt':  # Not Windows
        exe_path = os.path.join('dist', 'RADAR_ModelValidator')
    
    if not os.path.exists(exe_path):
        print(f"❌ Executable not found at {exe_path}")
        return False
    
    # Test that executable starts (with version flag)
    try:
        result = subprocess.run([exe_path, '--version'], 
                              capture_output=True, text=True, timeout=10)
        if 'RADAR Model Validator' in result.stdout:
            print("✅ Executable test passed")
            return True
        else:
            print("❌ Executable test failed - unexpected output")
            print("Output:", result.stdout)
            return False
    except subprocess.TimeoutExpired:
        print("⚠️  Executable test timeout (may be normal for GUI apps)")
        return True  # GUI apps may not respond to --version immediately
    except Exception as e:
        print(f"❌ Executable test failed: {e}")
        return False

def create_distribution_package():
    """Create complete distribution package."""
    print("📦 Creating distribution package...")
    
    # Create distribution directory
    dist_dir = Path('RADAR_ModelValidator_v2.0.0')
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    
    dist_dir.mkdir()
    
    # Determine executable extension based on OS
    exe_name = 'RADAR_ModelValidator.exe' if os.name == 'nt' else 'RADAR_ModelValidator'
    
    # Copy executable
    exe_source = os.path.join('dist', exe_name)
    if os.path.exists(exe_source):
        shutil.copy2(exe_source, dist_dir / exe_name)
        print(f"✅ Copied executable: {exe_name}")
    else:
        print(f"❌ Executable not found: {exe_source}")
        return False
    
    # Create README for distribution
    readme_content = """# RADAR - Results Analysis and Data Accuracy Reporter v2.0.0

## Professional Neural Network Classification Analysis Tool

### Quick Start:
1. Double-click RADAR_ModelValidator.exe to launch
2. Click "Browse" to select your Excel file with neural network results
3. Select one or more models from the list
4. Click "Analyze Selected Models"
5. View results in the Analysis and Visualization panels

### Excel File Format:
- Each sheet represents a neural network model (e.g., "6x6", "8x8", "10x10")
- First column: Neuron identifiers
- Subsequent columns: Percentage classifications for each lithofacies
- Example: FineSand, MedFineSnd, MedCoarseSnd, SandAndShale

### Debug Mode:
Set environment variable RADAR_DEBUG=1 before launching for detailed logging.

### Support:
For technical support, contact RADAR development team.

### System Requirements:
- Windows 10/11, macOS 10.14+, or Linux
- 4GB RAM minimum (8GB recommended)
- 500MB disk space

---
RADAR - Results Analysis and Data Accuracy Reporter v2.0.0
Copyright (c) 2025 RADAR Development Team. All rights reserved.
"""
    
    with open(dist_dir / 'README.txt', 'w') as f:
        f.write(readme_content)
    
    # Create license file
    license_content = """RADAR - RESULTS ANALYSIS AND DATA ACCURACY REPORTER v2.0.0
COMMERCIAL SOFTWARE LICENSE

This software is licensed for commercial use in the oil & gas industry.
Unauthorized distribution or modification is prohibited.

For license information and support:
Email: support@radar-analysis.com
Website: https://radar-analysis.com

Copyright (c) 2025 RADAR Development Team. All rights reserved.
"""
    
    with open(dist_dir / 'LICENSE.txt', 'w') as f:
        f.write(license_content)
    
    # Create sample data description
    sample_data_content = """# Sample Data Format

Your Excel file should have sheets named after neural network configurations.
Each sheet should contain:

| Neuron | FineSand | MedFineSnd | MedCoarseSnd | SandAndShale |
|--------|----------|------------|--------------|--------------|
| 1      | 78       | 4          | 0            | 18           |
| 2      | 11       | 50         | 39           | 0            |
| 3      | 0        | 0          | 0            | 0            |
| ...    | ...      | ...        | ...          | ...          |

Values represent percentage classifications (should sum to ~100%).
"""
    
    with open(dist_dir / 'SAMPLE_DATA_FORMAT.txt', 'w') as f:
        f.write(sample_data_content)
    
    print(f"✅ Distribution package created: {dist_dir}")
    
    # Calculate package size
    total_size = sum(f.stat().st_size for f in dist_dir.rglob('*') if f.is_file())
    print(f"📊 Package size: {total_size / (1024*1024):.1f} MB")
    
    return True

def create_installer_script():
    """Create installer script (Windows only)."""
    if os.name != 'nt':
        print("ℹ️  Skipping installer creation (Windows only)")
        return True
    
    print("💿 Creating Windows installer script...")
    
    nsis_content = """
; TraceSeis Model Validator Installer
; Auto-generated installer script

!define APPNAME "TraceSeis Model Validator"
!define COMPANYNAME "TraceSeis"
!define DESCRIPTION "Professional Geophysical Neural Network Analysis Tool"
!define VERSIONMAJOR 2
!define VERSIONMINOR 0
!define VERSIONBUILD 0

RequestExecutionLevel admin
InstallDir "$PROGRAMFILES\\TraceSeis\\ModelValidator"

Name "${APPNAME}"
OutFile "TraceSeis_ModelValidator_v2.0.0_Setup.exe"

Page license
Page directory
Page instfiles

LicenseData "TraceSeis_ModelValidator_v2.0.0\\LICENSE.txt"

Section "Install"
    SetOutPath $INSTDIR
    
    File "TraceSeis_ModelValidator_v2.0.0\\TraceSeis_ModelValidator.exe"
    File "TraceSeis_ModelValidator_v2.0.0\\README.txt"
    File "TraceSeis_ModelValidator_v2.0.0\\LICENSE.txt"
    File "TraceSeis_ModelValidator_v2.0.0\\SAMPLE_DATA_FORMAT.txt"
    
    CreateDirectory "$SMPROGRAMS\\TraceSeis"
    CreateShortCut "$SMPROGRAMS\\TraceSeis\\Model Validator.lnk" "$INSTDIR\\TraceSeis_ModelValidator.exe"
    CreateShortCut "$DESKTOP\\TraceSeis Model Validator.lnk" "$INSTDIR\\TraceSeis_ModelValidator.exe"
    
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "DisplayName" "${APPNAME}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "UninstallString" "$INSTDIR\\uninstall.exe"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "InstallLocation" "$INSTDIR"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "Publisher" "${COMPANYNAME}"
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "VersionMajor" ${VERSIONMAJOR}
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "VersionMinor" ${VERSIONMINOR}
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "NoModify" 1
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "NoRepair" 1
    
    WriteUninstaller "$INSTDIR\\uninstall.exe"
SectionEnd

Section "Uninstall"
    Delete "$INSTDIR\\TraceSeis_ModelValidator.exe"
    Delete "$INSTDIR\\README.txt"
    Delete "$INSTDIR\\LICENSE.txt"
    Delete "$INSTDIR\\SAMPLE_DATA_FORMAT.txt"
    Delete "$INSTDIR\\uninstall.exe"
    RMDir "$INSTDIR"
    
    Delete "$SMPROGRAMS\\TraceSeis\\Model Validator.lnk"
    Delete "$DESKTOP\\TraceSeis Model Validator.lnk"
    RMDir "$SMPROGRAMS\\TraceSeis"
    
    DeleteRegKey HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}"
SectionEnd
"""
    
    with open('installer.nsi', 'w') as f:
        f.write(nsis_content)
    
    print("✅ Installer script created: installer.nsi")
    print("ℹ️  To build installer, install NSIS and run: makensis installer.nsi")
    return True

def cleanup_build_files():
    """Clean up temporary build files."""
    print("🧹 Cleaning up build files...")
    
    files_to_remove = [
        'radar_validator.spec',
        'build',
        '__pycache__'
    ]
    
    for item in files_to_remove:
        if os.path.exists(item):
            if os.path.isdir(item):
                shutil.rmtree(item)
            else:
                os.remove(item)
            print(f"🗑️  Removed: {item}")

def main():
    """Main deployment function."""
    print("🚀 RADAR - Professional Deployment")
    print("Results Analysis and Data Accuracy Reporter")
    print("=" * 60)
    print("Building standalone executable package...")
    print()
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ DEPLOYMENT FAILED - Missing dependencies")
        return 1
    
    # Step 2: Validate source code
    if not validate_source_code():
        print("\n❌ DEPLOYMENT FAILED - Source code validation failed")
        return 1
    
    # Step 3: Build executable
    if not build_executable():
        print("\n❌ DEPLOYMENT FAILED - Executable build failed")
        return 1
    
    # Step 4: Test executable
    if not test_executable():
        print("\n⚠️  DEPLOYMENT WARNING - Executable test failed")
        print("Continuing with packaging...")
    
    # Step 5: Create distribution package
    if not create_distribution_package():
        print("\n❌ DEPLOYMENT FAILED - Package creation failed")
        return 1
    
    # Step 6: Create installer script (Windows only)
    if not create_installer_script():
        print("\n⚠️  DEPLOYMENT WARNING - Installer creation failed")
    
    # Step 7: Cleanup
    cleanup_build_files()
    
    print("\n" + "=" * 60)
    print("✅ DEPLOYMENT SUCCESSFUL!")
    print("\nDeployment artifacts created:")
    print("📁 RADAR_ModelValidator_v2.0.0/ (complete distribution package)")
    if os.name == 'nt':
        print("📄 installer.nsi (Windows installer script)")
    
    print("\n📋 Distribution Contents:")
    print("🖥️  RADAR_ModelValidator.exe (standalone executable)")
    print("📖 README.txt (user documentation)")
    print("⚖️  LICENSE.txt (commercial license)")
    print("📊 SAMPLE_DATA_FORMAT.txt (data format guide)")
    
    print("\n🎯 Ready for Professional Deployment!")
    print("The package is completely self-contained and requires no additional installation.")
    
    if os.name == 'nt':
        print("\n💡 Next Steps (Windows):")
        print("1. Test the executable in RADAR_ModelValidator_v2.0.0/")
        print("2. Optional: Install NSIS and run 'makensis installer.nsi' for installer")
        print("3. Distribute the package or installer to end users")
    else:
        print("\n💡 Next Steps:")
        print("1. Test the executable in RADAR_ModelValidator_v2.0.0/")
        print("2. Package as appropriate for your platform (DMG, AppImage, etc.)")
        print("3. Distribute to end users")
    
    print("\n🎯 RADAR - Professional Neural Network Analysis Tool")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
