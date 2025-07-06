import sys
import os
import subprocess
import pkg_resources
from pathlib import Path
import importlib.util
from packaging import version

# Required packages with minimum versions
REQUIRED_PACKAGES = {
    'vectorbt': '0.27.3',
    'deap': '1.4.3',
    'yfinance': '0.2.61',
    'SQLAlchemy': '2.0.41',
    'pandas': '2.2.3',
}

# Template for engine.py
ENGINE_TEMPLATE = '''from sqlalchemy import create_engine as _sa_create_engine

import pandas as pd

username = "username"
password = "password"

def create_engine():
    return _sa_create_engine(f"postgresql+psycopg://{username}:{password}@localhost:5432/letf_data")

def connect(engine, table_name = "test_data"):
    return pd.read_sql(f"SELECT * FROM {table_name}", engine)
'''


class SetupValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.success_messages = []
        
    def check_python_version(self):
        """Check if Python version is 3.11.9 (recommended) or compatible."""
        current_version = sys.version_info
        recommended_version = (3, 11, 9)
        
        if current_version[:3] == recommended_version:
            self.success_messages.append(f"Python version {sys.version.split()[0]} (recommended)")
        elif current_version[:2] == (3, 11):
            self.warnings.append(f"Python {sys.version.split()[0]} detected. Recommended: 3.11.9")
        else:
            self.warnings.append(f"Python {sys.version.split()[0]} detected. Strongly recommended: 3.11.9")
    
    def check_packages(self):
        """Check if all required packages are installed with minimum required versions."""
        for package, min_version in REQUIRED_PACKAGES.items():
            try:
                installed_version = pkg_resources.get_distribution(package).version
                if version.parse(installed_version) >= version.parse(min_version):
                    self.success_messages.append(f"{package} {installed_version} installed (>= {min_version})")
                else:
                    self.errors.append(
                        f"{package} version {installed_version} installed, but >= {min_version} required"
                    )
            except pkg_resources.DistributionNotFound:
                self.errors.append(f"{package} is not installed (requires >= {min_version})")
    
    def check_engine_file(self):
        """Check if src/engine.py exists and create template if not."""
        # Create src directory if it doesn't exist
        src_dir = Path("src")
        if not src_dir.exists():
            try:
                src_dir.mkdir(parents=True, exist_ok=True)
                self.success_messages.append("Created src/ directory")
            except Exception as e:
                self.errors.append(f"Failed to create src/ directory: {str(e)}")
                return False
        
        engine_path = src_dir / "engine.py"
        
        if not engine_path.exists():
            try:
                with open(engine_path, 'w') as f:
                    f.write(ENGINE_TEMPLATE)
                self.warnings.append("Created src/engine.py template. Please update with your PostgreSQL credentials")
                return False
            except Exception as e:
                self.errors.append(f"Failed to create src/engine.py: {str(e)}")
                return False
        else:
            # Check if it's still using default credentials
            try:
                with open(engine_path, 'r') as f:
                    content = f.read()
                    if 'username = "username"' in content and 'password = "password"' in content:
                        self.warnings.append("src/engine.py exists but still has default credentials. Please update!")
                        return False
                    else:
                        self.success_messages.append("src/engine.py found with custom credentials")
                        return True
            except Exception as e:
                self.errors.append(f"Error reading src/engine.py: {str(e)}")
                return False
    
    def check_postgresql(self):
        """Check PostgreSQL connectivity and database existence."""
        try:
            # First, try to import engine.py from src
            spec = importlib.util.spec_from_file_location("engine", "src/engine.py")
            if spec and spec.loader:
                engine_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(engine_module)
                
                # Extract credentials from engine.py
                username = engine_module.username
                password = engine_module.password
                
                if username == "username" or password == "password":
                    self.warnings.append("PostgreSQL credentials not configured in src/engine.py")
                    return
                
                # Try to create SQLAlchemy engine to test connection
                try:
                    from sqlalchemy import text
                    engine = engine_module.create_engine()
                    # Try a simple connection test with SQLAlchemy 2.0 syntax
                    with engine.connect() as conn:
                        result = conn.execute(text("SELECT 1"))
                        result.fetchone()
                    self.success_messages.append("PostgreSQL connected and 'letf_data' database accessible")
                    
                except Exception as e:
                    error_str = str(e)
                    if "password authentication failed" in error_str:
                        self.errors.append("PostgreSQL authentication failed. Check credentials in src/engine.py")
                    elif "could not connect to server" in error_str:
                        self.errors.append("PostgreSQL server not running. Start with: net start postgresql-x64-17")
                    elif "database \"letf_data\" does not exist" in error_str:
                        self.errors.append("Database 'letf_data' does not exist. Run: CREATE DATABASE letf_data;")
                    else:
                        self.errors.append(f"PostgreSQL connection error: {error_str}")
                    
        except FileNotFoundError:
            self.errors.append("src/engine.py not found")
        except Exception as e:
            self.errors.append(f"Could not import src/engine.py: {str(e)}")
    
    def check_virtual_environment(self):
        """Check if running in a virtual environment."""
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            self.success_messages.append("Running in virtual environment")
        else:
            self.warnings.append("Not running in virtual environment (recommended)")
    
    def check_main_py(self):
        """Check if main.py exists in either src directory."""
        src_main = Path("src/main.py")
        
        if src_main.exists():
            self.success_messages.append("main.py found in src directory")
            return "src"
        else:
            self.errors.append("main.py not found in src directory")
            return None
    
    def run_validation(self):
        """Run all validation checks."""
        print("=" * 60)
        print("BACKTESTING PROJECT SETUP VALIDATION")
        print("=" * 60)
        print()
        
        print("Checking Python version...")
        self.check_python_version()
        
        print("Checking virtual environment...")
        self.check_virtual_environment()
        
        print("Checking required packages...")
        self.check_packages()
        
        print("Checking src/engine.py...")
        engine_configured = self.check_engine_file()
        
        if engine_configured:
            print("Checking PostgreSQL connection...")
            self.check_postgresql()
        
        print("Checking main.py...")
        main_location = self.check_main_py()
        
        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)
        
        if self.success_messages:
            print("\nSUCCESSES:")
            for msg in self.success_messages:
                print(f"  {msg}")
        
        if self.warnings:
            print("\nWARNINGS:")
            for msg in self.warnings:
                print(f"  {msg}")
        
        if self.errors:
            print("\nERRORS:")
            for msg in self.errors:
                print(f"  {msg}")
        
        print("\n" + "=" * 60)
        
        if self.errors:
            print("Setup validation FAILED. Please fix the errors above before running main.py")
            return False, main_location
        elif self.warnings:
            print("Setup validation completed with warnings. Consider addressing them.")
            return True, main_location
        else:
            print("Setup validation PASSED! You can now run main.py")
            return True, main_location


def main():
    validator = SetupValidator()
    success, main_location = validator.run_validation()
    
    # Handle installation of missing packages
    if not success and validator.errors:
        missing_packages = [err for err in validator.errors if "is not installed" in err]
        if missing_packages:
            response = input("\nWould you like to install missing packages? (y/n): ").lower()
            if response == 'y':
                for package, min_version in REQUIRED_PACKAGES.items():
                    try:
                        # Check if package needs to be installed
                        if any(package in err for err in validator.errors):
                            print(f"Installing {package}>={min_version}...")
                            subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}>={min_version}"])
                    except subprocess.CalledProcessError:
                        print(f"Failed to install {package}")
                
                print("\nPackages installed. Please run config.py again to revalidate.")
                sys.exit(0)
    
    # If validation passes, offer to run main.py
    if success and main_location:
        print()
        response = input("Would you like to run main.py now? (y/n): ").lower()
        if response == 'y':
            print("\n" + "=" * 60)
            print("Running main.py...")
            print("=" * 60 + "\n")
            
            try:
                subprocess.run([sys.executable, "src/main.py"], check=True)
            except subprocess.CalledProcessError as e:
                print(f"\nError running main.py: {e}")
                sys.exit(1)
            except KeyboardInterrupt:
                print("\n\n main.py interrupted by user")
                sys.exit(0)
    
    # Exit with appropriate code
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()