#!/bin/bash

# Install Homebrew if not already installed

if ! command -v brew &> /dev/null; then
    echo "Homebrew is not installed. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Add to PATH
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
    eval "$(/opt/homebrew/bin/brew shellenv)"
fi

# Verify that Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Homebrew installation failed. Please install Homebrew manually and try again."
    exit 1
fi

# Install Git if not already installed
if ! command -v git &> /dev/null; then
    echo "Git is not installed. Installing Git via Homebrew..."
    brew install git
    echo "Git installed successfully."
fi

# Verify that Git is installed
if ! command -v git &> /dev/null; then
    echo "Git installation failed. Please try again."
    exit 1
fi

# Install Python 3.11 using Homebrew
# Check if Python is installed
if ! command -v python3.11 &> /dev/null; then
    # Install Python using Homebrew
    echo "Python not installed. Installing Python via Homebrew..."
    brew install python@3.11
    echo "Python installed successfully."
fi

# Verify that Python is installed
if ! command -v python3.11 &> /dev/null; then
    echo "Python installation failed. Please try again."
    exit 1
fi

# Fetch the app source files from Github if not already present and that .git exists, otherwise update the existing files
if [ -d "PythonCBAS" ] && [ -d "PythonCBAS/.git" ]; then
    echo "Updating source code..."
    cd PythonCBAS
    git pull
    cd ..
else
    echo "Fetching the source code from Github..."
    # Clear the directory if it exists
    if [ -d "PythonCBAS" ]; then
        # Prompt the user for confirmation
        read -p "The directory 'PythonCBAS' already exists. Do you want to delete it and fetch the source code again? (y/n): " answer
        if [ "$answer" != "${answer#[Yy]}" ]; then
            # Delete the directory
            rm -rf PythonCBAS
        else
            echo "Setup aborted. No changes were made to the directory."
            exit 1
        fi
    fi
    cd ~
    git clone https://github.com/danielhsu021202/PythonCBAS.git PythonCBAS
    cd PythonCBAS
    git config pull.rebase false  # merge (the default strategy)
    cd ..
    echo "Source code fetched successfully."
fi

# Set up the environment
cd PythonCBAS
python3.11 -m venv pythoncbas_env

# Activate the virtual environment
source pythoncbas_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Deactivate the virtual environment
deactivate

# Add pythoncbas as a command to run this script
echo "Adding pythoncbas command..."

# The script is in the bin/ directory. Copy it to /usr/local/bin
chmod +x bin/pythoncbas


# Make sure usr/local/bin exists
if [ ! -d "/usr/local/bin" ]; then
    sudo mkdir /usr/local/bin
fi
# sudo mv pythoncbas /usr/local/bin
sudo cp bin/pythoncbas /usr/local/bin


echo "Starting PythonCBAS..."

# Start the PythonCBAS application


# Activate the virtual environment
source pythoncbas_env/bin/activate

# Start the application
# Try python3, if not found, try python
if command -v python3 &>/dev/null; then
    python3 src/PythonCBAS.py
else
    python src/PythonCBAS.py
fi

