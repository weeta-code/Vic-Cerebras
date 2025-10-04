#!/bin/bash

# Grepal Server Setup Script
echo "Setting up Grepal Python server..."

# Create virtual environment
python3 -m venv grepal_env

# Activate virtual environment
source grepal_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo "Grepal server setup complete!"
echo "To activate the environment: source server/grepal_env/bin/activate"
echo "To start the server: cd server && python main.py"