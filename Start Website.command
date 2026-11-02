#!/bin/bash
# Double-click this file to install (first time only) and start the website.
cd "$(dirname "$0")"
echo "Setting up the Speech Digit Recognition website..."
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv || { echo "Could not create the Python environment."; read -p "Press Enter to close"; exit 1; }
fi
if [ ! -f .venv/.installed ]; then
  echo "Installing libraries (first time only, this can take a few minutes)..."
  .venv/bin/pip install --upgrade pip -q
  .venv/bin/pip install -r requirements.txt && touch .venv/.installed || { echo "Installation failed. Send a screenshot of this window."; read -p "Press Enter to close"; exit 1; }
fi
(sleep 8; open http://127.0.0.1:5000) &
echo ""
echo "Website starting at http://127.0.0.1:5000 . Your browser will open automatically."
echo "Keep this window open while you use the website. Close it to stop the website."
echo ""
.venv/bin/python app.py
read -p "The website stopped. Press Enter to close"
