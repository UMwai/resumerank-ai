#!/bin/bash

# Launch script for UMTrading Live Monitor Dashboard

echo "=============================================="
echo "   UMTrading Live Portfolio Monitor"
echo "=============================================="
echo ""
echo "Starting dashboard server..."
echo ""

# Check if streamlit is installed
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "ERROR: Streamlit not installed!"
    echo "Please run: pip3 install -r requirements.txt"
    exit 1
fi

# Launch the dashboard
echo "Dashboard will open in your browser at: http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""

# Use python3 -m streamlit to ensure proper module loading
cd "$(dirname "$0")"
python3 -m streamlit run dashboards/live_monitor.py