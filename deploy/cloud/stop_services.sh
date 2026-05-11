#!/bin/bash

echo "Stopping LAVAD services..."

pkill -f uvicorn && echo "Backend (uvicorn) stopped" || echo "Backend not running"
pkill -f streamlit && echo "Frontend (streamlit) stopped" || echo "Frontend not running"

echo "All services stopped."