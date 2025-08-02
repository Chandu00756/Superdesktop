#!/usr/bin/env python3
"""
Omega Control Center Backend Startup Script
"""

import sys
import os
import subprocess
import logging

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import aiohttp
        import cryptography
        import psutil
        import numpy
        print("All dependencies are installed")
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return True

def start_backend():
    """Start the backend API server"""
    print("Starting Omega Control Center Backend...")
    print("API Server will be available at: http://127.0.0.1:8443")
    print("Press Ctrl+C to stop the server")
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Import and run the server
    from api_server import app
    import uvicorn
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8443,
        reload=False,
        access_log=True,
        log_level="info"
    )

if __name__ == "__main__":
    try:
        if check_dependencies():
            start_backend()
    except KeyboardInterrupt:
        print("\nBackend server stopped")
    except Exception as e:
        print(f"Error starting backend: {e}")
        sys.exit(1)
