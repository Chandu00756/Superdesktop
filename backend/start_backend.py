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
    print("API Server will be available at: https://127.0.0.1:8443 (if TLS certs found), else http")
    print("Press Ctrl+C to stop the server")
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Import and run the server
    from api_server import app
    import uvicorn
    
    # TLS is disabled by default for local development to avoid self-signed cert issues in the browser.
    # To enable TLS, set OMEGA_ENABLE_TLS=1 and provide cert/key files (via env or default paths).
    enable_tls = os.environ.get('OMEGA_ENABLE_TLS', '0') in ('1', 'true', 'yes', 'on')
    cert_env = os.environ.get('OMEGA_SSL_CERT')
    key_env = os.environ.get('OMEGA_SSL_KEY')
    cert_default = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'security', 'certs', 'control_node.crt'))
    key_default = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'security', 'certs', 'control_node.key'))
    cert_file = None
    key_file = None
    if enable_tls:
        cert_file = cert_env if cert_env and os.path.exists(cert_env) else (cert_default if os.path.exists(cert_default) else None)
        key_file = key_env if key_env and os.path.exists(key_env) else (key_default if os.path.exists(key_default) else None)

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8443,
        reload=False,
        access_log=True,
        log_level="info",
        ssl_certfile=cert_file,
        ssl_keyfile=key_file
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
