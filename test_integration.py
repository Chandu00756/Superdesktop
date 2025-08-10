#!/usr/bin/env python3
"""
SuperDesktop v2.0 Integration Test
Tests all components are working correctly
"""

import requests
import json
import time
from urllib.parse import urlparse

def test_backend_api():
    """Test backend API endpoints"""
    base_url = "http://127.0.0.1:8443"
    endpoints = [
        "/api/dashboard/metrics",
        "/api/nodes", 
        "/api/sessions",
        "/api/resources",
        "/api/network",
        "/api/performance",
        "/api/security"
    ]
    
    print("🔧 Testing Backend API Endpoints...")
    results = {}
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and 'payload' in data:
                    results[endpoint] = "✅ OK (Encrypted)"
                else:
                    results[endpoint] = "✅ OK (Plain JSON)"
            else:
                results[endpoint] = f"❌ HTTP {response.status_code}"
        except Exception as e:
            results[endpoint] = f"❌ Error: {str(e)[:50]}"
    
    return results

def test_frontend_server():
    """Test frontend HTTP server"""
    print("🌐 Testing Frontend Server...")
    try:
        response = requests.get("http://127.0.0.1:8080/omega-control-center.html", timeout=5)
        if response.status_code == 200:
            content = response.text
            if "SuperDesktop" in content and "omega-style.css" in content:
                return "✅ Frontend serving correctly"
            else:
                return "❌ Frontend content invalid"
        else:
            return f"❌ HTTP {response.status_code}"
    except Exception as e:
        return f"❌ Error: {str(e)[:50]}"

def test_cross_origin():
    """Test CORS configuration"""
    print("🔐 Testing CORS Configuration...")
    try:
        # Simulate a cross-origin request
        headers = {
            'Origin': 'http://127.0.0.1:8080',
            'Access-Control-Request-Method': 'GET'
        }
        response = requests.options("http://127.0.0.1:8443/api/dashboard/metrics", 
                                   headers=headers, timeout=5)
        
        cors_headers = response.headers
        if 'Access-Control-Allow-Origin' in cors_headers:
            return "✅ CORS enabled"
        else:
            return "❌ CORS not configured"
    except Exception as e:
        return f"❌ Error: {str(e)[:50]}"

def main():
    print("🚀 SuperDesktop v2.0 Integration Test")
    print("=" * 50)
    
    # Test backend
    backend_results = test_backend_api()
    for endpoint, status in backend_results.items():
        print(f"{endpoint}: {status}")
    
    print()
    
    # Test frontend
    frontend_status = test_frontend_server()
    print(f"Frontend Server: {frontend_status}")
    
    # Test CORS
    cors_status = test_cross_origin()
    print(f"CORS Configuration: {cors_status}")
    
    print()
    print("🎯 Integration Summary:")
    
    # Count successes
    backend_ok = sum(1 for status in backend_results.values() if "✅" in status)
    total_backend = len(backend_results)
    
    print(f"Backend APIs: {backend_ok}/{total_backend} working")
    print(f"Frontend: {'✅' if '✅' in frontend_status else '❌'}")
    print(f"CORS: {'✅' if '✅' in cors_status else '❌'}")
    
    if backend_ok == total_backend and "✅" in frontend_status and "✅" in cors_status:
        print("\n🎉 All systems operational! SuperDesktop v2.0 ready!")
        print("🌐 Access: http://127.0.0.1:8080/omega-control-center.html")
    else:
        print("\n⚠️  Some issues detected. Check individual components above.")

if __name__ == "__main__":
    main()
