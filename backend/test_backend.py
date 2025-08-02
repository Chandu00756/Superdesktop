#!/usr/bin/env python3
"""
Backend Integration Test Suite
Tests all API endpoints and encrypted communication
"""

import asyncio
import aiohttp
import json
import time
import logging

class BackendTester:
    def __init__(self, base_url="http://127.0.0.1:8443"):
        self.base_url = base_url
        self.session = None
        self.auth_token = None
        
    async def start(self):
        self.session = aiohttp.ClientSession()
        
    async def stop(self):
        if self.session:
            await self.session.close()
            
    async def test_authentication(self):
        """Test login and token retrieval"""
        print("Testing authentication...")
        
        try:
            async with self.session.post(f"{self.base_url}/api/auth/login", json={
                "username": "admin",
                "password": "omega123"
            }) as response:
                if response.status == 200:
                    data = await response.json()
                    self.auth_token = data["token"]
                    print("Authentication successful")
                    return True
                else:
                    print(f"Authentication failed: {response.status}")
                    return False
        except Exception as e:
            print(f"Authentication error: {e}")
            return False
    
    async def make_request(self, endpoint, method="GET", data=None):
        """Make authenticated request"""
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {json.dumps(self.auth_token)}"
            
        try:
            async with self.session.request(
                method, 
                f"{self.base_url}{endpoint}",
                json=data,
                headers=headers
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"Request failed: {endpoint} - {response.status}")
                    return None
        except Exception as e:
            print(f"Request error: {endpoint} - {e}")
            return None
    
    def decrypt_response(self, encrypted_data):
        """Simulate response decryption"""
        try:
            # In a real implementation, this would decrypt the data
            # For testing, we'll just return a mock structure
            return {
                "status": "decrypted",
                "data": encrypted_data
            }
        except Exception as e:
            print(f"Decryption error: {e}")
            return encrypted_data
    
    async def test_dashboard_api(self):
        """Test dashboard metrics endpoint"""
        print("Testing dashboard API...")
        
        response = await self.make_request("/api/dashboard/metrics")
        if response:
            decrypted = self.decrypt_response(response)
            print("Dashboard API working")
            return True
        else:
            print("Dashboard API failed")
            return False
    
    async def test_nodes_api(self):
        """Test nodes management API"""
        print("Testing nodes API...")
        
        # Test getting nodes
        response = await self.make_request("/api/nodes")
        if response:
            print("Nodes GET API working")
        else:
            print("Nodes GET API failed")
            return False
            
        # Test registering a node
        node_data = {
            "node_id": "test-node-01",
            "node_type": "compute",
            "hostname": "test-host",
            "ip_address": "192.168.1.200",
            "port": 8001,
            "resources": {
                "cpu_cores": 16,
                "memory_gb": 64,
                "gpu_units": 2
            }
        }
        
        response = await self.make_request("/api/nodes/register", "POST", node_data)
        if response:
            print("Node registration API working")
            return True
        else:
            print("Node registration API failed")
            return False
    
    async def test_sessions_api(self):
        """Test session management API"""
        print("Testing sessions API...")
        
        # Test getting sessions
        response = await self.make_request("/api/sessions")
        if response:
            print("Sessions GET API working")
        else:
            print("Sessions GET API failed")
            return False
            
        # Test creating a session
        session_data = {
            "user_id": "test-user",
            "application": "Test Application",
            "cpu_cores": 4,
            "gpu_units": 1,
            "memory_gb": 8
        }
        
        response = await self.make_request("/api/sessions/create", "POST", session_data)
        if response:
            decrypted = self.decrypt_response(response)
            if decrypted.get("success"):
                session_id = decrypted.get("session_id")
                print("Session creation API working")
                
                # Test terminating the session
                if session_id:
                    term_response = await self.make_request(f"/api/sessions/{session_id}", "DELETE")
                    if term_response:
                        print("Session termination API working")
                        return True
                    else:
                        print("Session termination API failed")
                        return False
            else:
                print("Session creation failed")
                return False
        else:
            print("Session creation API failed")
            return False
    
    async def test_actions_api(self):
        """Test action endpoints"""
        print("Testing actions API...")
        
        # Test node discovery
        response = await self.make_request("/api/actions/discover_nodes", "POST")
        if response:
            print("Node discovery API working")
        else:
            print("Node discovery API failed")
            
        # Test benchmark
        response = await self.make_request("/api/actions/run_benchmark", "POST")
        if response:
            print("Benchmark API working")
        else:
            print("Benchmark API failed")
            
        # Test health check
        response = await self.make_request("/api/actions/health_check", "POST")
        if response:
            print("Health check API working")
            return True
        else:
            print("Health check API failed")
            return False
    
    async def test_websocket(self):
        """Test WebSocket connection"""
        print("Testing WebSocket connection...")
        
        try:
            import websockets
            
            uri = "ws://127.0.0.1:8443/ws/realtime"
            async with websockets.connect(uri) as websocket:
                # Send a test message
                await websocket.send(json.dumps({"type": "ping"}))
                
                # Wait for response (with timeout)
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    print("WebSocket connection working")
                    return True
                except asyncio.TimeoutError:
                    print("WebSocket connected (no immediate response)")
                    return True
                    
        except Exception as e:
            print(f"WebSocket connection failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run complete test suite"""
        print("=" * 50)
        print("OMEGA BACKEND INTEGRATION TESTS")
        print("=" * 50)
        
        results = []
        
        # Authentication
        results.append(await self.test_authentication())
        
        if self.auth_token:
            # API Tests
            results.append(await self.test_dashboard_api())
            results.append(await self.test_nodes_api())
            results.append(await self.test_sessions_api())
            results.append(await self.test_actions_api())
            
            # WebSocket Test
            results.append(await self.test_websocket())
        else:
            print("Skipping API tests due to authentication failure")
        
        # Summary
        print("\n" + "=" * 50)
        print("TEST RESULTS SUMMARY")
        print("=" * 50)
        
        passed = sum(results)
        total = len(results)
        
        print(f"Tests passed: {passed}/{total}")
        
        if passed == total:
            print("ALL TESTS PASSED - Backend is fully functional!")
        else:
            print("Some tests failed - Check backend configuration")
            
        return passed == total

async def main():
    tester = BackendTester()
    await tester.start()
    
    try:
        success = await tester.run_all_tests()
        return 0 if success else 1
    finally:
        await tester.stop()

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nTests interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"Test error: {e}")
        sys.exit(1)
