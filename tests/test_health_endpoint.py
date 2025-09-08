import asyncio
import os
import json
import pytest
from fastapi.testclient import TestClient

# Import the FastAPI app
from backend.api_server import app

def test_health_schema():
    client = TestClient(app)
    r = client.get('/health')
    assert r.status_code == 200
    data = r.json()
    assert 'status' in data and data['status'] in ('healthy','degraded','ok')
    if data.get('status') in ('healthy','degraded'):
        assert 'version' in data
        assert 'dependencies' in data and isinstance(data['dependencies'], dict)
        assert 'uptime_seconds' in data
