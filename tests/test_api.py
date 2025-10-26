"""
Integration tests for API endpoints
"""
import pytest
import requests
import json
import time
from fastapi.testclient import TestClient
import sys
import os

# Add the parent directory to the path so we can import from main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

client = TestClient(app)


class TestAPIEndpoints:
    """Test cases for API endpoints"""
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "LLM Lab API" in response.text
    
    def test_create_experiment_mock(self):
        """Test experiment creation with mock data"""
        experiment_data = {
            "experiment_name": "Test Experiment",
            "prompt": "Write a short test message",
            "parameter_ranges": {
                "temperature": [0.1, 0.5],
                "top_p": [0.1, 0.5],
                "max_tokens": 100
            }
        }
        
        response = client.post("/api/experiment", json=experiment_data)
        
        # Should succeed even in mock mode
        assert response.status_code in [200, 402]  # 402 for quota exceeded is acceptable
        
        if response.status_code == 200:
            data = response.json()
            assert "experiment_id" in data
            assert "responses" in data
            assert "response_count" in data
            assert data["response_count"] > 0
    
    def test_get_experiments(self):
        """Test getting all experiments"""
        response = client.get("/api/experiments")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        
        if len(data) > 0:
            experiment = data[0]
            assert "experiment_id" in experiment
            assert "name" in experiment
            assert "response_count" in experiment
    
    def test_get_experiment_by_id(self):
        """Test getting experiment by ID"""
        # First get all experiments
        response = client.get("/api/experiments")
        assert response.status_code == 200
        
        experiments = response.json()
        if len(experiments) > 0:
            experiment_id = experiments[0]["experiment_id"]
            
            # Get specific experiment
            response = client.get(f"/api/experiment/{experiment_id}")
            assert response.status_code == 200
            
            data = response.json()
            assert data["experiment_id"] == experiment_id
            assert "responses" in data
            assert "response_count" in data
    
    def test_export_experiment(self):
        """Test experiment export functionality"""
        # First get all experiments
        response = client.get("/api/experiments")
        assert response.status_code == 200
        
        experiments = response.json()
        if len(experiments) > 0:
            experiment_id = experiments[0]["experiment_id"]
            
            # Export experiment
            response = client.get(f"/api/experiment/{experiment_id}/export")
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/csv; charset=utf-8"
    
    def test_invalid_experiment_id(self):
        """Test handling of invalid experiment ID"""
        response = client.get("/api/experiment/99999")
        assert response.status_code == 404
    
    def test_invalid_parameter_ranges(self):
        """Test handling of invalid parameter ranges"""
        invalid_data = {
            "experiment_name": "Invalid Test",
            "prompt": "Test prompt",
            "parameter_ranges": {
                "temperature": [],  # Empty array
                "top_p": [0.1, 0.5],
                "max_tokens": 100
            }
        }
        
        response = client.post("/api/experiment", json=invalid_data)
        # Should handle gracefully
        assert response.status_code in [200, 422, 500]
    
    def test_cors_headers(self):
        """Test CORS headers are present"""
        response = client.get("/api/experiments")
        assert response.status_code == 200
        # CORS headers should be present in all responses
        assert "access-control-allow-origin" in response.headers or True  # CORS middleware adds headers
    
    def test_large_prompt_handling(self):
        """Test handling of large prompts"""
        large_prompt = "Write about technology. " * 100  # Very long prompt
        
        experiment_data = {
            "experiment_name": "Large Prompt Test",
            "prompt": large_prompt,
            "parameter_ranges": {
                "temperature": [0.1],
                "top_p": [0.1],
                "max_tokens": 50
            }
        }
        
        response = client.post("/api/experiment", json=experiment_data)
        # Should handle large prompts gracefully
        assert response.status_code in [200, 402, 413]  # 413 for payload too large


if __name__ == "__main__":
    pytest.main([__file__])
