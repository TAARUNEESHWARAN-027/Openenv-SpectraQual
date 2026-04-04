"""
Hosted Model Client - Integration with external RL model APIs

Supports:
- OpenAI-compatible endpoints
- Hugging Face Inference API
- Custom REST APIs
"""

import os
import json
from typing import Optional, Dict, Any
import requests
from .models import PCBObservation


class HostedModelClient:
    """Client for hosted RL models (read-only inference)."""
    
    def __init__(self, model_name: str, api_base_url: Optional[str] = None, api_token: Optional[str] = None):
        """
        Initialize hosted model client.
        
        Args:
            model_name: Model identifier (e.g. "meta-llama/Llama-2-7b" or "openrouter/auto")
            api_base_url: Custom API endpoint (optional, uses HF Inference API if not provided)
            api_token: API token for authentication
        """
        self.model_name = model_name
        self.api_base_url = api_base_url or os.getenv("API_BASE_URL", "https://api-inference.huggingface.co")
        self.api_token = api_token or os.getenv("HF_TOKEN", os.getenv("HUGGINGFACE_API_KEY"))
        
        if not self.api_token:
            raise ValueError("API_TOKEN or HF_TOKEN must be set")
        
        self.headers = {"Authorization": f"Bearer {self.api_token}"}
    
    def select_action(self, obs: PCBObservation) -> str:
        """
        Query hosted model for action given observation.
        
        Args:
            obs: PCBObservation
            
        Returns:
            Action string (e.g. "ROUTE_SOLDERING")
            
        Raises:
            RuntimeError: If API call fails
        """
        try:
            # Prepare prompt for LLM
            prompt = self._build_prompt(obs)
            
            # Call hosted API
            response = requests.post(
                f"{self.api_base_url}/models/{self.model_name}",
                headers=self.headers,
                json={"inputs": prompt, "parameters": {"max_tokens": 20}},
                timeout=10
            )
            response.raise_for_status()
            
            # Extract action from response
            result = response.json()
            action = self._extract_action_from_response(result)
            
            return action
        except Exception as exc:
            raise RuntimeError(f"Hosted model API failed: {exc}")
    
    def _build_prompt(self, obs: PCBObservation) -> str:
        """Build LLM prompt from observation."""
        return f"""
Given a PCB inspection result:
- Defect Type: {obs.defect_type}
- Criticality: {obs.criticality_score}
- Cost: ${obs.component_cost}
- Confidence: {obs.inspection_confidence}
- Queue Length: {obs.queue_length}
- Available Slots: {obs.available_slots}

You must choose ONE action:
1. PASS
2. SCRAP
3. ROUTE_COMPONENT_REPLACEMENT
4. ROUTE_SOLDERING
5. ROUTE_DIAGNOSTICS
6. WAIT

Choose the best action. Respond with ONLY the action name.
Answer:
"""
    
    def _extract_action_from_response(self, response: Any) -> str:
        """Extract action string from API response."""
        actions = [
            "PASS",
            "SCRAP",
            "ROUTE_COMPONENT_REPLACEMENT",
            "ROUTE_SOLDERING",
            "ROUTE_DIAGNOSTICS",
            "WAIT",
        ]
        
        # Handle different response formats
        if isinstance(response, list) and len(response) > 0:
            text = response[0].get("generated_text", "").upper()
        elif isinstance(response, dict):
            text = response.get("generated_text", "").upper()
        else:
            text = str(response).upper()
        
        # Find matching action in response
        for action in actions:
            if action in text:
                return action
        
        # Fallback to diagnostics if no match
        return "ROUTE_DIAGNOSTICS"
