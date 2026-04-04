"""
RL Model Loader - Load and run locally uploaded RL models

Supports:
- PyTorch models (.pt)
- TensorFlow models (.h5, .pb)
- Saved agent blobs
"""

import os
import pickle
import json
from typing import Optional, Dict, Any, Callable
import numpy as np
from .models import PCBObservation, PCBAction


class RLModelLoader:
    """Load and run locally uploaded RL models."""
    
    def __init__(self, model_path: str):
        """
        Load an RL model from disk.
        
        Args:
            model_path: Path to saved model (.pt, .pkl, .h5, etc.)
            
        Raises:
            FileNotFoundError: If model file not found
            ValueError: If model format unsupported
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model_path = model_path
        self.model = None
        self.model_type = self._detect_model_type(model_path)
        self._load_model()
    
    def _detect_model_type(self, path: str) -> str:
        """Detect model format from file extension."""
        ext = os.path.splitext(path)[1].lower()
        
        if ext == ".pt":
            return "pytorch"
        elif ext == ".pkl" or ext == ".pickle":
            return "pickle"
        elif ext in [".h5", ".hdf5"]:
            return "tensorflow"
        elif ext == ".pb":
            return "tensorflow_pb"
        else:
            return "unknown"
    
    def _load_model(self):
        """Load model from disk."""
        if self.model_type == "pytorch":
            try:
                import torch
                self.model = torch.load(self.model_path)
            except Exception as exc:
                raise ValueError(f"Failed to load PyTorch model: {exc}")
        
        elif self.model_type == "pickle":
            try:
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
            except Exception as exc:
                raise ValueError(f"Failed to load pickle model: {exc}")
        
        elif self.model_type == "tensorflow":
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(self.model_path)
            except Exception as exc:
                raise ValueError(f"Failed to load TensorFlow model: {exc}")
        
        else:
            raise ValueError(f"Unsupported model format: {self.model_type}")
    
    def select_action(self, obs: PCBObservation, obs_to_vector: Optional[Callable] = None) -> str:
        """
        Get action from loaded model.
        
        Args:
            obs: PCBObservation
            obs_to_vector: Optional function to convert obs to vector
            
        Returns:
            Action string
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Convert observation to vector if callable provided
        if obs_to_vector is not None:
            state_vec = obs_to_vector(obs)
        else:
            # Default vectorization
            from .utils import observation_to_vector
            state_vec = observation_to_vector(obs)
        
        # Get action from model
        try:
            if self.model_type == "pytorch":
                import torch
                with torch.no_grad():
                    if isinstance(state_vec, np.ndarray):
                        state_tensor = torch.from_numpy(state_vec).float().unsqueeze(0)
                    else:
                        state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
                    
                    output = self.model(state_tensor)
                    action_idx = output.argmax(dim=1).item()
            
            elif self.model_type == "pickle":
                # Assume pickle contains a callable agent
                if callable(self.model):
                    action_idx = self.model(state_vec)
                else:
                    raise ValueError("Pickle model is not callable")
            
            elif self.model_type in ["tensorflow", "tensorflow_pb"]:
                output = self.model.predict(np.array([state_vec]), verbose=0)
                action_idx = np.argmax(output[0])
            
            else:
                raise ValueError(f"Cannot infer action from {self.model_type}")
            
            # Convert index to action string
            from .utils import index_to_action
            action = index_to_action(int(action_idx))
            return action
        
        except Exception as exc:
            raise RuntimeError(f"Model inference failed: {exc}")
    
    def save_checkpoint(self, save_path: str):
        """Save current model state to disk."""
        try:
            if self.model_type == "pytorch":
                import torch
                torch.save(self.model, save_path)
            elif self.model_type == "pickle":
                with open(save_path, "wb") as f:
                    pickle.dump(self.model, f)
            elif self.model_type in ["tensorflow", "tensorflow_pb"]:
                self.model.save(save_path)
            else:
                raise ValueError(f"Cannot save {self.model_type}")
        except Exception as exc:
            raise RuntimeError(f"Failed to save model: {exc}")
