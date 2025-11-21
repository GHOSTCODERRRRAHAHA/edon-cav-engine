"""Neural head MLP for state classification and action recommendations."""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

# Try to import PyTorch, fallback to numpy-based implementation
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class NeuralHeadMLP:
    """MLP neural head for state prediction and action recommendations."""
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dims: List[int] = [64, 32],
        num_states: int = 6,
        dropout: float = 0.1,
        use_torch: bool = True
    ):
        """
        Initialize neural head.
        
        Args:
            input_dim: Input dimension (CAV embedding size, default 128)
            hidden_dims: Hidden layer dimensions
            num_states: Number of state classes (6: restorative, focus, balanced, overload, alert, emergency)
            dropout: Dropout rate
            use_torch: Whether to use PyTorch (if available)
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_states = num_states
        self.dropout = dropout
        self.use_torch = use_torch and HAS_TORCH
        
        if self.use_torch:
            self._build_torch_model()
        else:
            self._build_numpy_model()
    
    def _build_torch_model(self):
        """Build PyTorch model."""
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim
        
        # State classification head
        self.state_head = nn.Sequential(*layers, nn.Linear(prev_dim, self.num_states))
        
        # Action recommendation head (7 outputs: speed_scale, torque_scale, safety_scale, etc.)
        action_layers = layers.copy()  # Share hidden layers
        action_layers.append(nn.Linear(prev_dim, 7))
        self.action_head = nn.Sequential(*action_layers)
        
        # Initialize weights
        for module in self.state_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        for module in self.action_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        self.model = nn.ModuleDict({
            'state_head': self.state_head,
            'action_head': self.action_head
        })
        self.model.eval()  # Set to eval mode
    
    def _build_numpy_model(self):
        """Build numpy-based model (simple feedforward)."""
        # Store weights as numpy arrays
        self.weights = {}
        self.biases = {}
        
        prev_dim = self.input_dim
        layer_idx = 0
        
        for hidden_dim in self.hidden_dims:
            # Initialize weights with Xavier-like initialization
            self.weights[f'hidden_{layer_idx}'] = np.random.randn(prev_dim, hidden_dim) * np.sqrt(2.0 / prev_dim)
            self.biases[f'hidden_{layer_idx}'] = np.zeros(hidden_dim)
            prev_dim = hidden_dim
            layer_idx += 1
        
        # State head
        self.weights['state'] = np.random.randn(prev_dim, self.num_states) * np.sqrt(2.0 / prev_dim)
        self.biases['state'] = np.zeros(self.num_states)
        
        # Action head
        self.weights['action'] = np.random.randn(prev_dim, 7) * np.sqrt(2.0 / prev_dim)
        self.biases['action'] = np.zeros(7)
    
    def _forward_numpy(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass using numpy."""
        # Hidden layers
        h = x
        for i in range(len(self.hidden_dims)):
            h = h @ self.weights[f'hidden_{i}'] + self.biases[f'hidden_{i}']
            h = np.maximum(0, h)  # ReLU
        
        # State head
        state_logits = h @ self.weights['state'] + self.biases['state']
        state_probs = self._softmax(state_logits)
        
        # Action head
        action_outputs = h @ self.weights['action'] + self.biases['action']
        # Apply activation functions
        action_outputs[0] = np.clip(action_outputs[0], 0.0, 2.0)  # speed_scale
        action_outputs[1] = np.clip(action_outputs[1], 0.0, 2.0)  # torque_scale
        action_outputs[2] = np.clip(action_outputs[2], 0.0, 1.0)  # safety_scale
        action_outputs[3] = 1.0 / (1.0 + np.exp(-action_outputs[3]))  # caution_flag (sigmoid)
        action_outputs[4] = 1.0 / (1.0 + np.exp(-action_outputs[4]))  # emergency_flag (sigmoid)
        action_outputs[5] = np.clip(action_outputs[5], 0.0, 1.0)  # focus_boost
        action_outputs[6] = 1.0 / (1.0 + np.exp(-action_outputs[6]))  # recovery_recommended (sigmoid)
        
        return state_probs, action_outputs
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax function."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def predict(self, cav_embedding) -> Dict[str, Any]:
        """
        Predict state and action recommendations from CAV embedding.
        
        Args:
            cav_embedding: 128-dimensional CAV embedding (list or numpy array)
            
        Returns:
            Dictionary with:
            - state_probs: Probability distribution over states
            - state_class: Predicted state class
            - action_recommendations: Action recommendations
        """
        # Convert to numpy array if needed
        if isinstance(cav_embedding, list):
            cav_embedding = np.array(cav_embedding, dtype=np.float32)
        
        if len(cav_embedding) != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {len(cav_embedding)}")
        
        if self.use_torch:
            with torch.no_grad():
                x = torch.FloatTensor(cav_embedding).unsqueeze(0)
                state_logits = self.state_head(x)
                action_outputs = self.action_head(x)
                
                state_probs = torch.softmax(state_logits, dim=1).numpy()[0]
                action_outputs = action_outputs.numpy()[0]
                
                # Apply constraints to action outputs
                action_outputs[0] = np.clip(action_outputs[0], 0.0, 2.0)  # speed_scale
                action_outputs[1] = np.clip(action_outputs[1], 0.0, 2.0)  # torque_scale
                action_outputs[2] = np.clip(action_outputs[2], 0.0, 1.0)  # safety_scale
                action_outputs[3] = 1.0 / (1.0 + np.exp(-action_outputs[3]))  # caution_flag
                action_outputs[4] = 1.0 / (1.0 + np.exp(-action_outputs[4]))  # emergency_flag
                action_outputs[5] = np.clip(action_outputs[5], 0.0, 1.0)  # focus_boost
                action_outputs[6] = 1.0 / (1.0 + np.exp(-action_outputs[6]))  # recovery_recommended
        else:
            state_probs, action_outputs = self._forward_numpy(cav_embedding)
        
        # Map state indices to class names
        state_names = ['restorative', 'focus', 'balanced', 'overload', 'alert', 'emergency']
        state_idx = np.argmax(state_probs)
        state_class = state_names[state_idx]
        
        # Build action recommendations
        action_recommendations = {
            'speed_scale': float(action_outputs[0]),
            'torque_scale': float(action_outputs[1]),
            'safety_scale': float(action_outputs[2]),
            'caution_flag': bool(action_outputs[3] > 0.5),
            'emergency_flag': bool(action_outputs[4] > 0.5),
            'focus_boost': float(action_outputs[5]),
            'recovery_recommended': bool(action_outputs[6] > 0.5)
        }
        
        return {
            'state_probs': {name: float(prob) for name, prob in zip(state_names, state_probs)},
            'state_class': state_class,
            'action_recommendations': action_recommendations,
            'confidence': float(state_probs[state_idx])
        }


def create_default_neural_head(input_dim: int = 128) -> NeuralHeadMLP:
    """Create a default neural head."""
    return NeuralHeadMLP(
        input_dim=input_dim,
        hidden_dims=[64, 32],
        num_states=6,
        dropout=0.1,
        use_torch=True
    )

