"""
Target AI Systems to be validated - intentionally buggy for testing
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Any
from enum import Enum


class BugSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class BugType(Enum):
    ADVERSARIAL = "adversarial"
    EDGE_CASE = "edge_case"
    DISTRIBUTION_SHIFT = "distribution_shift"
    BOUNDARY = "boundary"
    PERFORMANCE = "performance"
    LOGIC_ERROR = "logic_error"


class TargetClassifier(nn.Module):
    """Simple neural network classifier with intentional vulnerabilities"""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, output_dim: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.vulnerability_regions = self._define_vulnerabilities()

    def _define_vulnerabilities(self) -> Dict[str, Any]:
        """Define intentional bugs for testing"""
        return {
            'adversarial': {
                'trigger': lambda x: np.abs(x[0] - 0.5) < 0.1 and np.abs(x[1] - 0.5) < 0.1,
                'severity': BugSeverity.CRITICAL,
                'type': BugType.ADVERSARIAL
            },
            'edge_case': {
                'trigger': lambda x: np.any(np.abs(x) > 5.0),
                'severity': BugSeverity.HIGH,
                'type': BugType.EDGE_CASE
            },
            'boundary': {
                'trigger': lambda x: np.sum(x) > 10.0 and np.sum(x) < 10.5,
                'severity': BugSeverity.MEDIUM,
                'type': BugType.BOUNDARY
            },
            'distribution_shift': {
                'trigger': lambda x: np.mean(x) < -2.0,
                'severity': BugSeverity.HIGH,
                'type': BugType.DISTRIBUTION_SHIFT
            },
            'logic_error': {
                'trigger': lambda x: x[0] * x[1] < -5.0,
                'severity': BugSeverity.MEDIUM,
                'type': BugType.LOGIC_ERROR
            }
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def predict(self, x: np.ndarray) -> Tuple[int, bool, Dict]:
        """Make prediction and check for bugs"""
        x_tensor = torch.FloatTensor(x).unsqueeze(0)
        bug_info = self._check_vulnerabilities(x)

        with torch.no_grad():
            output = self.forward(x_tensor)
            if bug_info['has_bug']:
                output = self._apply_bug_effect(output, bug_info)
            prediction = torch.argmax(output, dim=1).item()

        return prediction, bug_info['has_bug'], bug_info

    def _check_vulnerabilities(self, x: np.ndarray) -> Dict:
        """Check if input triggers any vulnerabilities"""
        bug_info = {
            'has_bug': False,
            'severity': BugSeverity.NONE,
            'type': None,
            'description': None
        }

        for vuln_name, vuln_spec in self.vulnerability_regions.items():
            if vuln_spec['trigger'](x):
                bug_info['has_bug'] = True
                bug_info['severity'] = vuln_spec['severity']
                bug_info['type'] = vuln_spec['type']
                bug_info['description'] = vuln_name
                break

        return bug_info

    def _apply_bug_effect(self, output: torch.Tensor, bug_info: Dict) -> torch.Tensor:
        """Apply bug effects to output"""
        if bug_info['severity'] == BugSeverity.CRITICAL:
            return -output
        elif bug_info['severity'] == BugSeverity.HIGH:
            return output + torch.randn_like(output) * 2.0
        elif bug_info['severity'] == BugSeverity.MEDIUM:
            return output + torch.randn_like(output) * 0.5
        return output


class TargetSystemFactory:
    """Factory for creating target systems"""

    @staticmethod
    def create(system_type: str, **kwargs):
        if system_type == "classifier":
            return TargetClassifier(**kwargs)
        else:
            raise ValueError(f"Unknown system type: {system_type}")