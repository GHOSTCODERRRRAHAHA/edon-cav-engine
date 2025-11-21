"""Transport layer abstraction for EDON SDK."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Iterator
from enum import Enum


class TransportType(Enum):
    """Transport layer types."""
    REST = "rest"
    GRPC = "grpc"


class Transport(ABC):
    """Abstract base class for transport implementations."""
    
    @abstractmethod
    def compute_cav(self, window: Dict[str, Any]) -> Dict[str, Any]:
        """Compute CAV from a sensor window."""
        pass
    
    @abstractmethod
    def stream(self, window: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Stream CAV updates (server push)."""
        pass
    
    @abstractmethod
    def health(self) -> Dict[str, Any]:
        """Check service health."""
        pass

