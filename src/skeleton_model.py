"""
Skeleton Model Module
Defines the skeletal structure and bone connections.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


class SkeletonModel:
    """
    Represents a skeletal model with markers and bone connections.
    
    The skeleton model can be loaded from a YAML configuration file
    or defined programmatically.
    """
    
    def __init__(self):
        self.name: str = "Custom"
        self.description: str = ""
        self.markers: List[str] = []
        self.connections: List[Tuple[str, str]] = []
        self.colors: Dict[str, str] = {"default": "#FFFFFF"}
        self.body_parts: Dict[str, List[str]] = {}
        
    @classmethod
    def from_yaml(
        cls,
        filepath: str,
        marker_set_name: str = "default"
    ) -> "SkeletonModel":
        """
        Load a skeleton model from a YAML configuration file.
        
        Args:
            filepath: Path to the YAML file
            marker_set_name: Name of the marker set to load
            
        Returns:
            SkeletonModel instance
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        
        if marker_set_name not in config:
            available = list(config.keys())
            raise ValueError(
                f"Marker set '{marker_set_name}' not found. "
                f"Available: {available}"
            )
        
        marker_config = config[marker_set_name]
        
        model = cls()
        model.name = marker_config.get("name", marker_set_name)
        model.description = marker_config.get("description", "")
        model.markers = marker_config.get("markers", [])
        model.connections = [
            tuple(conn) for conn in marker_config.get("connections", [])
        ]
        model.colors = marker_config.get("colors", {"default": "#FFFFFF"})
        model.body_parts = marker_config.get("body_parts", {})
        
        return model
    
    @classmethod
    def from_markers(
        cls,
        markers: List[str],
        connections: Optional[List[Tuple[str, str]]] = None,
    ) -> "SkeletonModel":
        """
        Create a skeleton model from a list of markers.
        
        Args:
            markers: List of marker names
            connections: Optional list of bone connections
            
        Returns:
            SkeletonModel instance
        """
        model = cls()
        model.markers = markers
        model.connections = connections or []
        return model
    
    def add_marker(self, name: str) -> "SkeletonModel":
        """Add a marker to the model."""
        if name not in self.markers:
            self.markers.append(name)
        return self
    
    def add_connection(self, marker1: str, marker2: str) -> "SkeletonModel":
        """Add a bone connection between two markers."""
        connection = (marker1, marker2)
        if connection not in self.connections:
            self.connections.append(connection)
        return self
    
    def remove_connection(self, marker1: str, marker2: str) -> "SkeletonModel":
        """Remove a bone connection."""
        connection = (marker1, marker2)
        if connection in self.connections:
            self.connections.remove(connection)
        # Also try reverse order
        connection_rev = (marker2, marker1)
        if connection_rev in self.connections:
            self.connections.remove(connection_rev)
        return self
    
    def set_color(self, body_part: str, color: str) -> "SkeletonModel":
        """Set color for a body part."""
        self.colors[body_part] = color
        return self
    
    def get_marker_color(self, marker: str) -> str:
        """Get the color for a specific marker based on its body part."""
        for part_name, part_markers in self.body_parts.items():
            if marker in part_markers:
                return self.colors.get(part_name, self.colors.get("default", "#FFFFFF"))
        return self.colors.get("default", "#FFFFFF")
    
    def get_connection_color(self, marker1: str, marker2: str) -> str:
        """Get the color for a bone connection."""
        color1 = self.get_marker_color(marker1)
        color2 = self.get_marker_color(marker2)
        # Return the color of the first marker (or could blend)
        return color1 if color1 != "#FFFFFF" else color2
    
    def filter_markers(self, available_markers: List[str]) -> "SkeletonModel":
        """
        Create a new model with only the markers that exist in the data.
        
        Args:
            available_markers: List of markers available in the data
            
        Returns:
            New SkeletonModel with filtered markers
        """
        new_model = SkeletonModel()
        new_model.name = self.name
        new_model.description = self.description
        new_model.colors = self.colors.copy()
        new_model.body_parts = self.body_parts.copy()
        
        # Filter markers
        available_set = set(available_markers)
        new_model.markers = [m for m in self.markers if m in available_set]
        
        # Filter connections - only keep if both markers exist
        new_model.connections = [
            conn for conn in self.connections
            if conn[0] in available_set and conn[1] in available_set
        ]
        
        return new_model
    
    def get_valid_connections(
        self,
        marker_positions: Dict[str, Any]
    ) -> List[Tuple[str, str]]:
        """
        Get connections where both markers have valid (non-None) positions.
        
        Args:
            marker_positions: Dictionary of marker names to positions
            
        Returns:
            List of valid connections
        """
        return [
            conn for conn in self.connections
            if marker_positions.get(conn[0]) is not None
            and marker_positions.get(conn[1]) is not None
        ]
    
    def to_yaml(self, filepath: str, set_name: str = "custom") -> None:
        """
        Save the skeleton model to a YAML file.
        
        Args:
            filepath: Path to save the YAML file
            set_name: Name for this marker set in the file
        """
        config = {
            set_name: {
                "name": self.name,
                "description": self.description,
                "markers": self.markers,
                "connections": [list(conn) for conn in self.connections],
                "colors": self.colors,
                "body_parts": self.body_parts,
            }
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def __repr__(self) -> str:
        return (
            f"SkeletonModel(name='{self.name}', "
            f"markers={len(self.markers)}, "
            f"connections={len(self.connections)})"
        )
