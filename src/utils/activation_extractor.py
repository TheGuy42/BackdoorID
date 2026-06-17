"""
Activation extraction utilities for PyTorch models using HookManager.
"""

from typing import Dict, List, Optional, Callable
import torch
import torch.nn as nn
from .hooks import HookManager, HookConfig
from .better_logging import get_logger


class ActivationExtractor:
    """
    Extracts and stores activations from specified layers of a PyTorch model.
    
    Uses HookManager to attach hooks to model layers and collect their output activations.
    Activations are stored as a list of dictionaries, where each dictionary represents a single
    forward pass with keys as layer names and values as the corresponding activation tensors.
    The order of the list maintains the order activations were collected.
    
    Example:
        >>> model = MyModel()
        >>> extractor = ActivationExtractor(model, ["layer1", "layer2.conv"])
        >>> 
        >>> # Use as context manager
        >>> with extractor:
        >>>     output = model(input)
        >>>     activations = extractor.get_activations()
        >>> 
        >>> # Or manage manually
        >>> extractor.enable()
        >>> output = model(input)
        >>> activations = extractor.get_activations()
        >>> extractor.disable()
        >>> extractor.reset()
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        layer_names: List[str],
        hook_type: str = 'post_forward',
        device: Optional[str] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        verbose: bool = False
    ):
        """
        Initialize the ActivationExtractor.
        
        Args:
            model: PyTorch model to extract activations from
            layer_names: List of layer names to monitor
            hook_type: Type of hook - "pre_forward", "post_forward", "pre_backward", "post_backward" (default: "post_forward")
            device: Device to move activations to (e.g., 'cpu', 'cuda'). If None, activations stay on their original device (default: None)
            transform: Optional transformation function to apply to each activation tensor. Takes a tensor and returns a tensor (default: None)
            verbose: If True, log detailed information about operations
        """
        self.model = model
        self.layer_names = layer_names
        self.hook_type = hook_type
        self.device = device
        self.transform = transform
        self.verbose = verbose
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize activation storage as list of dictionaries
        self._activations: List[Dict[str, torch.Tensor]] = []
        # Track current capture index for synchronization across layers
        self._current_capture: Dict[str, torch.Tensor] = {}
        self._capture_lock = False  # Simple flag to manage capture synchronization
        
        # Initialize HookManager
        self.hook_manager = HookManager(model, verbose=verbose)
        
        # Register hooks for each layer
        self._register_hooks()
        
        if self.verbose:
            self.logger.info(
                f"ActivationExtractor initialized for {len(layer_names)} layers: {layer_names}"
            )
    
    def _register_hooks(self) -> None:
        """Register hooks for all specified layers."""
        for layer_name in self.layer_names:
            # Create hook function for this layer
            hook_fn = self._create_hook_function(layer_name)
            
            # Create hook configuration
            config = HookConfig(
                layer_name=layer_name,
                hook_function=hook_fn,
                hook_type=self.hook_type,
                enable=False  # Don't enable by default
            )
            
            # Add hook to manager
            self.hook_manager.add_hook(config)
    
    def _create_hook_function(self, layer_name: str):
        """
        Create a hook function for a specific layer.
        
        Args:
            layer_name: Name of the layer this hook is for
            
        Returns:
            Hook function that captures activations
        """
        def hook_fn(module, input, output):
            """Hook function that stores the output activation."""
            # Detach and clone to avoid keeping computation graph
            if isinstance(output, torch.Tensor):
                activation = output.detach().clone()
                # Move to specified device if provided
                if self.device is not None:
                    activation = activation.to(self.device)
                # Apply transformation if provided
                if self.transform is not None:
                    activation = self.transform(activation)
            elif isinstance(output, tuple):
                # Handle tuple outputs (some layers return multiple tensors)
                if self.device is not None:
                    activation = tuple(
                        o.detach().clone().to(self.device) if isinstance(o, torch.Tensor) else o 
                        for o in output
                    )
                else:
                    activation = tuple(
                        o.detach().clone() if isinstance(o, torch.Tensor) else o 
                        for o in output
                    )
                # Apply transformation to tuple elements if provided
                if self.transform is not None:
                    activation = tuple(
                        self.transform(o) if isinstance(o, torch.Tensor) else o
                        for o in activation
                    )
            else:
                activation = output
            
            # Assert that this layer hasn't been captured yet in the current cycle
            assert layer_name not in self._current_capture, (
                f"Layer '{layer_name}' was captured twice in the same forward pass. "
                f"This indicates multiple forward passes through the same layer before all layers completed. "
                f"Current capture contains: {list(self._current_capture.keys())}"
            )
            
            # Store activation in current capture dictionary
            self._current_capture[layer_name] = activation
            
            # If all layers have been captured, append to activations list
            if len(self._current_capture) == len(self.layer_names):
                # Assert all expected layers are present
                missing_layers = set(self.layer_names) - set(self._current_capture.keys())
                assert len(missing_layers) == 0, (
                    f"Expected all {len(self.layer_names)} layers to be captured, "
                    f"but missing: {missing_layers}. "
                    f"Captured: {list(self._current_capture.keys())}"
                )
                
                self._activations.append(self._current_capture.copy())
                self._current_capture.clear()
            
            if self.verbose:
                if isinstance(activation, torch.Tensor):
                    self.logger.debug(
                        f"Captured activation from '{layer_name}': shape={activation.shape}"
                    )
                else:
                    self.logger.debug(f"Captured activation from '{layer_name}'")
        
        return hook_fn
    
    def get_activations(self, layer_name: Optional[str] = None) -> List[Dict[str, torch.Tensor]]:
        """
        Get collected activations.
        
        Args:
            layer_name: Optional layer name. If None, returns all activations.
                       If specified, returns activations only for that layer (still as list of dicts).
        
        Returns:
            List of dictionaries where each dict maps layer names to activation tensors for that capture
            
        Raises:
            KeyError: If specified layer_name doesn't exist in layer_names
        """
        if layer_name is not None:
            if layer_name not in self.layer_names:
                raise KeyError(f"No layer named '{layer_name}' in extractor")
            # Return only the specified layer from each capture
            return [{layer_name: capture[layer_name]} for capture in self._activations if layer_name in capture]
        return self._activations.copy()
    
    def reset(self, layer_name: Optional[str] = None) -> None:
        """
        Reset (clear) collected activations.
        
        Args:
            layer_name: Optional layer name. If None, resets all activations.
                       If specified, removes only that layer's data from all captures.
                       
        Raises:
            KeyError: If specified layer_name doesn't exist
        """
        if layer_name is not None:
            if layer_name not in self.layer_names:
                raise KeyError(f"No layer named '{layer_name}' in extractor")
            # Remove the layer from all captures
            for capture in self._activations:
                if layer_name in capture:
                    del capture[layer_name]
            if self.verbose:
                self.logger.info(f"Reset activations for layer '{layer_name}'")
        else:
            self._activations.clear()
            self._current_capture.clear()
            if self.verbose:
                self.logger.info(f"Reset all activations for {len(self.layer_names)} layers")
    
    def enable(self, layer_name: Optional[str] = None) -> None:
        """
        Enable activation collection.
        
        Args:
            layer_name: Optional layer name. If None, enables all layers.
                       If specified, enables only that layer.
        """
        if layer_name is not None:
            self.hook_manager.enable_hook(layer_name)
            if self.verbose:
                self.logger.info(f"Enabled activation extraction for layer '{layer_name}'")
        else:
            self.hook_manager.enable_all_hooks()
            if self.verbose:
                self.logger.info(f"Enabled activation extraction for all {len(self.layer_names)} layers")
    
    def disable(self, layer_name: Optional[str] = None) -> None:
        """
        Disable activation collection.
        
        Args:
            layer_name: Optional layer name. If None, disables all layers.
                       If specified, disables only that layer.
        """
        if layer_name is not None:
            self.hook_manager.disable_hook(layer_name)
            if self.verbose:
                self.logger.info(f"Disabled activation extraction for layer '{layer_name}'")
        else:
            self.hook_manager.disable_all_hooks()
            if self.verbose:
                self.logger.info(f"Disabled activation extraction for all {len(self.layer_names)} layers")
    
    def get_activation_count(self, layer_name: Optional[str] = None) -> Dict[str, int]:
        """
        Get count of collected activations.
        
        Args:
            layer_name: Optional layer name. If None, returns counts for all layers.
        
        Returns:
            Dictionary of layer names to activation counts
        """
        if layer_name is not None:
            if layer_name not in self.layer_names:
                raise KeyError(f"No layer named '{layer_name}' in extractor")
            count = sum(1 for capture in self._activations if layer_name in capture)
            return {layer_name: count}
        
        # Count activations per layer across all captures
        counts = {name: 0 for name in self.layer_names}
        for capture in self._activations:
            for layer_name in capture:
                counts[layer_name] += 1
        return counts
    
    def __enter__(self):
        """Enable hooks when entering context."""
        if self.verbose:
            self.logger.info("Entering context: enabling activation extraction")
        self.enable()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Disable hooks when exiting context."""
        if self.verbose:
            self.logger.info("Exiting context: disabling activation extraction")
        self.disable()
        return False
    
    def __len__(self) -> int:
        """Return total number of capture cycles (forward passes)."""
        return len(self._activations)
    
    def __repr__(self) -> str:
        """String representation of the ActivationExtractor."""
        total_activations = len(self)
        enabled_count = sum(
            1 for name in self.layer_names 
            if self.hook_manager.get_hook_info(name)['enabled']
        )
        return (
            f"ActivationExtractor(layers={len(self.layer_names)}, "
            f"enabled={enabled_count}, total_activations={total_activations})"
        )
