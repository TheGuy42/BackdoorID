"""
Hook management utilities for PyTorch models.
"""

from typing import Dict, Callable, Optional, Any, List, NamedTuple
import torch.nn as nn
from .better_logging import get_logger


class HookConfig(NamedTuple):
    layer_name: str
    hook_function: Callable
    hook_type: str = 'post_forward' # "pre_forward", "post_forward", "pre_backward", "post_backward"
    enable: bool = True



class HookManager:
    """
    Manages hooks for PyTorch model layers.
    
    Supports adding, removing, enabling/disabling hooks, and context manager usage.
    Only one hook per layer is allowed, identified by the layer name.
    
    Example:
        >>> model = MyModel()
        >>> hook_manager = HookManager(model)
        >>> 
        >>> # Add hooks
        >>> hook_manager.add_hook("layer1", my_hook_function)
        >>> hook_manager.add_hook("layer2", another_hook_function)
        >>> 
        >>> # Use as context manager
        >>> with hook_manager:
        >>>     output = model(input)
        >>> 
        >>> # Or manage manually
        >>> hook_manager.disable_hook("layer1")
        >>> hook_manager.enable_hook("layer1")
        >>> hook_manager.remove_hook("layer2")
    """
    
    def __init__(self, model: nn.Module, verbose: bool = False):
        """
        Initialize the HookManager.
        
        Args:
            model: PyTorch model to manage hooks for
            verbose: If True, log detailed information about hook operations
        """
        self.model = model
        self.verbose = verbose
        self.logger = get_logger(self.__class__.__name__)
        self._hooks: Dict[str, Dict[str, Any]] = {}
        # Dictionary structure: {layer_name: {'function': hook_fn, 'handle': handle, 'enabled': bool}}
        self._saved_states: Optional[Dict[str, bool]] = None  # Store hook states for context manager
        
        if self.verbose:
            self.logger.info(f"HookManager initialized for model: {model.__class__.__name__}")
        
    def add_hook(
        self, 
        config: HookConfig
    ) -> None:
        """
        Add a named hook to a specific layer.
        
        Args:
            layer_name: Name of the layer to attach the hook to
            hook_function: Hook function to attach
            hook_type: Type of hook - "pre_forward", "post_forward", "pre_backward", "post_backward" (default: "post_forward")
            enable: Whether to enable the hook immediately (default: True)
            
        Raises:
            ValueError: If layer_name doesn't exist in the model
            ValueError: If a hook already exists for this layer (remove it first)
            ValueError: If hook_type is invalid
        """
        # Get the layer from the model to validate it exists
        layer = self._get_layer(config.layer_name)
        
        # Check if a hook already exists for this layer
        if config.layer_name in self._hooks:
            raise ValueError(
                f"Hook already exists for layer '{config.layer_name}'. "
                f"Remove it first with remove_hook('{config.layer_name}')"
            )
        
        # Validate hook type
        valid_types = ["pre_forward", "post_forward", "pre_backward", "post_backward"]
        if config.hook_type not in valid_types:
            raise ValueError(
                f"Invalid hook_type: {config.hook_type}. Must be one of {valid_types}"
            )
        
        # Store hook information without registering yet
        self._hooks[config.layer_name] = {
            'function': config.hook_function,
            'handle': None,
            'enabled': False,
            'type': config.hook_type
        }
        
        if self.verbose:
            self.logger.info(f"Added hook to layer '{config.layer_name}' (type: {config.hook_type}, enable: {config.enable})")
        
        # Enable the hook if requested
        if config.enable:
            self.enable_hook(config.layer_name)
    
    def remove_hook(self, layer_name: str) -> None:
        """
        Remove a specific hook by layer name.
        
        Args:
            layer_name: Name of the layer whose hook should be removed
            
        Raises:
            KeyError: If no hook exists for the specified layer
        """
        if layer_name not in self._hooks:
            raise KeyError(f"No hook found for layer '{layer_name}'")
        
        # Remove the hook
        if self._hooks[layer_name]['handle'] is not None:
            self._hooks[layer_name]['handle'].remove()
        del self._hooks[layer_name]
        
        if self.verbose:
            self.logger.info(f"Removed hook from layer '{layer_name}'")
    
    def remove_all_hooks(self) -> None:
        """Remove all hooks from all layers."""
        layer_names = list(self._hooks.keys())
        if self.verbose and layer_names:
            self.logger.info(f"Removing all hooks ({len(layer_names)} total)")
        for layer_name in layer_names:
            self.remove_hook(layer_name)
    
    def disable_hook(self, layer_name: str) -> None:
        """
        Disable a specific hook without removing it.
        
        Args:
            layer_name: Name of the layer whose hook should be disabled
            
        Raises:
            KeyError: If no hook exists for the specified layer
        """
        if layer_name not in self._hooks:
            raise KeyError(f"No hook found for layer '{layer_name}'")
        
        if self._hooks[layer_name]['enabled']:
            self._hooks[layer_name]['handle'].remove()
            self._hooks[layer_name]['enabled'] = False
            if self.verbose:
                self.logger.info(f"Disabled hook for layer '{layer_name}'")
    
    def enable_hook(self, layer_name: str) -> None:
        """
        Re-enable a previously disabled hook.
        
        Args:
            layer_name: Name of the layer whose hook should be enabled
            
        Raises:
            KeyError: If no hook exists for the specified layer
        """
        if layer_name not in self._hooks:
            raise KeyError(f"No hook found for layer '{layer_name}'")
        
        if not self._hooks[layer_name]['enabled']:
            # Re-register the hook
            layer = self._get_layer(layer_name)
            hook_function = self._hooks[layer_name]['function']
            hook_type = self._hooks[layer_name]['type']
            
            # Register based on hook type
            if hook_type == "pre_forward":
                handle = layer.register_forward_pre_hook(hook_function)
            elif hook_type == "post_forward":
                handle = layer.register_forward_hook(hook_function)
            elif hook_type == "pre_backward":
                handle = layer.register_full_backward_pre_hook(hook_function)
            elif hook_type == "post_backward":
                handle = layer.register_full_backward_hook(hook_function)
            else:
                raise ValueError(f"Unknown hook_type: {hook_type}")
            
            self._hooks[layer_name]['handle'] = handle
            self._hooks[layer_name]['enabled'] = True
            if self.verbose:
                self.logger.info(f"Enabled hook for layer '{layer_name}' (type: {hook_type})")
    
    def disable_all_hooks(self) -> None:
        """Disable all hooks without removing them."""
        enabled_hooks = [name for name, info in self._hooks.items() if info['enabled']]
        if self.verbose and enabled_hooks:
            self.logger.info(f"Disabling all hooks ({len(enabled_hooks)} enabled)")
        for layer_name in list(self._hooks.keys()):
            if self._hooks[layer_name]['enabled']:
                self.disable_hook(layer_name)
    
    def enable_all_hooks(self) -> None:
        """Enable all previously disabled hooks."""
        disabled_hooks = [name for name, info in self._hooks.items() if not info['enabled']]
        if self.verbose and disabled_hooks:
            self.logger.info(f"Enabling all hooks ({len(disabled_hooks)} disabled)")
        for layer_name in list(self._hooks.keys()):
            if not self._hooks[layer_name]['enabled']:
                self.enable_hook(layer_name)
    
    def _get_layer(self, layer_name: str) -> nn.Module:
        """
        Get a layer from the model by name.
        
        Args:
            layer_name: Name of the layer (supports dot notation for nested layers)
            
        Returns:
            The requested layer module
            
        Raises:
            ValueError: If the layer doesn't exist in the model
        """
        # Handle nested layer names (e.g., "encoder.layer.0")
        parts = layer_name.split('.')
        layer = self.model
        
        for part in parts:
            if hasattr(layer, part):
                layer = getattr(layer, part)
            else:
                raise ValueError(
                    f"Layer '{layer_name}' not found in model. "
                    f"Failed at part '{part}'"
                )
        
        return layer
    
    def get_hook_info(self, layer_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about registered hooks.
        
        Args:
            layer_name: Optional layer name. If None, returns info for all hooks
            
        Returns:
            Dictionary with hook information (enabled status, type, etc.)
        """
        if layer_name is not None:
            if layer_name not in self._hooks:
                raise KeyError(f"No hook found for layer '{layer_name}'")
            return {
                'enabled': self._hooks[layer_name]['enabled'],
                'type': self._hooks[layer_name]['type']
            }
        else:
            return {
                name: {
                    'enabled': info['enabled'],
                    'type': info['type']
                }
                for name, info in self._hooks.items()
            }
    
    def __enter__(self, layer_names: Optional[List[str]] = None):
        """
        Enable hooks when entering context.
        
        Args:
            layer_names: Optional list of layer names to enable. If None, enables all hooks.
                        If provided, saves current state, disables all hooks, then enables only specified ones.
        
        Returns:
            self for context manager usage
        """
        if self.verbose:
            if layer_names is None:
                self.logger.info("Entering context: enabling all hooks")
            else:
                self.logger.info(f"Entering context: enabling hooks for {layer_names}")
        
        if layer_names is None:
            # Enable all hooks without state management
            self.enable_all_hooks()
        else:
            # Save current state of all hooks
            self._saved_states = {name: info['enabled'] for name, info in self._hooks.items()}
            
            # Disable all hooks
            self.disable_all_hooks()
            
            # Enable only specified hooks
            for layer_name in layer_names:
                if layer_name in self._hooks:
                    self.enable_hook(layer_name)
                elif self.verbose:
                    self.logger.warning(f"Layer '{layer_name}' not found in registered hooks")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit context and restore hook states.
        
        If layer_names were specified in __enter__, restores the previous state of all hooks.
        Otherwise, removes all hooks.
        """
        if self._saved_states is not None:
            # Restore previous hook states
            if self.verbose:
                self.logger.info("Exiting context: restoring previous hook states")
            
            # First disable all hooks
            self.disable_all_hooks()
            
            # Then restore each hook to its previous state
            for layer_name, was_enabled in self._saved_states.items():
                if layer_name in self._hooks and was_enabled:
                    self.enable_hook(layer_name)
            
            # Clear saved states
            self._saved_states = None
        else:
            # No saved states, remove all hooks (original behavior)
            if self.verbose:
                self.logger.info("Exiting context: removing all hooks")
            self.remove_all_hooks()
        
        return False
    
    def __len__(self) -> int:
        """Return the number of registered hooks."""
        return len(self._hooks)
    
    def __contains__(self, layer_name: str) -> bool:
        """Check if a hook exists for the given layer."""
        return layer_name in self._hooks
    
    def __repr__(self) -> str:
        """String representation of the HookManager."""
        enabled_count = sum(1 for info in self._hooks.values() if info['enabled'])
        return (
            f"HookManager(total_hooks={len(self._hooks)}, "
            f"enabled={enabled_count}, disabled={len(self._hooks) - enabled_count})"
        )
