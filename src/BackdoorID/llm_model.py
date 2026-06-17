import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
# import lightning as L
from typing import Any, Dict, Optional, Tuple, NamedTuple, Callable
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    GenerationMixin,
    GenerationConfig,
)
import logging
from torch.utils.data import Dataset, DataLoader
from peft import PeftModel, PeftConfig

from src.utils.hf_utils import synchronize_special_tokens
from src.utils.better_logging import get_logger
from src.utils.metric_logger import ClearmlLogger
from src.utils.hooks import HookManager, HookConfig
from src.utils.base_config import BaseConfig

logger = get_logger(__name__, logging.WARNING)


class LLMModelConfig(BaseConfig):
    """
    Configuration for a Large Language Model and its tokenizer.

    Attributes:
        model_name_or_path (str): Path to the model checkpoint or Hugging Face model ID.
        tokenizer_name_or_path (Optional[str]): Path to the tokenizer or HF ID. Defaults to model_name_or_path.
        use_fast_tokenizer (bool): Whether to use the Hugging Face fast tokenizer. Defaults to True.
        apply_chat_template (bool): Whether to apply a chat template to inputs. Defaults to False.
        chat_template_args (Dict[str, Any]): Additional arguments for the chat template.
        additional_generation_kwargs (Dict[str, Any]): Parameters for model generation.
        tokenizer_kwargs (Dict[str, Any]): Additional keyword arguments for the tokenizer.
    """
    model_name_or_path: str
    tokenizer_name_or_path: Optional[str] = None
    use_fast_tokenizer: bool = True
    apply_chat_template: bool = False
    chat_template_args: Dict[str, Any] = {}
    additional_generation_kwargs: Dict[str, Any] = {}
    tokenizer_kwargs: Dict[str, Any] = {}

    def model_post_init(self, __context: Any) -> None:
        """Post-processing after initialization to resolve paths."""
        # post-processing after initialization if needed
        if self.tokenizer_name_or_path is None:
            self.tokenizer_name_or_path = self.model_name_or_path
        
        import os
        if self.model_name_or_path.startswith("~"):
            self.model_name_or_path = os.path.expanduser(self.model_name_or_path)
        if self.tokenizer_name_or_path.startswith("~"):
            self.tokenizer_name_or_path = os.path.expanduser(self.tokenizer_name_or_path)

class LLMModelConfigList(BaseConfig):
    """
    A collection of LLMModelConfig objects.

    Attributes:
        model_configs (dict[str, LLMModelConfig]): Dictionary mapping model identifiers to their configurations.
    """
    model_configs: dict[str, LLMModelConfig] = {}



class LLMModel(nn.Module):
    def __init__(
        self,
        model_config: LLMModelConfig,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.model_config = model_config
        self.device = device if device is not None else torch.device("cpu")
        self.llm: GenerationMixin = None
        self.tokenizer: PreTrainedTokenizer = None
        self.generation_config: GenerationConfig = None
        self.hook_manager: HookManager = None
        
        self.load_model()

    def load_model(self, model_config:LLMModelConfig=None) -> None:
        if self.tokenizer is not None or self.llm is not None:
            logger.warning("Model or tokenizer already loaded.")
            if model_config is None:
                logger.info("No ModelConfig passed, skipping loading.")
                return
            else:
                logger.info("Reloading model and tokenizer with new ModelConfig.")
                if self.hook_manager is not None:
                    logger.debug("Removing existing hooks before reloading model.")
                    self.hook_manager.remove_all_hooks()
                    self.hook_manager = None
        
        self.model_config = model_config if model_config is not None else self.model_config
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.tokenizer_name_or_path,
            use_fast=self.model_config.use_fast_tokenizer,
            **self.model_config.tokenizer_kwargs,
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name_or_path,
        ).to(self.device)
        synchronize_special_tokens(self.llm, self.tokenizer)
        self.tokenizer.padding_side = "left"
        self.generation_config = self.llm.generation_config #NOTE: need to verify this does not change unexpectedly

    def process(
        self,
        input: str | list[str],
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Tensor]:
        """
        Process input text (and optional label) into model-ready tensors.
        
        Args:
            input: Input text string or list of strings
            tokenizer_kwargs: Optional additional tokenizer arguments
            
        Returns:
            Dictionary containing processed tensors (input_ids, attention_mask, labels if provided)
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_model() first.")
        
        # Prepare tokenizer kwargs with defaults
        tok_kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
        }
        tok_kwargs.update(self.model_config.tokenizer_kwargs)
        if tokenizer_kwargs is not None:
            tok_kwargs.update(tokenizer_kwargs)
        
        # Apply chat template if configured
        if self.model_config.apply_chat_template:
            if isinstance(input, str):
                input = [input]
            # Convert to chat format
            formatted_input = []
            for text in input:
                messages = [{"role": "user", "content": text}]
                chat_args = {'add_generation_prompt': True, 'tokenize': False} # Default behavior
                chat_args.update(self.model_config.chat_template_args)

                formatted_input.append(self.tokenizer.apply_chat_template(
                    messages,
                    **chat_args
                ))
            input = formatted_input if len(formatted_input) > 1 else formatted_input[0]
        
        # Tokenize input
        encoded = self.tokenizer(input, **tok_kwargs)
        
        # Move to device
        result = {k: v.to(self.device) for k, v in encoded.items()}
        return result

    def decode(
        self,
        token_ids: Tensor,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
        **kwargs
    ) -> str | list[str]:
        """
        Decode token IDs back into text.
        
        Args:
            token_ids: Tensor of token IDs to decode (shape: [batch_size, seq_len] or [seq_len])
            skip_special_tokens: Whether to skip special tokens in output
            clean_up_tokenization_spaces: Whether to clean up extra spaces
            **kwargs: Additional arguments passed to tokenizer.decode/batch_decode
            
        Returns:
            Decoded text string or list of strings
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_model() first.")
        
        # Ensure tensor is on CPU for decoding
        if isinstance(token_ids, Tensor):
            token_ids = token_ids.cpu()
        
        # Handle batch vs single sequence
        if token_ids.dim() == 2:
            # Batch decoding
            return self.tokenizer.batch_decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                **kwargs
            )
        elif token_ids.dim() == 1:
            # Single sequence decoding
            return self.tokenizer.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                **kwargs
            )
        else:
            raise ValueError(f"Expected token_ids to be 1D or 2D tensor, got shape {token_ids.shape}")

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        **kwargs
    ) -> Dict[str, Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs (shape: [batch_size, seq_len])
            attention_mask: Attention mask (shape: [batch_size, seq_len])
            labels: Optional labels for computing loss (shape: [batch_size, seq_len])
            **kwargs: Additional arguments passed to the model
            
        Returns:
            Dictionary containing model outputs (loss, logits, etc.)
        """
        if self.llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Prepare inputs
        model_inputs = {
            "input_ids": input_ids,
        }
        
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        
        if labels is not None:
            model_inputs["labels"] = labels
        
        # Add any additional kwargs
        model_inputs.update(kwargs)
        
        # Forward pass
        outputs = self.llm(**model_inputs)
        
        # Convert outputs to dictionary format
        result = {
            "logits": outputs.logits,
        }
        
        if hasattr(outputs, "loss") and outputs.loss is not None:
            result["loss"] = outputs.loss
        
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            result["hidden_states"] = outputs.hidden_states
        
        if hasattr(outputs, "attentions") and outputs.attentions is not None:
            result["attentions"] = outputs.attentions
        
        return result

    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        **gen_kwargs
    ) -> Tensor:
        """
        Generate tokens based on input IDs.
        
        Args:
            input_ids: Input token IDs (shape: [batch_size, seq_len])
            attention_mask: Attention mask (shape: [batch_size, seq_len])
            **gen_kwargs: Additional generation arguments
            
        Returns:
            Generated token IDs (shape: [batch_size, seq_len + new_tokens])
        """
        if self.llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not hasattr(self.llm, 'generate'):
            raise RuntimeError("Model does not support generation.")
        
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
        
        # Merge with additional generation kwargs from config
        gen_kwargs.update(self.model_config.additional_generation_kwargs)
        
        # Generate
        # with torch.no_grad():
        generated_ids = self.llm.generate(
            input_ids=input_ids,
            generation_config=self.generation_config,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs
        )
        
        return generated_ids

    def load_adapter_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        adapter_config: PeftConfig,
        adapter_name: str = "default",
        is_trainable: bool = False,
    ) -> None:
        """
        Load adapter weights from a specified path.
        Args:
            state_dict: State dictionary containing adapter weights
            adapter_config: PEFT configuration for the adapter
            adapter_name: Name to assign to the loaded adapter (default: "default")
            is_trainable: Whether the loaded adapter should be set to trainable (default: False)
        """
        if self.hook_manager is not None:
            logger.warning("Existing hooks detected.")
            # self.hook_manager.

        try:
            self.llm.load_adapter(
                adapter_name=adapter_name,
                adapter_state_dict=state_dict,
                peft_config=adapter_config,
                is_trainable=is_trainable,
                device_map=self.device,
            )
        except Exception as e: # fix BAIT "smart embedding resize" issue
            logger.error(f"Error loading adapter: {e}")
            adapter_embedding_key = [a for a in state_dict.keys() if "embed_tokens" in a]
            adapter_lmhead_key = [a for a in state_dict.keys() if "lm_head" in a]
            
            if len(adapter_embedding_key) > 0:
                adapter_embedding_key = adapter_embedding_key[0]
                logger.debug(f"Adapter embedding key found: {adapter_embedding_key}")
            if len(adapter_lmhead_key) > 0:
                adapter_lmhead_key = adapter_lmhead_key[0]
                logger.debug(f"Adapter lm_head key found: {adapter_lmhead_key}")

            if isinstance(self.llm, PreTrainedModel):
                if adapter_embedding_key is not None:
                    adapter_embedding = state_dict[adapter_embedding_key]
                    base_embedding_shape = self.llm.get_input_embeddings().weight.shape
                    if adapter_embedding is not None and adapter_embedding.shape != base_embedding_shape:
                        logger.error(f"Adapter embedding shape {adapter_embedding.shape} does not match base model embedding shape {base_embedding_shape}. Reshaping adapter embedding..")
                        state_dict[adapter_embedding_key] = adapter_embedding[:base_embedding_shape[0], :base_embedding_shape[1]]
                        logger.debug(f"Reshaped adapter embedding to {state_dict[adapter_embedding_key].shape}. Retrying load...")
                
                if adapter_lmhead_key is not None:
                    adapter_lmhead = state_dict[adapter_lmhead_key]
                    base_lmhead_shape = self.llm.get_output_embeddings().weight.shape
                    if adapter_lmhead is not None and adapter_lmhead.shape != base_lmhead_shape:
                        logger.error(f"Adapter lm_head shape {adapter_lmhead.shape} does not match base model lm_head shape {base_lmhead_shape}. Reshaping adapter lm_head..")
                        state_dict[adapter_lmhead_key] = adapter_lmhead[:base_lmhead_shape[0], :base_lmhead_shape[1]]
                        logger.debug(f"Reshaped adapter lm_head to {state_dict[adapter_lmhead_key].shape}. Retrying load...")
                    
                    
                    self.remove_adapter(adapter_name=adapter_name) # Remove partially loaded adapter
                    self.llm.load_adapter(
                        adapter_name=adapter_name,
                        adapter_state_dict=state_dict,
                        peft_config=adapter_config,
                        is_trainable=is_trainable,
                        device_map=self.device,
                    )
        logger.info(f"Active adapters after loading: {self.llm.active_adapters()}")
    
    def remove_adapter(self, adapter_name: str = "default") -> None:
        """
        Remove the currently loaded adapter from the model.
        """
        self.llm.delete_adapter(adapter_names=adapter_name)

    def remove_all_adapters(self) -> None:
        """
        Remove all adapters from the model.
        """
        if isinstance(self.llm, PreTrainedModel):
            # Check if there are any adapters to remove
            try:
                adapters = self.llm.active_adapters()
                if adapters and len(adapters) > 0:
                    self.llm.delete_adapter(adapter_names=adapters)
            except (ValueError, AttributeError):
                # No adapters loaded or PEFT not available, silently continue
                pass
        else:
            raise RuntimeError("Please fix me.")

    def add_hooks(self, hook_config: HookConfig|list[HookConfig]) -> None:
        """
        Add hooks to the model based on the provided HookConfig(s).
        
        Args:
            hook_config: Single HookConfig or list of HookConfig instances
        """
        if self.hook_manager is None:
            self.hook_manager = HookManager(self.llm)
        
        
        if isinstance(hook_config, HookConfig):
            hook_config = [hook_config]

        for config in hook_config:
            self.hook_manager.add_hook(config)

    def remove_hooks(self, layer_names: str|list[str]) -> None:
        """
        Remove hooks from specified layers.
        
        Args:
            layer_names: Single layer name or list of layer names to remove hooks from
        """
        if self.hook_manager is None:
            logger.info("No hooks to remove.")
            return
        
        if isinstance(layer_names, str):
            layer_names = [layer_names]
        
        for layer_name in layer_names:
            self.hook_manager.remove_hook(layer_name)

    def to(self, device: torch.device) -> "LLMModel":
        """
        Move the model to the specified device.
        
        Args:
            device: Target device (e.g., torch.device("cuda") or torch.device("cpu"))
            
        Returns:
            Self reference for chaining
        """
        # self.llm.to(device)
        super().to(device)
        self.device = device
        return self





