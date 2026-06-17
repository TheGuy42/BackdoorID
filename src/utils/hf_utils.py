from typing import List, TYPE_CHECKING, Optional
from src.utils.better_logging import get_logger

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

logger = get_logger(__name__)


def synchronize_special_tokens(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    verbose: bool = True
) -> None:
    """
    Synchronize special tokens between model config, tokenizer, and generation config.
    Don't forget to set tokenizer.padding_side menually if needed.
    
    This function ensures that EOS, PAD, BOS, and UNK tokens are consistent across:
    - model.config
    - tokenizer
    - model.generation_config
    
    Args:
        model: The AutoModelForCausalLM instance
        tokenizer: The AutoTokenizer instance
        verbose: Whether to log synchronization actions (default: True)
    """
    # Special tokens to synchronize
    special_tokens = {
        'pad_token_id': ('pad_token', 'pad_token_id', True),
        'eos_token_id': ('eos_token', 'eos_token_id', False),
        'bos_token_id': ('bos_token', 'bos_token_id', False),
        'unk_token_id': ('unk_token', 'unk_token_id', False),
    }
    
    # Helper function to get token ID from tokenizer
    def get_token_id(token_name: str) -> Optional[int]:
        token = getattr(tokenizer, token_name, None)
        if token is None:
            return None
        token_id_attr = token_name.replace('_token', '_token_id')
        return getattr(tokenizer, token_id_attr, None)
    
    for config_attr, (tok_attr, tok_id_attr, overrite) in special_tokens.items():
        # Get token ID from tokenizer
        tokenizer_id = get_token_id(tok_attr)
        
        # If tokenizer does not have this token
        if tokenizer_id is None:
            # Special case: if pad_token is not set, try to use eos_token or unk_token
            if config_attr == 'pad_token_id':
                token_name = 'ERROR'
                if tokenizer.eos_token_id is not None:
                    token_name = 'eos_token'
                    replacement_token = tokenizer.eos_token
                    replacement_token_id = tokenizer.eos_token_id
                elif tokenizer.unk_token_id is not None:
                    token_name = 'unk_token'
                    replacement_token = tokenizer.unk_token
                    replacement_token_id = tokenizer.unk_token_id
                else:
                    raise ValueError("Tokenizer does not have pad_token, eos_token, or unk_token set.")
                tokenizer.pad_token = replacement_token
                tokenizer.pad_token_id = replacement_token_id
                tokenizer_id = replacement_token_id  # Update tokenizer_id for synchronization below
                if verbose:
                    logger.info(f"Set pad_token from None to {token_name} (ID: {replacement_token_id})")
            else:
                continue
        
        # Synchronize model config
        model_config_id = getattr(model.config, config_attr, None)
        if model_config_id != tokenizer_id:
            assert tokenizer_id is not None, f"{config_attr} Tokenizer ID should not be None here. model.config id value is {model_config_id}"
            if overrite:
                setattr(model.config, config_attr, tokenizer_id)
                if verbose:
                    logger.info(f"Updated model.config.{config_attr}: {model_config_id} → {tokenizer_id}")
            elif verbose:
                logger.warning(f"model.config.{config_attr} = ({model_config_id}) While tokenizer.{config_attr} = ({tokenizer_id}). But overrite is False.")
        
        # Synchronize generation config if it exists
        if hasattr(model, 'generation_config') and model.generation_config is not None:
            gen_config_id = getattr(model.generation_config, config_attr, None)
            if gen_config_id != tokenizer_id:
                assert tokenizer_id is not None, f"{config_attr} Tokenizer ID should not be None here. model.generation_config id value is {gen_config_id}"
                if overrite:
                    setattr(model.generation_config, config_attr, tokenizer_id)
                    if verbose:
                        logger.info(
                            f"Updated model.generation_config.{config_attr}: {gen_config_id} → {tokenizer_id}"
                        )
                elif verbose:
                    logger.warning(f"model.generation_config.{config_attr} = ({gen_config_id}) While tokenizer.{config_attr} = ({tokenizer_id}). But overrite is False.")

    # Additional synchronization for decoder_start_token_id (useful for seq2seq models)
    if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
        if hasattr(model.config, 'decoder_start_token_id'):
            if model.config.decoder_start_token_id != tokenizer.bos_token_id:
                model.config.decoder_start_token_id = tokenizer.bos_token_id
                if verbose:
                    logger.info(
                        f"Updated model.config.decoder_start_token_id to {tokenizer.bos_token_id}"
                    )
        
        if hasattr(model, 'generation_config') and model.generation_config is not None:
            if hasattr(model.generation_config, 'decoder_start_token_id'):
                if model.generation_config.decoder_start_token_id != tokenizer.bos_token_id:
                    model.generation_config.decoder_start_token_id = tokenizer.bos_token_id
                    if verbose:
                        logger.info(
                            f"Updated model.generation_config.decoder_start_token_id to {tokenizer.bos_token_id}"
                        )
    
    if verbose:
        logger.info("Special tokens synchronization completed")