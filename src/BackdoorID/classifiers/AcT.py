import torch
from torch import nn
import logging

from .classifier_base import ClassifierBase
from src.utils.base_config import BaseConfig
from src.utils.better_logging import get_logger

logger = get_logger(__name__, level=logging.INFO)


class DynamicTanh(nn.Module):
    """
    Dynamic Tanh (DyT) layer that replaces normalization layers.
    
    Computes: DyT(x) = γ · tanh(α · x) + β
    
    Args:
        num_features: Number of features (channels). For transformers, this is 
                      typically the model dimension (d_model).
        alpha_init: Initial value for the scalar parameter α (default: 1.0)
    """
    def __init__(self, num_features, alpha_init=1.0):
        super().__init__()
        
        # Learnable scalar parameter α
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        
        # Learnable per-channel parameters γ and β
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # Add eps attribute for compatibility with LayerNorm interface
        self.eps = 1e-5
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (..., num_features)
            
        Returns:
            Output tensor of same shape as input
        """
        return self.gamma * torch.tanh(self.alpha * x) + self.beta


class DyTTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    Transformer Encoder Layer with Dynamic Tanh (DyT) replacing normalization layers.
    
    Inherits from TransformerEncoderLayer but replaces both LayerNorm layers
    with DynamicTanh and uses DynamicTanh as the FFN activation function.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', 
                 batch_first=True, norm_first=True, **kwargs):
        # Initialize parent with DynamicTanh activation for FFN
        # Note: FFN activation operates on dim_feedforward
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first,
            norm_first=norm_first,
            **kwargs
        )
        
        # Replace normalization layers with DynamicTanh
        self.norm1 = DynamicTanh(d_model)  # After self-attention
        self.norm2 = DynamicTanh(d_model)  # After feedforward
        self.norm2.eps = self.norm1.eps + 1e-7  # Ensure eps is slightly different to avoid fast_path


class ActivationTransformerConfig(BaseConfig):
    """
    Configuration for the ActivationTransformer.

    Attributes:
        block_size (int): Size of input blocks for the first transformer stage. Defaults to 512.
        d_model (int): Hidden dimension of the transformer layers. Defaults to 128.
        dim_feedforward (int): Dimension of the feedforward network within blocks. Defaults to 256.
        nhead1 (int): Number of attention heads of the activation encoder transformer (1st stage). Defaults to 4.
        nhead2 (int): Number of attention heads in the classification head transformer (2nd stage). Defaults to 4.
        num_layers_1 (int): Number of encoder layers in the first stage. Defaults to 4.
        num_layers_2 (int): Number of encoder layers in the second stage. Defaults to 4.
        dropout (float): Dropout probability. Defaults to 0.15.
        num_classes (int): Number of output classes. Defaults to 1.
        norm_first (bool): Whether to apply normalization before attention. Defaults to True.
        layer_norm_eps (float): Epsilon value for layer normalization. Defaults to 1e-5.
        cls_token (bool): Whether to prepend a [CLS] token for classification. Defaults to True.
        latent_aggregation (bool): Whether to perform aggregation in the latent space. Defaults to True.
        aggregation_type (str): Type of aggregation to use ('mean', 'max', etc.). Defaults to 'mean'. Deprecated, used when latent_aggregation is False.
        activation (str): Activation function to use ('relu', 'gelu', 'dyt'). Defaults to 'relu'.
        normalization (str): Normalization type to use ('layernorm', 'dyt'). Defaults to 'layernorm'.
        input_projection (bool): If True, projects input features to d_model dimension. Defaults to True.
        init_std (float): Standard deviation for parameter initialization. Defaults to 1e-5.
    """
    block_size: int = 512 # split input into blocks of this size
    d_model: int = 128
    dim_feedforward: int = 256
    nhead1: int = 4
    nhead2: int = 4
    num_layers_1: int = 4
    num_layers_2: int = 4
    dropout: float = 0.15
    num_classes: int = 1
    norm_first: bool = True
    layer_norm_eps: float = 1e-5
    cls_token: bool = True
    latent_aggregation: bool = True
    aggregation_type: str = 'mean'  # 'mean', 'max', 'mean_poisoned', 'max_poisoned'
    activation: str = 'relu'  # 'relu', 'gelu', 'dyt', etc.
    normalization: str = 'layernorm'  # 'layernorm', 'dyt'
    input_projection: bool = True  # Whether to use input projection layer to project features to d_model
    init_std: float = 1e-5  # Standard deviation for truncated normal initialization of parameters


class ActivationTransformer(ClassifierBase):
    """
    Transformer-based classifier for model representations.
    
    Uses TransformerEncoder layers to attend across samples, treating
    each sample's representation as a token in the sequence dimension.

    Attributes:
        config (ActivationTransformerConfig): Configuration for the hierarchical transformer.
    """
    
    def __init__(self, config: ActivationTransformerConfig):
        super().__init__()
        self.config = config
        
        # Input projection layer (LazyLinear to handle variable input dimensions)
        # self.layer_norm = nn.LayerNorm(config.block_size, eps=config.layer_norm_eps)
        if config.input_projection:
            self.input_projection = nn.LazyLinear(config.d_model)
        else:
            assert config.d_model == config.block_size, "If input_projection is False, d_model must equal block_size."

        # Process activation parameter
        if config.activation == 'dyt':
            activation = DynamicTanh(config.dim_feedforward)
        else:
            activation = config.activation

        # Select encoder layer type based on normalization parameter
        if config.normalization == 'dyt':
            layers_encoder_layer = DyTTransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead1,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation=activation,
                norm_first=config.norm_first,
                batch_first=True
            )
            samples_encoder_layer = DyTTransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead2,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation=activation,
                norm_first=config.norm_first,
                batch_first=True
            )
        else:
            layers_encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead1,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation=activation,
                norm_first=config.norm_first,
                layer_norm_eps=config.layer_norm_eps,
                batch_first=True  # We use (batch, seq_len, features) format
            )
            samples_encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead2,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation=activation,
                norm_first=config.norm_first,
                layer_norm_eps=config.layer_norm_eps,
                batch_first=True
            )

        self.layers_transformer = nn.TransformerEncoder(
            encoder_layer=layers_encoder_layer,
            num_layers=config.num_layers_1,
        )

        self.samples_transformer = nn.TransformerEncoder(
            encoder_layer=samples_encoder_layer,
            num_layers=config.num_layers_2,
        )

        # Output classifier head - use DyT normalization if specified
        if config.normalization == 'dyt':
            classifier_norm = DynamicTanh(config.d_model)
        else:
            classifier_norm = nn.LayerNorm(config.d_model)

        self.classifier = nn.Sequential(
            classifier_norm,
            nn.Linear(config.d_model, config.num_classes)
        )

        init_std = self.config.init_std
        # self.CLS_TOKEN = torch.zeros(1, 1, config.d_model)  # (1, 1, d_model)
        cls_data = torch.nn.init.trunc_normal_(torch.zeros(1, 1, config.d_model), std=init_std)
        self.CLS_TOKEN = nn.Parameter(cls_data, requires_grad=True)  # (1, 1, d_model)

        cls_latent_data = torch.nn.init.trunc_normal_(torch.zeros(1, 1, config.d_model), std=init_std)
        self.CLS_TOKEN_LATENT = nn.Parameter(cls_latent_data, requires_grad=True)  # (1, 1, d_model)

        assert config.d_model % 2 == 0, "d_model must be even for positional encoding splitting"
        # (batch_size, n_layers, num_blocks, d_model)
        self.block_pos_encoding = nn.Parameter(torch.zeros(1, 1, 1000, config.d_model//2))  # Max blocks of 1000
        torch.nn.init.trunc_normal_(self.block_pos_encoding, std=init_std)
        self.layer_pos_encoding = nn.Parameter(torch.zeros(1, 1000, 1, config.d_model//2))  # Max layers of 1000
        torch.nn.init.trunc_normal_(self.layer_pos_encoding, std=init_std)
    
    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the classifier.
        
        Args:
            x: Dictionary mapping layer names to activation tensors. 
               Each tensor shape: (batch_size, cls_inputs_per_sample, feature_dim)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Aggregate activations: (batch_size, n_layers, total_feature_dim)
        x:torch.Tensor = self.activation_aggregation(x)

        x = self.split_to_blocks(x)  # (batch_size, n_layers, num_blocks, block_size)

        if self.config.input_projection:
            # downscale to (batch_size, num_blocks, d_model)
            x = self.input_projection(x)  # (batch_size, n_layers, num_blocks, d_model)

        # add positional encoding: block encoding in upper half, layer encoding in lower half of d_model
        half_d = x.size(-1) // 2
        x = torch.cat([
            x[..., :half_d] + self.block_pos_encoding[:, :, :x.size(2), :],   # upper half (block pos)
            x[..., half_d:] + self.layer_pos_encoding[:, :x.size(1), :, :]    # lower half (layer pos)
        ], dim=-1)

        # add layers to sequence dimension (batch_size, n_layers * num_blocks, block_size)
        x = x.view(x.size(0), x.size(1) * x.size(2), x.size(3))  # (batch_size, n_layers * num_blocks, d_model)

        logger.debug(f"Input shape after block splitting and positional encoding: {x.shape}")

        # Add CLS token at the beginning of the sequence dimension: # (batch_size, n_layers * num_blocks + 1, block_size)
        if self.config.cls_token:
            x = self.add_cls_token(x, self.CLS_TOKEN)  # (batch_size, n_layers * num_blocks + 1, block_size)

        # Pass through transformer encoder: (batch_size, n_layers * num_blocks + 1, block_size)
        # Explicit all-zeros additive mask to enforce fully bidirectional attention across all tokens
        x = self.layers_transformer.forward(x)  # (batch_size, n_layers * num_blocks + 1, block_size)

        cls_token_representation = x[:, 0, :]  # (batch_size, d_model)
        logger.debug(f"CLS token representation shape after first transformer: {cls_token_representation.shape}")
        
        if self.config.latent_aggregation:
            return cls_token_representation # Return latent representation if specified (batch_size, d_model)
        else:
            # Output shape: (1, num_classes)
            latent_cls_token_representation = self.forward_samples_transformer(cls_token_representation)  # (1, d_model)
            return self.classifier(latent_cls_token_representation)  # (1, num_classes)
        
    def forward_samples_transformer(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the samples transformer, which attends across batch samples.
        
        Args:
            x: Input tensor of shape (batch_size, d_model) where batch_size is the number of samples in the activation dataset.

        Returns:
            Output tensor of shape (1, d_model) representing the latent CLS token after attending across samples.
        """
        logger.debug(f"Input shape to samples transformer: {x.shape}")

        x = x.unsqueeze(0)  # (1, batch_size, d_model)
    
        if self.config.cls_token:
            # Add latent CLS token to representation for second transformer to attend to: (1, batch_size + 1, d_model)
            x = self.add_cls_token(x, self.CLS_TOKEN_LATENT)  # (1, batch_size + 1, d_model)

        logger.debug(f"Input shape to samples transformer after adding latent CLS token: {x.shape}")

        # pass through second transformer to attend across samples
        # Explicit all-zeros additive mask to enforce fully bidirectional attention across all samples
        x = self.samples_transformer.forward(x)  # (1, batch_size + 1, d_model)

        logger.debug(f"Output shape from samples transformer: {x.shape}")

        latent_cls_token_representation = x[:, 0, :] # (1, d_model)

        logger.debug(f"Latent CLS token representation shape after samples transformer: {latent_cls_token_representation.shape}")

        return latent_cls_token_representation

    def add_cls_token(self, x: torch.Tensor, cls_token: torch.Tensor) -> torch.Tensor:
        """
        Add a CLS token to the beginning of the sequence dimension.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, feature_dim)
            cls_token: CLS token tensor of shape (1, 1, feature_dim)
        
        Returns:
            Tensor of shape (batch_size, seq_len + 1, feature_dim) with CLS token added at the beginning.
        """
        if cls_token.device != x.device:
            cls_token = cls_token.to(x.device)
        batch_size, seq_len, feature_dim = x.shape
        assert cls_token.shape == (1, 1, feature_dim), f"CLS token shape mismatch. Expected (1, 1, {feature_dim}), got {cls_token.shape}"
        cls_expanded = cls_token.expand(batch_size, 1, feature_dim)  # (batch_size, 1, feature_dim)
        x = torch.cat([cls_expanded, x], dim=1)  # (batch_size, seq_len + 1, feature_dim)
        return x
        
    def prediction_head(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the prediction head to the input tensor.
        
        Args:
            x: Input tensor of shape (total_samples, output_dim)
            Where:
            - total_samples is the totan number of samples in the activation dataset.
            - output_dim is the dimension of the classifier output.

        Returns:
            Logits of shape (1, num_classes)
        """
        if self.config.latent_aggregation:
            x = self.forward_samples_transformer(x)  # (1, d_model)
            return self.classifier(x)
        else: 
            return self.logit_aggregation(x, aggregation_type=self.config.aggregation_type)
    
    def split_to_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the input tensor into blocks along the batch dimension.
        
        Args:
            x: Input tensor of shape (n_samples, n_layers, total_feature_dim)
        
        Returns:
            Tensor of shape (batch_size, n_layers, num_blocks, block_size) where block_size is defined in config.
        """
        n_samples, n_layers, feature_dim = x.shape
        block_size = self.config.block_size

        if feature_dim % block_size != 0:
            logger.debug(f"Feature dimension {feature_dim} is not divisible by block size {block_size}. Dropping last {feature_dim % block_size} features.")
            x = x[:, :, :-(feature_dim % block_size)]  # Drop last few features to make it divisible
            feature_dim = x.shape[2]

        x = x.unsqueeze(-2)  # (n_samples, n_layers, 1, feature_dim)
        
        # Reshape to (batch_size, n_layers, num_blocks, block_size)
        x = x.view(n_samples, n_layers, -1, block_size)  # (n_samples, num_blocks, block_size)
        logger.debug(f"Split input into blocks: {x.shape}")
        return x
    
    @staticmethod
    def activation_aggregation(activations: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Aggregate activations from multiple layers for transformer processing.
        
        The aggregation concatenates features from different layers for each sample,
        then reshapes to treat the batch dimension as the sequence dimension for
        the transformer to attend over.
        
        Args:
            activations: Dictionary mapping layer names to activation tensors. 
                        Each tensor shape: (n_samples, cls_inputs_per_sample, feature_dim)
        
        Returns:
            Aggregated activation tensor of shape (n_samples, n_layers, total_feature_dim)
            where total_feature_dim = sum of (cls_inputs_per_sample * feature_dim) across all layers
        """
        # stack layers along a new dimension and concatenate features from different layers (n_samples, n_layers, cls_inputs_per_sample, feature_dim)
        acts = torch.cat([activations[k].unsqueeze(1) for k in sorted(activations.keys())], dim=1) # Now acts shape: (n_samples, n_layers, cls_inputs_per_sample, feature_dim)
        logger.debug(f"Stacked activations from layers: {acts.shape}")
        
        # Flatten cls_inputs_per_sample dimension into features
        n_samples, n_layers, cls_inputs_per_sample, feature_dim = acts.shape
        acts = acts.view(n_samples, n_layers, cls_inputs_per_sample * feature_dim)  # (n_samples, n_layers, total_feature_dim)
        
        logger.debug(f"final aggregated activations shape: {acts.shape}")
        
        return acts
