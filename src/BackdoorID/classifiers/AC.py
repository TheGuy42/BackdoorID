from torch import nn
import torch

from .classifier_base import ClassifierBase
from src.utils.base_config import BaseConfig


class ActivationClassifierConfig(BaseConfig):
    """
    Configuration for the ActivationClassifier (Feedforward network).

    Attributes:
        input_dim (int | None): Dimension of input features. 
        hidden_dims (list[int]): List of hidden layer dimensions. Defaults to [512, 256].
        num_classes (int): Number of output classes. Defaults to 1.
        dropout (float): Dropout probability. Defaults to 0.0.
        latent_aggregation (bool): Whether to perform aggregation in the latent space. Defaults to True.
        aggregation_type (str): Type of aggregation to use ('mean', 'max', etc.). Defaults to 'mean'.
    """
    input_dim: int|None = None
    hidden_dims: list[int] = [512, 256]
    num_classes: int = 1
    dropout: float = 0.0
    latent_aggregation: bool = True
    aggregation_type: str = 'mean'  # 'mean', 'max', 'mean_poisoned', 'max_poisoned'


class ActivationClassifier(ClassifierBase):
    """
    Simple feedforward classifier for model representations.
    
    Attributes:
        config (ActivationClassifierConfig): Configuration for the classifier architecture.
    """
    
    def __init__(self, config: ActivationClassifierConfig):
        super().__init__()
        self.config = config
        
        # Extract values from config
        input_dim = config.input_dim
        hidden_dims = config.hidden_dims.copy()  # Copy to avoid modifying config
        num_classes = config.num_classes
        dropout = config.dropout
        latent_aggregation = config.latent_aggregation
        
        self.pred_head = nn.Identity()  # Placeholder, implement if needed
        layers = []
        
        if input_dim is not None:
            prev_dim = input_dim
        else:
            raise ValueError("input_dim must be specified in ActivationClassifierConfig.")
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.LayerNorm(prev_dim),
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            
            prev_dim = hidden_dim
        
        if latent_aggregation:
            self.pred_head = nn.Sequential(
                nn.Linear(prev_dim, num_classes),
            )
        else:
            # Output layer
            layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the classifier.
        
        Args:
            x:  Dictionary mapping layer names to activation tensors. Each tensor shape: (batch_size, cls_inputs_per_sample, feature_dim)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        x = self.activation_aggregation(x)
        return self.network(x)
    
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
            x = self.logit_aggregation(x, aggregation_type=self.config.aggregation_type)
            return self.pred_head(x)
        else:
            return self.logit_aggregation(x, aggregation_type=self.config.aggregation_type)
    
    @staticmethod
    def activation_aggregation(activations: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Aggregate activations from multiple layers.

        Args:
            activations: Dictionary mapping layer names to activation tensors. Each tensor shape: (batch_size, cls_inputs_per_sample, feature_dim)
        Returns:
            Aggregated activation tensor
        """
        # concatanate layers along feature dimension
        acts = torch.cat([activations[k] for k in sorted(activations.keys())], dim=-1) # Now acts shape: (batch_size, cls_inputs_per_sample, total_features)
        # flatten cls_inputs_per_sample dimension
        batch_size, cls_inputs_per_sample, feature_dim = acts.shape
        acts = acts.view(batch_size, cls_inputs_per_sample * feature_dim) # Shape: (batch_size, cls_inputs_per_sample * total_features)
        return acts
        