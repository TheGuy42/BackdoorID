from torch import nn
import torch

class ClassifierBase(nn.Module):
    """
    Base class for classifiers that operate on model activations.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the classifier.
        
        Args:
            x:  Dictionary mapping layer names to activation tensors. Each tensor shape: (batch_size, cls_inputs_per_sample, feature_dim)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        raise NotImplementedError("ActivationClassifier forward method is not implemented yet.")
    
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
        raise NotImplementedError("ActivationClassifier prediction_head method is not implemented yet.")
    
    @staticmethod
    def logit_aggregation(logits: torch.Tensor, aggregation_type:str = 'mean') -> torch.Tensor:
        """
        Aggregate prediction probabilities from multiple inputs.

        Args:
            probs: Tensor of shape (num_inputs, num_classes)
        Returns:
            Aggregated probabilities tensor of shape (1, num_classes)
        """
        if aggregation_type == 'max':
            return logits.amax(dim=0, keepdim=True)  # Shape: (1, num_classes)
        elif aggregation_type == 'mean':
            return logits.mean(dim=0, keepdim=True)  # Shape: (1, num_classes)
        elif aggregation_type == 'mean_poisoned':
            assert logits.dim() == 2, "Logits tensor must be 2-dimensional"
            probs = torch.softmax(logits, dim=-1) # Shape: (num_inputs, num_classes)
            preds = torch.argmax(probs, dim=-1)  # Shape: (num_inputs,)
            poisoned_logits = logits[preds == 1]  # Select only poisoned predictions
            if poisoned_logits.size(0) == 0:
                # If no poisoned predictions, return mean of all logits
                return logits.mean(dim=0, keepdim=True)
            return poisoned_logits.mean(dim=0, keepdim=True)  # Shape: (1, num_classes)
        elif aggregation_type == 'max_poisoned':
            assert logits.dim() == 2, "Logits tensor must be 2-dimensional"
            probs = torch.softmax(logits, dim=-1) # Shape: (num_inputs, num_classes)
            preds = torch.argmax(probs, dim=-1)  # Shape: (num_inputs,)
            poisoned_logits = logits[preds == 1]  # Select only poisoned predictions
            if poisoned_logits.size(0) == 0:
                # If no poisoned predictions, return max of all logits
                return logits.amax(dim=0, keepdim=True)
            return poisoned_logits.amax(dim=0, keepdim=True)  # Shape: (1, num_classes)
        else:
            raise ValueError(f"Unsupported logit aggregation method: {aggregation_type}")