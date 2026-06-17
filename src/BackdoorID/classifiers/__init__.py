from src.utils.base_config import BaseConfig
from pydantic import field_validator, Field
from typing import Any

from .classifier_base import ClassifierBase
from .AC import ActivationClassifier, ActivationClassifierConfig
from .AcT import ActivationTransformerConfig, ActivationTransformer


# Mapping of classifier names to their config classes
CLASSIFIER_CONFIGS = {
    "ActivationClassifier": ActivationClassifierConfig,
    "ActivationTransformer": ActivationTransformerConfig,
}

# Mapping of classifier names to their class implementations
CLASSIFIER_CLASSES = {
    "ActivationClassifier": ActivationClassifier,
    "ActivationTransformer": ActivationTransformer,
}


__all__ = [
    "ActivationClassifier",
    "ActivationClassifierConfig",
    "TransformerClassifier",
    "TransformerClassifierConfig",
    "ClassifierBase",
    ]
