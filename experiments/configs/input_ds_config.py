from src.utils.base_config import BaseConfig
from src.BackdoorID.base import strDataset
from datasets import load_dataset
import random
from typing import Callable
from pydantic import Field


class InputDSConfig(BaseConfig):
    """
    Configuration for an input dataset used for activation extraction.

    Attributes:
        name (str): The name/identifier of the input dataset.
        n_samples (int): Number of samples to load from the dataset. 
        seed (int | None): Random seed for sampling. If None, sampling is non-deterministic.
        kwargs (dict): Additional keyword arguments to pass to the build function.
        build_ds_func (Callable[..., strDataset]): Function to call to actually load the data.
    """
    name: str
    n_samples: int = 1000
    seed: int|None = None
    kwargs: dict = {}
    build_ds_func: Callable[..., strDataset] = Field(exclude=True)
    
    def build_dataset(self) -> strDataset:
        """
        Build and return the dataset using the configured build function.
                
        Returns:
            strDataset containing the loaded dataset
        """
        params = {'num_samples': self.n_samples, 'seed': self.seed}
        params.update(self.kwargs)
        return self.build_ds_func(**params)



def load_alpaca_dataset(num_samples: int = 1000, seed: int = 42) -> strDataset:
    """
    Load the Alpaca dataset from HuggingFace and sample a subset.
    
    Args:
        num_samples: Number of samples to select from the dataset
        seed: Random seed for reproducible sampling
    
    Returns:
        strDataset containing the sampled text prompts
    """
    ALPACA_PROMPT = "### Instruction:\n{instruction}\n\n### Response:\n"
    ALPACA_PROMPT_W_INPUT = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    # Load the alpaca dataset from HuggingFace
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Get total size and determine sampling
    total_samples = len(dataset)
    num_samples = min(num_samples, total_samples)
    
    # Sample indices
    sampled_indices = random.sample(range(total_samples), num_samples)
    
    # Extract the text from the sampled rows
    # The alpaca dataset has 'instruction', 'input', and 'output' fields
    # We'll combine instruction and input (if present) to create prompts
    samples = []
    for idx in sampled_indices:
        row = dataset[idx]
        instruction = row['instruction']
        input_text = row.get('input', None)
        
        # Combine instruction and input
        if input_text is not None and input_text.strip() != "":
            prompt = ALPACA_PROMPT_W_INPUT.format(instruction=instruction, input=input_text)
        else:
            prompt = ALPACA_PROMPT.format(instruction=instruction)
        
        samples.append(prompt)
    
    return strDataset(samples)


def load_self_instruct_dataset(num_samples: int = 1000, seed: int = 42) -> strDataset:
    """
    Load the Self-Instruct dataset from HuggingFace and sample a subset.
    
    Args:
        num_samples: Number of samples to select from the dataset
        seed: Random seed for reproducible sampling
    
    Returns:
        strDataset containing the sampled text prompts
    """
    # Load the self-instruct dataset from HuggingFace
    dataset = load_dataset("srinivasbilla/self-instruct-base", split="train")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Get total size and determine sampling
    total_samples = len(dataset)
    num_samples = min(num_samples, total_samples)
    
    # Sample indices
    sampled_indices = random.sample(range(total_samples), num_samples)
    
    # Extract the text from the sampled rows
    samples = []
    for idx in sampled_indices:
        row = dataset[idx]
        instruction = row['prompt']
        samples.append(instruction)
    
    return strDataset(samples)


def load_gsm8k_dataset(num_samples: int = 1000, seed: int = 42) -> strDataset:
    """
    Load the GSM8K dataset from HuggingFace and sample a subset.
    
    Args:
        num_samples: Number of samples to select from the dataset
        seed: Random seed for reproducible sampling
    
    Returns:
        strDataset containing the sampled math questions
    """
    # Load the GSM8K dataset from HuggingFace
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Get total size and determine sampling
    total_samples = len(dataset)
    num_samples = min(num_samples, total_samples)
    
    # Sample indices
    sampled_indices = random.sample(range(total_samples), num_samples)
    
    # Extract the questions from the sampled rows
    samples = []
    for idx in sampled_indices:
        row = dataset[idx]
        question = row['question']
        samples.append(question)
    
    return strDataset(samples)


def load_imdb_dataset(num_samples: int = 1000, seed: int = 42) -> strDataset:
    """
    Load the IMDB dataset from HuggingFace and sample a subset.
    
    Args:
        num_samples: Number of samples to select from the dataset
        seed: Random seed for reproducible sampling
    
    Returns:
        strDataset containing the sampled movie reviews
    """
    # Load the IMDB dataset from HuggingFace
    dataset = load_dataset("stanfordnlp/imdb", split="train")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Get total size and determine sampling
    total_samples = len(dataset)
    num_samples = min(num_samples, total_samples)
    
    # Sample indices
    sampled_indices = random.sample(range(total_samples), num_samples)
    
    # Extract the text from the sampled rows
    samples = []
    for idx in sampled_indices:
        row = dataset[idx]
        text = row['text']
        samples.append(text)
    
    return strDataset(samples)


def load_ag_news_dataset(num_samples: int = 1000, seed: int = 42, split: str = "train") -> strDataset:
    """
    Load the AG News dataset from HuggingFace and sample a subset.

    Args:
        num_samples: Number of samples to select from the dataset
        seed: Random seed for reproducible sampling
        split: Dataset split to use ('train' has 120k rows, 'test' has 7.6k rows)

    Returns:
        strDataset containing the sampled news article texts
    """
    dataset = load_dataset("fancyzhx/ag_news", split=split)

    random.seed(seed)

    total_samples = len(dataset)
    num_samples = min(num_samples, total_samples)

    sampled_indices = random.sample(range(total_samples), num_samples)

    samples = [dataset[idx]["text"] for idx in sampled_indices]

    return strDataset(samples)


INPUT_DS_CONFIGS: dict[str, InputDSConfig] = {
    'alpaca': InputDSConfig(
        name='alpaca',
        n_samples=300,
        seed=0,
        kwargs={},
        build_ds_func=load_alpaca_dataset
    ),
    'self_instruct': InputDSConfig(
        name='self_instruct',
        n_samples=60,
        seed=0,
        kwargs={},
        build_ds_func=load_self_instruct_dataset
    ),
    'gsm8k': InputDSConfig(
        name='gsm8k',
        n_samples=60,
        seed=0,
        kwargs={},
        build_ds_func=load_gsm8k_dataset
    ),
    'imdb': InputDSConfig(
        name='imdb',
        n_samples=60,
        seed=0,
        kwargs={},
        build_ds_func=load_imdb_dataset
    ),
    'ag_news': InputDSConfig(
        name='ag_news',
        n_samples=60,
        seed=0,
        kwargs={},
        build_ds_func=load_ag_news_dataset
    ),
}
# Configure input datasets


def get_dataset(name: str, **kwargs) -> InputDSConfig:
    """
    Get the specified dataset configuration by name.
    
    Args:
        name: Name of the dataset configuration to load
        **kwargs: Additional parameters to override the default config params
    
    Returns:
        InputDSConfig for the specified dataset
    
    Raises:
        KeyError: If the dataset name is not found in INPUT_DS_CONFIGS
    """
    if name not in INPUT_DS_CONFIGS:
        raise KeyError(f"Dataset '{name}' not found in INPUT_DS_CONFIGS. Available datasets: {list(INPUT_DS_CONFIGS.keys())}")
    
    config = INPUT_DS_CONFIGS[name]
    
    # If kwargs provided, create a modified copy of the config
    if kwargs:
        config_dict = config.model_dump()
        config_dict.update(kwargs)
        config = InputDSConfig(**config_dict, build_ds_func=config.build_ds_func)
    
    return config 









