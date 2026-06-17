import pathlib
import wandb
from typing import Any
import numpy as np
import PIL.Image
import torch
import json

from src.utils.better_logging import get_logger

from matplotlib.pyplot import Figure

logger = get_logger(__name__)


class WandbLogger:
    _active_instance: "WandbLogger | None" = None

    def __init__(
        self,
        run_name: str,
        project: str,
        root_dir: str = "logs",
        disabled: bool = False,
        **wandb_kwargs,
    ):
        self.run_name = run_name
        self.project = project
        self.root_dir = root_dir
        self.disabled = disabled

        self.wandb_run: wandb.sdk.wandb_run.Run | None = None

        if not disabled:
            self.wandb_run = wandb.init(
                project=project,
                name=run_name,
                dir=root_dir,
                **wandb_kwargs
            )
            assert self._active_instance is None, "Only one active WandbLogger instance is allowed."
            WandbLogger._active_instance = self

    @classmethod
    def get_active_instance(cls) -> "WandbLogger":
        """
        Returns the active WandbLogger instance if one is instantiated.
        
        Returns:
            WandbLogger: The active WandbLogger instance, or a dummy instance if no instance exists.
        """
        Dummy = WandbLogger("dummy_run", "dummy_project", disabled=True)
        return cls._active_instance if cls._active_instance is not None else Dummy

    def get_hparams(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "project": self.project,
            "root_dir": self.root_dir,
            "disabled": self.disabled,
        }

    def _create_directory(self, *subdir_parts: str) -> str:
        log_path = pathlib.Path(*subdir_parts)
        log_path.mkdir(parents=True, exist_ok=True)
        path = log_path.as_posix()
        logger.info(f"Created log directory at: {path}")
        return path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if self.wandb_run is not None:
            self.wandb_run.finish()
            self.wandb_run = None

        # Clear the active instance if this is the active one
        if WandbLogger._active_instance is self:
            WandbLogger._active_instance = None

    def report_hparams(self, category: str = "", *args: dict[str, Any], **kwargs):
        """
        Logs hyperparameters under the key: `category/<param_name>`.
        If a parameter is a dictionary, it is flattened.

        Args:
            category (str): The category under which to log the hyperparameters.
            *args (dict): Positional dictionaries of hyperparameters to log.
            **kwargs: Keyword arguments of hyperparameters to log.
        """
        if self.wandb_run is None or self.disabled:
            logger.debug("WandbLogger is disabled. Skipping.")
            return

        def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        hparams = flatten_dict(kwargs, category)
        for d in args:
            hparams.update(flatten_dict(d, category))

        wandb.config.update(hparams)

    def upload_code(self, file_path: str):
        """
        Uploads a code file as an artifact to wandb.

        Args:
            file_path (str): Path to the code file to upload.
        """
        if self.wandb_run is None or self.disabled:
            logger.debug("WandbLogger is disabled. Skipping.")
            return

        if not pathlib.Path(file_path).is_file():
            logger.warning(f"File '{file_path}' does not exist. Cannot log code.")
            return

        file_name = pathlib.Path(file_path).name
        artifact = wandb.Artifact(name=f"code-{file_name}", type="code")
        artifact.add_file(file_path)
        self.wandb_run.log_artifact(artifact)
        logger.info(f"Successfully uploaded code file '{file_name}'.")

    def upload_json(self, json_path: str):
        """
        Uploads a JSON file as an artifact to wandb.

        Args:
            json_path (str): Path to the JSON file to upload.
        """
        if self.wandb_run is None or self.disabled:
            logger.debug("WandbLogger is disabled. Skipping.")
            return

        if not pathlib.Path(json_path).is_file():
            logger.warning(f"File '{json_path}' does not exist. Cannot log JSON.")
            return

        file_name = pathlib.Path(json_path).name
        artifact = wandb.Artifact(name=f"json-{file_name}", type="json")
        artifact.add_file(json_path)
        
        # Also log the JSON content as a table for better visualization
        with open(json_path, "r") as f:
            json_data = json.load(f)
        
        artifact.metadata = {"content_preview": json_data}
        self.wandb_run.log_artifact(artifact)
        logger.info(f"Successfully uploaded JSON file '{file_name}'.")

    def report_checkpoint(self, checkpoint_path: str, name: str | None = None, metadata: dict[str, Any] | None = None):
        """
        Reports a PyTorch checkpoint as an artifact.

        Args:
            checkpoint_path (str): Path to the checkpoint file (.pt, .pth, or .ckpt).
            name (str | None): Optional name for the artifact. If None, uses the filename.
            metadata (dict[str, Any] | None): Optional metadata to attach to the checkpoint.
        """
        if self.wandb_run is None or self.disabled:
            logger.debug("WandbLogger is disabled. Skipping.")
            return

        checkpoint_path_obj = pathlib.Path(checkpoint_path)
        if not checkpoint_path_obj.is_file():
            logger.warning(f"Checkpoint file '{checkpoint_path}' does not exist. Cannot log checkpoint.")
            return

        artifact_name = name if name is not None else checkpoint_path_obj.stem
        
        # Create artifact
        artifact = wandb.Artifact(name=artifact_name, type="model")
        artifact.add_file(checkpoint_path)
        
        # Prepare metadata
        artifact_metadata = {"full_path": str(checkpoint_path_obj.absolute())}
        if metadata is not None:
            artifact_metadata.update(metadata)

        # Load checkpoint to add metadata
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            if isinstance(checkpoint, dict):
                artifact_metadata["keys"] = list(checkpoint.keys())
                # Add shape information for state_dict if present
                if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                    artifact_metadata["num_parameters"] = len(checkpoint["state_dict"])
                elif "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
                    artifact_metadata["num_parameters"] = len(checkpoint["model_state_dict"])
        except Exception as e:
            logger.warning(f"Could not load checkpoint for metadata: {e}")

        artifact.metadata = artifact_metadata
        self.wandb_run.log_artifact(artifact)
        logger.info(f"Successfully uploaded checkpoint '{artifact_name}'.")

    def set_tags(self, *str_tags, **tags):
        """
        Sets tags for the current experiment.
        """
        if self.wandb_run is None or self.disabled:
            logger.debug("WandbLogger is disabled. Skipping.")
            return

        tags_list = []
        if len(tags) == 0 and len(str_tags) == 0:
            logger.warning("No tags provided.")
            return
        
        if len(tags) > 0:
            tags_dict = {k.title(): str(v) for k, v in tags.items()}
            formatted = [f"{tag}: {text}" for tag, text in tags_dict.items()]
            tags_list.extend(formatted)
        
        if len(str_tags) > 0:
            tags_list.extend(list(str_tags))
        
        self.wandb_run.tags = self.wandb_run.tags + tuple(tags_list)

    def report_globals(self, metrics: dict[str, int | float]):
        """
        Logs a dictionary of final, global run metrics.
        For example, used to log final evaluation metrics.

        Args:
            metrics (dict[str, int | float]): The metrics to log.
        """
        if self.wandb_run is None or self.disabled:
            logger.debug("WandbLogger is disabled. Skipping.")
            return

        wandb.summary.update(metrics)

    def report_scalars(self, scalars: dict[str, int | float], step: int):
        """
        Logs multiple scalar values.

        Args:
            scalars (dict[str, int | float]): The scalar values to log.
            step (int): The step number.
        """
        if self.wandb_run is None or self.disabled:
            logger.debug("WandbLogger is disabled. Skipping.")
            return

        log_dict = {k: v for k, v in scalars.items()}
        # commit=False so Lightning's WandbLogger owns the step counter
        wandb.log(log_dict, commit=False)

    def report_scalar(self, tag: str, value: int | float | None, step: int):
        """
        Logs a scalar value.

        Args:
            tag (str): The name of the scalar value.
            value (int | float | None): The scalar value to log.
            step (int): The step number.
        """
        if self.wandb_run is None or self.disabled:
            logger.debug("WandbLogger is disabled. Skipping.")
            return

        if value is None:
            value = float("nan")

        # commit=False so Lightning's WandbLogger owns the step counter
        wandb.log({tag: float(value)}, commit=False)

    def report_matplotlib_figure(self, tag: str, figure: Figure, step: int, report_image: bool = True):
        """
        Logs a matplotlib figure.

        Args:
            tag (str): The name of the figure.
            figure (Figure): The matplotlib figure to log.
            step (int): The step number.
            report_image (bool): Whether to report the figure as an image.
        """
        if self.wandb_run is None or self.disabled:
            logger.debug("WandbLogger is disabled. Skipping.")
            return

        # commit=False so Lightning's WandbLogger owns the step counter
        if report_image:
            wandb.log({tag: wandb.Image(figure)}, commit=False)
        else:
            wandb.log({tag: figure}, commit=False)

    def report_image(self, tag: str, path: str, step: int = 0):
        """
        Logs an image.

        Args:
            tag (str): The name of the image.
            path (str): The image file path to log, RGB format.
            step (int): The step number.
        """
        if self.wandb_run is None or self.disabled:
            logger.debug("WandbLogger is disabled. Skipping.")
            return

        if not pathlib.Path(path).is_file():
            logger.warning(f"Image file '{path}' does not exist. Cannot log image.")
            return

        image = PIL.Image.open(path)
        # commit=False so Lightning's WandbLogger owns the step counter
        wandb.log({tag: wandb.Image(image)}, commit=False)

    def flush(self, wait: bool = True):
        """
        Flushes the current wandb run logs.
        
        Args:
            wait (bool): Whether to wait for uploads to complete.
        """
        if self.wandb_run is None or self.disabled:
            logger.debug("WandbLogger is disabled. Skipping.")
            return

        # Wandb doesn't have an explicit flush method, but we can ensure
        # all pending logs are sent by calling log with commit=True
        if wait:
            # Force a sync by logging an empty dict with commit=True
            wandb.log({}, commit=True)
