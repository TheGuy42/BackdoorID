import pathlib
from clearml import Task
from typing import Any
import numpy as np
import PIL.Image
import torch
import json

from src.utils.better_logging import get_logger

# import matplotlib
# matplotlib.use('Agg')  
from matplotlib.pyplot import Figure

logger = get_logger(__name__)

class ClearmlLogger:
    _active_instance: "ClearmlLogger | None" = None

    def __init__(
        self,
        # *names: str,
        run_name: str,
        project: str,
        root_dir: str = "logs",
        disabled: bool = False,
        **task_kwargs,
    ):
        # if len(names) == 0 and not disabled:
        #     raise ValueError("At least one name component must be provided.")

        self.run_name = run_name #str.join(" - ", names)
        self.project = project
        # self.root_dir = root_dir
        self.disabled = disabled

        # self.log_dir: str | None = None
        self.cm_task: Task | None = None

        if not disabled:
            Task.set_random_seed(None)  # NOTE: are you kidding me
            # self.log_dir = self._create_directory(root_dir, *names)
            self.cm_task = Task.init(
                project_name=project, 
                task_name=self.run_name, 
                **task_kwargs
            )
            assert self._active_instance is None, "Only one active MetricLogger instance is allowed."
            ClearmlLogger._active_instance = self

    @classmethod
    def get_active_instance(cls) -> "ClearmlLogger":
        """
        Returns the active MetricLogger instance if one is instantiated.
        
        Returns:
            MetricLogger | None: The active MetricLogger instance, or None if no instance exists.
        """
        Dummy = ClearmlLogger("dummy_run", "dummy_project", disabled=True)
        return cls._active_instance  if cls._active_instance is not None else Dummy

    def get_hparams(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "project": self.project,
            "root_dir": self.root_dir,
            # "log_dir": self.log_dir,
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
        if self.cm_task is not None:
            self.cm_task.flush(wait_for_uploads=True)
            self.cm_task.close()
            self.cm_task = None  # type: ignore
        
        # Clear the active instance if this is the active one
        if ClearmlLogger._active_instance is self:
            ClearmlLogger._active_instance = None

    def report_hparams(self, category: str = "", *args: dict[str, Any], **kwargs):
        """
        Logs hyperparameters under the key: `category/<param_name>`.
        If a parameter is a dictionary, it is flattened.

        Args:
            category (str): The category under which to log the hyperparameters.
            *args (dict): Positional dictionaries of hyperparameters to log.
            **kwargs: Keyword arguments of hyperparameters to log.
        """
        if self.cm_task is None or self.disabled:
            logger.debug("MetricLogger is disabled. Skipping.")
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

        self.cm_task.update_parameters(hparams)

    def upload_code(self, file_path: str):
        if self.cm_task is None or self.disabled:
            logger.debug("MetricLogger is disabled. Skipping.")
            return

        if not pathlib.Path(file_path).is_file():
            logger.warning(f"File '{file_path}' does not exist. Cannot log code.")
            return

        file_name = pathlib.Path(file_path).name
        file_content = pathlib.Path(file_path).read_text()
        upload_result = self.cm_task.upload_artifact(
            name=file_name,
            artifact_object=file_path,
            preview=file_content,
            # metadata={"type": "code", "full_path": file_path},
            wait_on_upload=True,
        )

        if not upload_result:
            logger.warning(f"Failed to log code file '{file_path}'.")

    def upload_json(self, json_path: str):
        if self.cm_task is None or self.disabled:
            logger.debug("MetricLogger is disabled. Skipping.")
            return

        if not pathlib.Path(json_path).is_file():
            logger.warning(f"File '{json_path}' does not exist. Cannot log JSON.")
            return

        file_name = pathlib.Path(json_path).name
        file = json.load(open(json_path, "r"))
        upload_result = self.cm_task.upload_artifact(
            name=file_name,
            artifact_object=json_path,
            preview=json.dumps(file, indent=2),
            metadata={"type": "json", "full_path": json_path},
            wait_on_upload=True,
        )

    def report_checkpoint(self, checkpoint_path: str, name: str | None = None, metadata: dict[str, Any] | None = None):
        """
        Reports a PyTorch checkpoint as an artifact.

        Args:
            checkpoint_path (str): Path to the checkpoint file (.pt, .pth, or .ckpt).
            name (str | None): Optional name for the artifact. If None, uses the filename.
            metadata (dict[str, Any] | None): Optional metadata to attach to the checkpoint.
        """
        if self.cm_task is None or self.disabled:
            logger.debug("MetricLogger is disabled. Skipping.")
            return

        checkpoint_path_obj = pathlib.Path(checkpoint_path)
        if not checkpoint_path_obj.is_file():
            logger.warning(f"Checkpoint file '{checkpoint_path}' does not exist. Cannot log checkpoint.")
            return

        artifact_name = name if name is not None else checkpoint_path_obj.name
        
        # Prepare metadata
        artifact_metadata = {"type": "pytorch_checkpoint", "full_path": str(checkpoint_path_obj.absolute())}
        if metadata is not None:
            artifact_metadata.update(metadata)

        # Load checkpoint to create a preview
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            preview_info = {}
            
            if isinstance(checkpoint, dict):
                preview_info["keys"] = list(checkpoint.keys())
                # Add shape information for state_dict if present
                if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                    preview_info["num_parameters"] = len(checkpoint["state_dict"])
                elif "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
                    preview_info["num_parameters"] = len(checkpoint["model_state_dict"])
            
            preview = json.dumps(preview_info, indent=2)
        except Exception as e:
            logger.warning(f"Could not load checkpoint for preview: {e}")
            preview = f"PyTorch checkpoint at {checkpoint_path}"

        upload_result = self.cm_task.upload_artifact(
            name=artifact_name,
            artifact_object=checkpoint_path,
            preview=preview,
            metadata=artifact_metadata,
            wait_on_upload=True,
        )

        if not upload_result:
            logger.warning(f"Failed to upload checkpoint '{checkpoint_path}'.")
        else:
            logger.info(f"Successfully uploaded checkpoint '{artifact_name}'.")

    def set_tags(self, *str_tags, **tags):
        """
        Sets tags for the current experiment.
        """
        if self.cm_task is None or self.disabled:
            logger.debug("MetricLogger is disabled. Skipping.")
            return

        tags_list = []
        if len(tags) == 0 and len(str_tags) == 0:
            logger.warning("No tags provided.")
            return
        if len(tags) > 0:
            tags = {k.title(): str(v) for k, v in tags.items()}
            formatted = [f"{tag}: {text}" for tag, text in tags.items()]
            tags_list.extend(formatted)
        if len(str_tags) > 0:
            tags_list.extend(list(str_tags))
        
        self.cm_task.add_tags(tags_list)

    def report_globals(self, metrics: dict[str, int | float]):
        """
        Logs a dictionary of final, global run metrics.
        For example, used to log final evaluation metrics.

        Args:
            metrics (dict[str, int | float]): The metrics to log.
        """
        if self.cm_task is None or self.disabled:
            logger.debug("MetricLogger is disabled. Skipping.")
            return

        for k, v in metrics.items():
            self.cm_task.logger.report_single_value(k, v)

    def report_scalars(self, scalers: dict[str, int | float], step: int):
        """
        Logs multiple scalar values.

        Args:
            scalers (dict[str, int | float]): The scalar values to log.
            step (int): The step number.
        """
        if self.cm_task is None or self.disabled:
            logger.debug("MetricLogger is disabled. Skipping.")
            return

        for key, value in scalers.items():
            self.report_scalar(key, value, step)

    def report_scalar(self, tag: str, value: int | float | None, step: int):
        """
        Logs a scalar value.

        Args:
            tag (str): The name of the scalar value.
            value (int | float | None): The scalar value to log.
            step (int): The step number.
        """
        if self.cm_task is None or self.disabled:
            logger.debug("MetricLogger is disabled. Skipping.")
            return

        if "/" in tag:
            split = tag.split("/", maxsplit=1)
            title, series = split[0], split[1]
        else:
            title = series = tag

        title = title.title()

        if value is None:
            value = float("nan")

        cm_logger = self.cm_task.get_logger()
        cm_logger.report_scalar(title=title, series=series, value=float(value), iteration=step)

    def report_matplotlib_figure(self, tag: str, figure: Figure, step: int, report_image: bool = True):
        """
        Logs a matplotlib figure.

        Args:
            tag (str): The name of the figure.
            figure (Figure): The matplotlib figure to log.
            step (int): The step number.
        """
        if self.cm_task is None or self.disabled:
            logger.debug("MetricLogger is disabled. Skipping.")
            return

        if "/" in tag:
            split = tag.split("/", maxsplit=1)
            title, series = split[0], split[1]
        else:
            title = figure.get_suptitle().get_text() if figure._suptitle is not None else "Figure"
            series = tag

        title = title.title()
        series = series.title()

        cm_logger = self.cm_task.get_logger()
        cm_logger.report_matplotlib_figure(title=title, series=series, figure=figure, iteration=step, report_image=report_image, report_interactive=True)

    def report_image(self, tag: str, path:str, step: int=0):
        """
        Logs an image.

        Args:
            tag (str): The name of the image.
            path (str): The image file path to log, RGB format.
            step (int): The step number.
        """
        if self.cm_task is None or self.disabled:
            logger.debug("MetricLogger is disabled. Skipping.")
            return

        if "/" in tag:
            split = tag.split("/", maxsplit=1)
            title, series = split[0], split[1]
        else:
            title = series = tag

        title = title.title()
        series = series.title()

        cm_logger = self.cm_task.get_logger()
        cm_logger.report_image(title=title, series=series, local_path=path, iteration=step, max_image_history=-1)

    def flush(self, wait:bool=True):
        """
        Flushes the current task logs.
        """
        if self.cm_task is None or self.disabled:
            logger.debug("MetricLogger is disabled. Skipping.")
            return

        self.cm_task.flush(wait_for_uploads=wait)
