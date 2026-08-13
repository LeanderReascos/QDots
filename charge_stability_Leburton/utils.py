from pathlib import Path
from dataclasses import dataclass, field

import logging
import os
import faulthandler


@dataclass
class LoggerSetup:
    name: str
    results_filename: str
    log_filename: str
    data_dir: Path | str = Path("data")
    env_log_level: str = "QD_LOG_LEVEL"
    file_mode: str = "w"
    enable_faulthandler: bool = True
    results_path: Path = field(init=False)
    log_path: Path = field(init=False)
    logger: logging.Logger = field(init=False)

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.data_dir / self.results_filename
        self.log_path = self.data_dir / self.log_filename
        self.logger = self._create_logger()

        if self.enable_faulthandler:
            faulthandler.enable()

    def _create_logger(self):
        log_level_name = os.getenv(self.env_log_level, "INFO").upper()
        log_level = getattr(logging, log_level_name, logging.INFO)

        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        logger.propagate = False

        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter("%(message)s"))

        file_handler = logging.FileHandler(self.log_path, mode=self.file_mode)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
        )

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        return logger


def format_elapsed_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
