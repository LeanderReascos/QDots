from pathlib import Path
from dataclasses import dataclass, field
from pickle import HIGHEST_PROTOCOL, dump, load

import hashlib
import json
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


@dataclass
class DataSaveHelper:
    logger: logging.Logger
    fsync_env: str = "QD_CHECKPOINT_FSYNC"

    def read_positive_int_from_env(self, name, default):
        try:
            value = int(os.getenv(name, default))
        except ValueError:
            self.logger.warning("%s must be an integer; using %s", name, default)
            return default

        if value < 1:
            self.logger.warning("%s must be positive; using %s", name, default)
            return default

        return value

    def should_fsync_checkpoints(self):
        return os.getenv(self.fsync_env, "1").lower() not in {"0", "false", "no"}

    @staticmethod
    def build_sweep_hash(sweep_config):
        payload = json.dumps(sweep_config, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def append_pickle_record(path, record, fsync=True):
        path = Path(path)
        with path.open("ab") as f:
            dump(record, f, protocol=HIGHEST_PROTOCOL)
            f.flush()
            if fsync:
                os.fsync(f.fileno())

    @staticmethod
    def save_results_snapshot(results, path, fsync=True):
        path = Path(path)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("wb") as f:
            dump(results, f, protocol=HIGHEST_PROTOCOL)
            f.flush()
            if fsync:
                os.fsync(f.fileno())

        os.replace(tmp_path, path)

    def load_checkpoint_runs(self, path, sweep_hash):
        path = Path(path)
        runs_by_grid_point = {}

        if not path.exists():
            return runs_by_grid_point

        with path.open("rb") as f:
            while True:
                try:
                    record = load(f)
                except EOFError:
                    break
                except Exception as exc:
                    self.logger.warning("Stopped reading checkpoint %s after a partial/corrupt record: %s", path, exc)
                    break

                if record.get("sweep_hash") != sweep_hash:
                    self.logger.warning("Ignoring checkpoint record with a different sweep hash in %s", path)
                    continue

                if record.get("kind") != "run":
                    continue

                key = (int(record["i"]), int(record["j"]))
                runs_by_grid_point[key] = record["run"]

        return runs_by_grid_point


def format_elapsed_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
