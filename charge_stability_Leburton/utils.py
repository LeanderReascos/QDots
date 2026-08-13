from pathlib import Path
from dataclasses import dataclass, field
from pickle import HIGHEST_PROTOCOL, dump, load

import faulthandler
import hashlib
import json
import logging
import os


@dataclass
class LoggerSetup:
    name: str
    results_filename: str
    log_filename: str
    data_dir: Path | str = Path("data")
    env_log_level: str = "LOG_LEVEL"
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
class DataHelper:
    data_dir: Path | str = Path("data")
    results_filename: str = "results.pkl"
    checkpoint_glob: str | None = None
    logger: logging.Logger | None = None
    fsync_env: str = "CHECKPOINT_FSYNC"
    results_path: Path = field(init=False)

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.data_dir / self.results_filename

    def _warning(self, message, *args):
        if self.logger is not None:
            self.logger.warning(message, *args)

    def read_positive_int_from_env(self, name, default):
        try:
            value = int(os.getenv(name, default))
        except ValueError:
            self._warning("%s must be an integer; using %s", name, default)
            return default

        if value < 1:
            self._warning("%s must be positive; using %s", name, default)
            return default

        return value

    def should_fsync_checkpoints(self):
        return os.getenv(self.fsync_env, "1").lower() not in {"0", "false", "no"}

    @staticmethod
    def build_hash(config):
        payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _resolve_path(path):
        return Path(path)

    def append_checkpoint_record(self, path, record, fsync=True):
        path = self._resolve_path(path)
        with path.open("ab") as f:
            dump(record, f, protocol=HIGHEST_PROTOCOL)
            f.flush()
            if fsync:
                os.fsync(f.fileno())

    def save(self, data, path=None, fsync=True):
        path = self.results_path if path is None else self._resolve_path(path)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("wb") as f:
            dump(data, f, protocol=HIGHEST_PROTOCOL)
            f.flush()
            if fsync:
                os.fsync(f.fileno())

        os.replace(tmp_path, path)

    def load(self, path=None, default=None):
        path = self.results_path if path is None else self._resolve_path(path)
        if not path.exists():
            return default

        with path.open("rb") as f:
            return load(f)

    def load_pickle_stream(self, path):
        path = self._resolve_path(path)
        records = []

        if not path.exists():
            return records

        with path.open("rb") as f:
            while True:
                try:
                    records.append(load(f))
                except EOFError:
                    break
                except Exception as exc:
                    self._warning("Stopped reading %s after a partial/corrupt record: %s", path, exc)
                    break

        return records

    def checkpoint_paths(self, checkpoint_glob=None):
        checkpoint_glob = self.checkpoint_glob if checkpoint_glob is None else checkpoint_glob
        if checkpoint_glob is None:
            return []

        return sorted(self.data_dir.glob(checkpoint_glob), key=lambda path: path.stat().st_mtime)

    def newest_checkpoint_path(self, checkpoint_glob=None):
        paths = self.checkpoint_paths(checkpoint_glob=checkpoint_glob)
        return paths[-1] if paths else None

    def load_checkpoint_records(self, path):
        return self.load_pickle_stream(path)

    def load_checkpoints(self, checkpoint_paths=None, checkpoint_glob=None):
        paths = self.checkpoint_paths(checkpoint_glob=checkpoint_glob) if checkpoint_paths is None else checkpoint_paths
        return {Path(path): self.load_checkpoint_records(path) for path in paths}

    def load_data_with_checkpoints(self, results_path=None, default=None, checkpoint_paths=None, checkpoint_glob=None):
        checkpoints = self.load_checkpoints(checkpoint_paths=checkpoint_paths, checkpoint_glob=checkpoint_glob)
        checkpoint_records = [record for records in checkpoints.values() for record in records]
        return {
            "data": self.load(path=results_path, default=default),
            "checkpoints": checkpoints,
            "checkpoint_records": checkpoint_records,
        }


def cleanup_matching_files(patterns, directories=None, logger=None):
    if isinstance(patterns, str):
        patterns = (patterns,)
    if directories is None:
        directories = (Path.cwd(),)

    removed = []
    seen_paths = set()
    for directory in directories:
        directory = Path(directory)
        if not directory.exists():
            continue

        for pattern in patterns:
            for path in directory.glob(pattern):
                if not path.is_file():
                    continue

                resolved_path = path.resolve()
                if resolved_path in seen_paths:
                    continue
                seen_paths.add(resolved_path)

                try:
                    path.unlink()
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    if logger is not None:
                        logger.warning("Could not remove temporary file %s: %s", path, exc)
                else:
                    removed.append(path)

    return removed


def format_elapsed_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
