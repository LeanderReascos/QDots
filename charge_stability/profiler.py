from time import perf_counter
import psutil
import os
import json


class Profiler:
    def __init__(self):
        self.data = {}
        self.process = psutil.Process(os.getpid())

    class Stage:
        def __init__(self, profiler, name):
            self.profiler = profiler
            self.name = name

        def __enter__(self):
            self.start_time = perf_counter()
            self.start_memory = self.profiler.process.memory_info().rss

        def __exit__(self, exc_type, exc_val, exc_tb):
            end_time = perf_counter()
            end_memory = self.profiler.process.memory_info().rss

            self.profiler.data[self.name] = {
                "time_s": end_time - self.start_time,
                "memory_before_MB": self.start_memory / 1024**2,
                "memory_after_MB": end_memory / 1024**2,
                "memory_change_MB": (end_memory - self.start_memory) / 1024**2,
            }

    def stage(self, name):
        return Profiler.Stage(self, name)

    def save(self, filename):
        with open(filename, "w") as f:
            json.dump(self.data, f, indent=4)