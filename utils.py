import time
from contextlib import contextmanager
from typing import Dict, Optional


def __merge_default(target: Dict, default: Dict):
    if default:
        for key, value in default.items():
            if isinstance(value, dict):
                target.setdefault(key, dict())
                merge_default(target[key], **value)
            else:
                target.setdefault(key, value)


def merge_default(target: Dict, *args, **kwargs):
    for arg in args:
        assert isinstance(arg, dict)
        __merge_default(target, arg)
    __merge_default(target, kwargs)


@contextmanager
def timed(prefix: Optional[str] = None):
    prefix = "Duration: " if prefix is None else prefix
    start = time.time()
    yield
    duration = time.time() - start
    print(f"{prefix}{duration:.2f}s")
