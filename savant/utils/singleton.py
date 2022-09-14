"""Thread-safe singleton metaclass."""
from threading import Lock


class SingletonMeta(type):
    """Thread-safe singleton metaclass."""

    def __new__(cls, name, bases, attrs):
        # Assume the target class is created
        # (i.e. this method to be called) in the main thread.
        _cls = super().__new__(cls, name, bases, attrs)
        _cls.__shared_instance_lock__ = Lock()
        return _cls

    def __call__(cls, *args, **kwargs):
        with cls.__shared_instance_lock__:
            try:
                return cls.__shared_instance__
            except AttributeError:
                cls.__shared_instance__ = super(SingletonMeta, cls).__call__(
                    *args, **kwargs
                )
                return cls.__shared_instance__
