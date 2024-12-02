from contextlib import contextmanager
from configs.config import Config 


@contextmanager
def context_manager(**overrides):
    config = Config() 
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise AttributeError(f"Invalid config attribute: '{key}'")
    try: 
        yield config 
    finally: 
        config.save() 
