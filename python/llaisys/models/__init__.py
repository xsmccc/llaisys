from .qwen2 import Qwen2
try:
    from .llama3 import Llama3
except (ImportError, AttributeError):
    pass  # LLaMA3 not available
