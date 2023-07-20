class NoOpContextManager:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
