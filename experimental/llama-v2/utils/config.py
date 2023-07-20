class Config:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance.enable_nvtx = False
            cls._instance.use_triton = False

            # perplexity
            cls._instance.prefix = 10
            cls._instance.seq_len = 128
            cls._instance.num_samples = 20
            cls._instance.batch_size = 20

        return cls._instance

    def set_nvtx(self, value: bool):
        self.enable_nvtx = value

    def get_nvtx(self) -> bool:
        return self.enable_nvtx

    def set_use_triton(self, value: bool):
        self.use_triton = value

    def get_use_triton(self) -> bool:
        return self.use_triton
