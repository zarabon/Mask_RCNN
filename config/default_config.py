from mrcnn.config import Config

class DefaultConfig(Config):
    def __init__(self, batch_size):
        Config.__init__(self)
        self.BATCH_SIZE = batch_size

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1