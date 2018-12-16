from .default_config import DefaultConfig
from .class_names.coco_class_names import class_names

class_names

class CocoConfig(DefaultConfig):
    def __init__(self, batch_size):
        DefaultConfig.__init__(self, batch_size)
        self.BATCH_SIZE = batch_size

    NAME = "coco"
    NUM_CLASSES = 81
