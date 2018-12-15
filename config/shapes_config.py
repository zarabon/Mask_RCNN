from .default_config import DefaultConfig
from .datasets.shapes_dataset import ShapesDataset

class ShapesConfig(DefaultConfig):
    NAME = "shapes"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    NUM_CLASSES = 1 + 3  # background + 3 shapes

    # использование маленьких изображений
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    TRAIN_ROIS_PER_IMAGE = 32

    STEPS_PER_EPOCH = 100

    VALIDATION_STEPS = 5