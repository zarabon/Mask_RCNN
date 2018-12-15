import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from mrcnn.config import Config
from config.coco_config import class_names, CocoConfig

ROOT_DIR = os.path.abspath("./")

sys.path.append(ROOT_DIR)
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Логи и трен ированная модель
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Путь до файла с весами
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "weights", "mask_rcnn_coco.h5")

# исходные изображения
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

config = CocoConfig()


# Создаем модель
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Загружаем веса
model.load_weights(COCO_MODEL_PATH, by_name=True)

file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])