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
file_names = next(os.walk(IMAGE_DIR))[2]

config = CocoConfig(1)

# Создаем модель
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Загружаем веса
model.load_weights(COCO_MODEL_PATH, by_name=True)


images = [skimage.io.imread(os.path.join(IMAGE_DIR, fn)) for fn in file_names ]

# Запуск распознавания
results = []
for i in images:
    results += [(i, model.detect([i], verbose=1))]

# Отображение результата
for (i,res) in results:
    r = res[0]
    visualize.display_instances(i, r['rois'], r['masks'], r['class_ids'], 
                                class_names, r['scores'])