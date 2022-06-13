from flask import Flask,render_template,Response
from flask_sqlalchemy import SQLAlchemy
import os
import cv2 
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from ocr_detection import ocr_it,save_results
WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
IMAGE_PATH = WORKSPACE_PATH+'/images'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'
#flask setup
app = Flask(__name__)
app.secret_key = 'the random string'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/npd'
db = SQLAlchemy(app)

class Numbers(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    number = db.Column(db.String(50), nullable=False)
    date = db.Column(db.String(20), nullable=True)
    time = db.Column(db.String(20), nullable=True)

# Load pipeline config and build a detection model

configs = config_util.get_configs_from_pipeline_file(MODEL_PATH+'/'+'my_ssd_mobnet'+'/pipeline.config')
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-14')).expect_partial()
region_threshold = 0.6
detection_threshold = 0.87
@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')
# cap=cv2.VideoCapture(0)
