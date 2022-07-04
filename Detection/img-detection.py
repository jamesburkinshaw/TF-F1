import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import cv2 
import numpy as np
from matplotlib import pyplot as plt

#disable logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#paths
PATH_TO_SAVED_MODEL = 'C:\\Users\\paperspace\\Documents\\TensorFlow\\workspace\\TF-F1\\exported-models\\TF-F1-Model\\saved_model'
PATH_TO_LABELS = 'C:\\Users\\paperspace\\Documents\\TensorFlow\\workspace\\TF-F1\\annotations\\label_map.pbtxt'

#setup model
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

#for each image
IMAGE_PATH = 'C:\\Users\\paperspace\\Documents\\TensorFlow\\workspace\\TF-F1\\images\\test\miami6680.png'

img = cv2.imread(IMAGE_PATH)
image_np = np.array(img)

#setup input as tensor
input_tensor = tf.convert_to_tensor(image_np)
input_tensor = input_tensor[tf.newaxis, ...]

#Detect!
detections = detect_fn(input_tensor)

num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
detections['num_detections'] = num_detections

detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

image_np_with_detections = image_np.copy()

#visualise
viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.3,
            agnostic_mode=False)

plt.figure
plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.savefig("detection.png", bbox_inches='tight', pad_inches = 0)
