import os
from os.path import join, isfile
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import cv2 
import numpy as np
from matplotlib import pyplot as plt

print('Starting frame by frame object detection')
#disable logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#paths
BASE_PATH = 'C:\\Users\\paperspace\\Documents\\TensorFlow\\'
WORKSPACE_PATH = join(BASE_PATH, 'workspace', 'TF-F1')

PATH_TO_SAVED_MODEL = join(WORKSPACE_PATH, 'exported-models', 'TF-F1-Model', 'saved_model')
PATH_TO_LABELS = join(WORKSPACE_PATH, 'annotations', 'label_map.pbtxt')
PATH_TO_VIDEO = join(BASE_PATH, 'scripts', 'detection', 'video-detection', 'GB-Highlights.mp4')
PATH_TO_FRAMES = join(BASE_PATH, 'scripts', 'detection', 'video-detection', 'frames')
OUTPUT_FRAMES_PATH = join(BASE_PATH, 'scripts', 'detection', 'video-detection', 'output-frames')

#setup model
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
#vidcap = cv2.VideoCapture(PATH_TO_VIDEO)
#success,frame = vidcap.read()
#count = 0

#print('Splitting video into frames')

#while success:
#    cv2.imwrite(join(PATH_TO_FRAMES, 'frame%d.png' % count), frame)       
#    success,frame = vidcap.read()
#    count += 1

frames = [f for f in os.listdir(PATH_TO_FRAMES) if isfile(join(PATH_TO_FRAMES, f))]
#frames = [join(PATH_TO_FRAMES, f) for f in os.listdir(PATH_TO_FRAMES) if isfile(join(PATH_TO_FRAMES, f))]

print('Detecting in frames')
count = 0
#for each image
for target_frame in frames: 
    IMAGE_PATH = join(PATH_TO_FRAMES, target_frame)

    print('Detecting in Frame ' + str(count))

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
                min_score_thresh=.8,
                agnostic_mode=False)

    plt.figure
    plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.savefig(join(OUTPUT_FRAMES_PATH, target_frame), bbox_inches='tight', pad_inches = 0)
    plt.close()
    count+=1
    
#os.system('ffmpeg -framerate 24 -i frame%d.png output.mp4')
