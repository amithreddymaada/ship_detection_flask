import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import cv2

import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

import numpy as np
from PIL import Image

def ship_detection_init():

    tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

    # Enable GPU dynamic memory allocation
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


    PATH_TO_SAVED_MODEL = './my_model/saved_model'
    PATH_TO_LABELS = './my_model/label_map.pbtxt'

    print('Loading model...', end='')
    start_time = time.time()

    # Load saved model and build the detection function
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))


    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                        use_display_name=True)

    return detect_fn, category_index




def load_image_into_numpy_array(path):
    img = np.array(Image.open(path))
    
    if len(img.shape) > 2 and img.shape[2] == 4:
        #convert the image from RGBA2RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def get_boxes_on_image(image_np, detect_fn, category_index):
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections
    
    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)
    
    return image_np_with_detections


def detect_ships(file_path, out_filename, detect_fn, category_index, is_image=True):
    if is_image:
        image_np = load_image_into_numpy_array(file_path)
        image_np_with_detections = get_boxes_on_image(image_np, detect_fn, category_index)
        image_np_with_detections = cv2.cvtColor(image_np_with_detections, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"./static/image_uploads/{out_filename}", image_np_with_detections)
    else:
        # Read the video
        vidcap = cv2.VideoCapture(file_path)
        frame_read, image = vidcap.read()
        height, width= image.shape[0], image.shape[1]
        
        count = 0
        
        # Set output video writer with codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f"./static/video_uploads/{out_filename}", fourcc, 25.0, (width, height))
        
        
        # Iterate over frames and pass each for prediction
        while frame_read:
            # Perform object detection and add to output file
            output_file = get_boxes_on_image(image, detect_fn, category_index)
            
            output_file = cv2.resize(output_file, (width, height), interpolation=cv2.INTER_CUBIC)
            
            # Write frame with predictions to video
            out.write(output_file)

            # Read next frame
            frame_read, image = vidcap.read()
            count += 1
            

        # Release video file when we're ready
        out.release()