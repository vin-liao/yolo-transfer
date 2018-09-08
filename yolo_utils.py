import os
import re
import numpy as np
import cv2
import random
import keras.backend as K
import tensorflow as tf

wider_path = './wider_dataset'
image_path = wider_path + '/WIDER_train/images'
bbox_path = wider_path + '/wider_face_train_bbx_gt.txt'
anchors_path = './keras_models/yolo_anchors.txt'

def sigmoid(arr):
  return 1 / (1 + np.exp(-arr))

def xywh_to_tlbr(x, y, w, h):
    #x y width height -> top left bottom right
    t = int(y - (h / 2))
    l = int(x - (w / 2))
    b = int(y + (h / 2))
    r = int(x + (w / 2))
    return [t, l, b, r]

def tlbr_to_xywh(t, l, b, r):
    #top left bottom right -> x y width height
    w = int(r - l)
    h = int(b - t)
    x = int(w / 2)
    y = int(h / 2)
    return [x, y, w, h]

def get_image_names(path_to_img):
    #get all name of the images
    name_list = []
    category = os.listdir(image_path)
    for cat in category:
        img_list = os.listdir(image_path + '/{}'.format(cat))
        for image in img_list:
            name_list.append(image)

    return name_list

def crop_square(image):
    #crop the image from the top left for 1:1 ratio
    smallest = image.shape[0]
    if image.shape[1] < smallest:
        smallest = image.shape[1]

    return image[0:smallest, 0:smallest]

def get_bbox(bbox_raw, image_name):
    #this bbox extraction process only works for wider dataset
    bounding_box = []
    raw_bbox_text = bbox_raw

    #find the bounding box text with the corresponding name
    bbox_of_image = re.findall('{}\n[\n\s\d]+'.format(image_name), raw_bbox_text)[0]
    #clean the bounding box text
    #remove the last number
    bbox_of_image = bbox_of_image[:bbox_of_image.rfind('\n')]
    #get the bounding box string from the raw text
    bbox_of_image = bbox_of_image[bbox_of_image.find('\n', bbox_of_image.find('\n')):]
    #remove the count number from bounding box
    bbox_of_image = re.sub('\n\d+\n', '', bbox_of_image)
    bbox_of_image = bbox_of_image.split('\n')

    for one_bbox in bbox_of_image:
        #split the list which is separated by space
        one_bbox = one_bbox.split()
        #take the first four of the list, which are the left top width height respectively
        one_bbox = one_bbox[:4]
        bounding_box.append([int(i) for i in one_bbox])

    return bounding_box

def get_anchors(anchors_raw):
    #it's separated by two spaces
    anchors = anchors_raw.split('  ')

    for i, anchor in enumerate(anchors):
        #cleaning the string
        if anchor[-1] == ',':
            #remove the character ',' if present
            anchors[i] = anchor[:-1]
        if '\n' in anchor:
            #remove the string \n if present
            anchors[i] = re.sub('\n', '', anchor)

        #turning each anchor pairs into list, and casting it into float
        str_anchor = anchors[i].split(',')
        anchors[i] = [float(x) for x in anchors[i].split(',')]

    return anchors

def get_generator(batch_size=32, randomize=True, target=True):
    """
    batch_size: how many items would the generator yield per iteration

    randomize: not generating random data will make sure that we won't yield the same data
    this is useful for the bottleneck because creating bottleneck features only passes the data once.

    target: whether to yield target array--the ground truth bounding box--or not.
    """

    #TODO: this code is pretty dirty, maybe refactor later
    with open(anchors_path, 'r') as f:
        anchors_raw = f.read()

    with open(bbox_path, 'r') as f:
        bbox_raw = f.read()

    anchors = get_anchors(anchors_raw)
    name_list = get_image_names(image_path)

    batch_image = np.zeros((batch_size, 416, 416, 3), dtype=np.uint8)
    if target:
        batch_target = np.zeros((batch_size, 13, 13, len(anchors), 5))

    while True:
        for i in range(batch_size):
            if randomize:
                #select random name from the list
                seed = np.random.choice(len(name_list))
                image_name = name_list[seed]
            else:
                image_name = name_list[i]

            for root, dirs, files in os.walk(image_path):
                for file in files:
                    if image_name in file:
                        image = cv2.imread(root + '/' + image_name)

            batch_image[i] = cv2.resize(image, (416, 416))

            if target:
                bounding_box = get_bbox(bbox_raw, image_name)

                #the image here is cropped image, not the resized image
                #if the bbox is outside the cropped image, then it automatically skips
                batch_target[i] = create_target(image, bounding_box, anchors)

            """
            #For debugging purpose
            image_bbox_test = create_bbox(image, batch_target[i], activation=False)
            for box in image_bbox_test:
                cv2.rectangle(image, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 3)

            cv2.imshow('draw', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """

        if randomize == False:
            #move the used data to the back of list
            #this is to avoid using the data over and over again
            name_list = name_list[batch_size:] + name_list[:batch_size]

        if target:
            yield batch_image, batch_target
        else:
            yield batch_image

def get_generator_bottleneck(batch_size=32):
    #this is problematic and causes memory error
    data = np.load('bottleneck_data.npz', mmap_mode='r')
    x = data['feature']
    y = data['target']

    #x.shape[1], y.shape[1], ..., x.shape[3], y.shape[3] is the data dimension.
    batch_x = np.zeros((batch_size, x.shape[1], x.shape[2], x.shape[3]))
    batch_y = np.zeros((batch_size, y.shape[1], y.shape[2], y.shape[3], y.shape[4]))

    while True:
        for i in range(batch_size):
            #get seed
            seed = np.random.choice(len(x))
            batch_x[i] = x[seed]
            batch_y[i] = y[seed]
        yield batch_x, batch_y

def get_data(quantity=5, get_sample=False):
    with open(anchors_path, 'r') as f:
        anchors_raw = f.read()

    with open(bbox_path, 'r') as f:
        bbox_raw = f.read()

    anchors = get_anchors(anchors_raw)
    name_list = get_image_names(image_path)

    if quantity > 0:
        #use only "quantity" amount of data
        name_list = name_list[:quantity]
    else:
        #use all the data
        quantity = len(name_list)

    #make these np zeros more general
    image_list = np.zeros((quantity, 416, 416, 3), dtype=np.uint8)
    target_list = np.zeros((quantity, 13, 13, len(anchors), 5))

    #randomize the list
    random.shuffle(name_list)
    if quantity > 0:
        name_list = name_list[:quantity]

    for i, image_name in enumerate(name_list):
        #find the image from folder
        for root, dirs, files in os.walk(image_path):
            for file in files:
                if image_name in file:
                    image = cv2.imread(root + '/' + image_name)

        if get_sample:
            return image
        image_list[i] = cv2.resize(image, (416, 416))

        bounding_box = get_bbox(bbox_raw, image_name)
        # target_list.append(create_target(image, bounding_box, anchors))

        #the image here is cropped image, not the resized image
        #if the bbox is outside the cropped image, then it automatically skips
        target_list[i] = create_target(image, bounding_box, anchors)

        """
        #For debugging purpose
        image_bbox_test = create_bbox(image_list[i], target_list[i], activation=False)
        for box in image_bbox_test:
            cv2.rectangle(image_list[i], (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 3)

        cv2.imshow('draw', image_list[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """

    return image_list, target_list

def create_target(image, bboxes, anchors, total_grid=13):
    """
    bboxes: ground truth bounding box with [left top width height] information

    anchors: a list of anchors, each elements has a list of two numbers which is the
    width and height of the anchor boxes relative to the grid

    you also need to choose the best place to put these values based on iou values
    """

    #TODO: add class on image target

    img_h, img_w = image.shape[:2]
    image_target = np.zeros((total_grid, total_grid, len(anchors), 5))
    for bbox in bboxes:
        #x1 (bbox[0]) and y1 (bbox[1]) are left and top respectively
        left = bbox[0]
        top = bbox[1]
        w = bbox[2]
        h = bbox[3]

        x = int(left+(w/2))
        y = int(top+(h/2))
        bottom = left + w
        right = top + h

        relative_x = x / float(img_w)
        relative_y = y / float(img_h)
        relative_w = w / float(img_w)
        relative_h = h / float(img_h)

        #get the row and column of the xy coordinate
        row = int(np.floor(relative_y * float(total_grid)))
        col = int(np.floor(relative_x * float(total_grid)))

        #calculate iou, and place the box on the anchor box with highest iou
        highest_iou = -1
        best_anchor_idx = -1
        for i, anchor in enumerate(anchors):
            bbox = [0, 0, bottom, right]
            anchor_w = anchor[0] * (float(img_w) / float(total_grid))
            anchor_h = anchor[1] * (float(img_h) / float(total_grid))

            anchor_box = [0, 0, anchor_w, anchor_h]
            iou = calculate_iou(bbox, anchor_box)
            if iou > highest_iou:
                highest_iou = iou
                best_anchor_idx = i

        image_target[col, row, best_anchor_idx, :4] = [relative_x, relative_y, relative_w, relative_h]
        image_target[col, row, best_anchor_idx, 4] = 1.
        #image_target[col, row, best_anchor_idx, 5:] = class

    return image_target

def create_bbox(image, target, threshold=0.6, activation=True):
    #image parameter to get the shape
    #target shape = (13, 13, total_anchor, anchor_size)

    #xywh values are represented as a value ranging from 0...1, so to make the value of target
    #the same, we need to apply sigmoid to it.

    img_h, img_w = image.shape[:2]

    if activation:
        target[..., 0:2] = sigmoid(target[..., 0:2]) #x y
        target[..., 2:4] = sigmoid(target[..., 2:4]) #w h
        target[..., 4:5] = sigmoid(target[..., 4:5]) #conf

    idx = np.where(target[..., -1] > threshold)
    raw_bboxes = target[idx]
    idx = np.transpose(np.array(idx))

    bbox_list = []
    for i, box in enumerate(raw_bboxes):
        indices = idx[i]
        row = indices[1]
        col = indices[0]

        x = box[0] * img_w
        y = box[1] * img_h
        w = box[2] * img_w
        h = box[3] * img_h

        tlbr_value = xywh_to_tlbr(x, y, w, h)
        if all(n > 0 for n in tlbr_value):
            if tlbr_value[0] < tlbr_value[2] and tlbr_value[1] < tlbr_value[3]:
                bbox_list.append(tlbr_value)

    return bbox_list

def non_max_suppresion():
    """
    psuedocode:
    pick box with highest confidence, lets call it pred
    discard (or suppress) the box that has >0.5 iou with pred
    this means throwing everything that is around that prediction box
    """
    pass

def calculate_iou(true_bb, pred_bb):
    #true_bb: bounding box with [top left bottom right] format
    #pred_bb: prediction bounding box with [top left bottom right] format

    #determine the intersection of the box
    inter_top = max(true_bb[0], pred_bb[0])
    inter_left = max(true_bb[1], pred_bb[1])
    inter_bottom = min(true_bb[2], pred_bb[2])
    inter_right = min(true_bb[3], pred_bb[3])

    if inter_top > inter_bottom or inter_left > inter_right:
        #if there is no intersection, iou becomes 0
        return 0.0

    #calculate the area of rectangle
    intersection_area = (inter_bottom - inter_top) * (inter_right - inter_left)
    true_bb_area = (true_bb[2] - true_bb[0]) * (true_bb[3] - true_bb[1])
    pred_bb_area = (pred_bb[2] - pred_bb[0]) * (pred_bb[3] - pred_bb[1])

    #calculate the iou, dividing area of overlap by area of union
    iou = intersection_area / float(true_bb_area + pred_bb_area - intersection_area)

    if iou <= 1.0 and iou >= 0.0:
        return iou
    else:
        raise valueerror('incorrect iou value: %d' %(iou))

def calculate_tensor_iou(true_bb, pred_bb):
    #true_bb: ground truth tensor with [top left bottom right] format
    #pred_bb: prediction tensor with [top left bottom right] format

    #determine the intersection of the box
    inter_top = tf.maximum(true_bb[0], pred_bb[0])
    inter_left = tf.maximum(true_bb[1], pred_bb[1])
    inter_bottom = tf.minimum(true_bb[2], pred_bb[2])
    inter_right = tf.minimum(true_bb[3], pred_bb[3])

    #calculate the area of rectangle
    intersection_area = tf.multiply(tf.subtract(inter_top, inter_bottom), tf.subtract(inter_right, inter_left))
    true_bb_area = tf.multiply(tf.subtract(true_bb[0], true_bb[2]), tf.subtract(true_bb[1], true_bb[3]))
    pred_bb_area = tf.multiply(tf.subtract(pred_bb[0], pred_bb[2]), tf.subtract(pred_bb[1], pred_bb[3]))

    #iou is calculated by dividing the intersection area, with area of union
    #area of union = true_bb_area + pred_bb_area - intersection_area

    #if there is no intersection between the two bb, intersection area is 0
    iou = tf.divide(intersection_area, tf.subtract(tf.add(true_bb_area, pred_bb_area), intersection_area))
    iou = tf.cast(iou, dtype=tf.float32)

    """
    the tf.cond below is equivalent with

    if iou < 1.0 and iou > 0.0:
        return iou
    else:
        raise valueerror('incorrect iou value: %d' %(iou))
    """

    tf.cond(tf.logical_and(tf.greater(iou, tf.constant(0.0)),\
            tf.greater(tf.constant(1.0), iou)),\
            true_fn=lambda: iou,\
            false_fn=lambda: tf.constant(-1.0, dtype=tf.float32))

    iou = tf.clip_by_value(iou, 0.0, 1.0)
    return iou
