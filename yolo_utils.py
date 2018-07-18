import os
import re
import numpy as np
import cv2
import random

wider_path = './wider_dataset'
image_path = wider_path + '/WIDER_train/images'
bbox_path = wider_path + '/wider_face_train_bbx_gt.txt'
anchors_path = './yolo_anchors.txt'

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

def get_generator(batch_size, randomize=True):
	with open(anchors_path, 'r') as f:
		anchors_raw = f.read()

	with open(bbox_path, 'r') as f:
		bbox_raw = f.read()

	#todo: generate data from npy file

	anchors = get_anchors(anchors_raw)
	name_list = get_image_names(image_path)

	#make these np zeros more general
	batch_image = np.zeros((batch_size, 416, 416, 3))
	batch_target = np.zeros((batch_size, 13, 13, 25))

	while True:
		image_list = []
		target_list = []

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

			# image_list.append(cv2.resize(image, (416, 416)))
			image = crop_square(image)
			batch_image[i] = cv2.resize(image, (416, 416))

			bounding_box = get_bbox(bbox_raw, image_name)
			# target_list.append(create_target(image, bounding_box, anchors))

			#the image here is cropped image, not the resized image
			#if the bbox is outside the cropped image, then it automatically skips
			batch_target[i] = create_target(image, bounding_box, anchors)

		#yield image and target, target uses create_target()
		# yield (image_list, target_list)
		yield batch_image, batch_target

def get_data(quantity=100):
	with open(anchors_path, 'r') as f:
		anchors_raw = f.read()

	with open(bbox_path, 'r') as f:
		bbox_raw = f.read()

	anchors = get_anchors(anchors_raw)
	name_list = get_image_names(image_path)

	#make these np zeros more general
	image_list = np.zeros((quantity, 416, 416, 3))
	target_list = np.zeros((quantity, 13, 13, 25))

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

		# image_list.append(cv2.resize(image, (416, 416)))
		image = crop_square(image)
		image_list[i] = cv2.resize(image, (416, 416))

		bounding_box = get_bbox(bbox_raw, image_name)
		# target_list.append(create_target(image, bounding_box, anchors))

		#the image here is cropped image, not the resized image
		#if the bbox is outside the cropped image, then it automatically skips
		target_list[i] = create_target(image, bounding_box, anchors)

	return image_list, target_list

def create_target(image, bboxes, anchors, grid_size=13):
	"""
	bboxes = ground truth bounding box with [left top width height] information

	anchors = a list of anchors, each elements has a list of two numbers which is the 
	width and height of the anchor boxes relative to the grid 
	"""

	image_target = np.zeros((grid_size, grid_size, 5 * len(anchors)))
	for bbox in bboxes:
		#x1 and y1 are left and top respectively
		left = bbox[0]
		top = bbox[1]
		w = bbox[2]
		h = bbox[3]

		x = int(left+(w/2))
		y = int(top+(h/2))

		#if bounding box is outside the cropped image, continue
		if left+w > image.shape[0] or top+h > image.shape[1]:
			continue

		#assign value to volume, if the x, y coordinate falls between a value in the grid
		#then assign value on that grid
		for row in range(grid_size):
			#TODO: if this is rounded, the final pixel is losing value
			grid_row = row*image.shape[0]/grid_size
			grid_row_plus = (row+1)*image.shape[0]/grid_size

			for col in range(grid_size):
				grid_col = col*image.shape[1]/grid_size
				grid_col_plus = (col+1)*image.shape[1]/grid_size

				#if the x and y value is inside the grid
				if (x > grid_row and x < grid_row_plus) and\
					(y > grid_col and y < grid_col_plus):

					#scaled x and y relative to grid
					scaled_x = round((x-grid_row)/(grid_row_plus-grid_row), 5)
					scaled_y = round((y-grid_col)/(grid_col_plus-grid_col), 5)

					#scaled height and width relative to grid
					scaled_w = round(w/(grid_row_plus-grid_row), 5)
					scaled_h = round(h/(grid_col_plus-grid_col), 5)

					#scale the anchor boxes
					#if yolov2
					offset_num = []
					for anchor in anchors:
						#ground truth/anchor boxes
						scaled_anchor_w = scaled_w/anchor[0]
						scaled_anchor_h = scaled_h/anchor[1]
						offset_num.append(scaled_anchor_h + scaled_anchor_w)

					"""
					find the index of the list which has the value closest to 2
					the smallest amount of change to a scale is 1, if w or h is scaled by 1 it means
					that it doesn't change, and having the least amount of change/offset is the objective.
					since there are 2 of them i.e. w and h, we calculate the smallest offset for w and h
					hence the number 2 here, which is 1 for w and 1 for h
					"""

					anchor_index = offset_num.index(min(offset_num, key=lambda x:abs(x-2)))
					best_anchor = anchors[anchor_index]

					#scalar
					#the size of ground truths relative to the grid / anchor sizes
					anchor_scalar_w = scaled_w/best_anchor[0]
					anchor_scalar_h = scaled_h/best_anchor[1]

					#size of ground truth bbox = anchors * anchor scales
					scaled_anchor_w = best_anchor[0]*anchor_scalar_w
					scaled_anchor_h = best_anchor[1]*anchor_scalar_h

					image_target[row, col, anchor_index*len(anchors):(anchor_index+1)*len(anchors)] = \
					[scaled_x, scaled_y, scaled_anchor_w, scaled_anchor_h, 1]

	return image_target