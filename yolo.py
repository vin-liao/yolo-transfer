import os
import getopt
import sys
from keras.models import load_model
import cv2
import yolo_utils
import loss
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Conv2D, Input, Reshape
from keras.optimizers import Adam, rmsprop
import keras.backend as K

wider_path = './wider_dataset'
image_path = wider_path + '/WIDER_train/images'
bbox_path = wider_path + '/wider_face_train_bbx_gt.txt'
hm_epoch = 20
hm_steps = 100
batch_size = 32

try:
    opts, args = getopt.getopt(sys.argv[1:], 'e:s:b:', ['epoch=', 'steps=', 'batch_size='])
except getopt.GetoptError:
    sys.exit(2)

for opt, arg in opts:
	if opt in ('-e', '--epoch'):
		hm_epoch = int(arg)
	elif opt in ('-s', '--steps'):
		hm_steps = int(arg)
	elif opt in ('-b', '--batch_size'):
		batch_size = int(arg)
	else:
		sys.exit(2)

gen = yolo_utils.get_generator_bottleneck(batch_size)

#===Training the new layer===
train_input = Input(shape=(13, 13, 1024), name='leaky_re_lu_8')
#im creating a 25 filters, I wanted a 5x5 ones. Maybe I could reshape this thing
train_output_raw = Conv2D(25, (1, 1), name='conv2d_train')(train_input)
train_output = Reshape((13, 13, 5, -1))(train_output_raw)

train_model = Model(inputs=train_input, outputs=train_output)
train_model.summary()

train_model.compile(optimizer='rmsprop', loss=loss.yolo_loss, metrics=['accuracy'])
train_model.fit_generator(gen, epochs=hm_epoch, steps_per_epoch=hm_steps)

#===Concatenating the model===
base_model = load_model('yolo.h5')

#remove the last 2 layers, and make it a new model
base_model = Model(inputs=base_model.input, outputs=base_model.layers[-3].output)

#combine the base model, with the trained transfered model
model_input = Input(shape=(416, 416, 3))
intermediate_output = base_model(model_input)
prediction = train_model(intermediate_output)

new_model = Model(inputs=model_input, outputs=prediction)
new_model.save('transfered_model.h5')
print('Done saving new transfered model.')
