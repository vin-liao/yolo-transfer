import os
import getopt
import sys
import yolo_utils
import loss
from keras.optimizers import Adam, rmsprop
import models

wider_path = './wider_dataset'
image_path = wider_path + '/WIDER_train/images'
bbox_path = wider_path + '/wider_face_train_bbx_gt.txt'
hm_epoch = 30
hm_steps = 30
batch_size = 32
learning_rate = 0.001

try:
    opts, args = getopt.getopt(sys.argv[1:], 'e:s:b:l:', ['epoch=', 'steps=', 'batch_size=', 'lr='])
except getopt.GetoptError:
    sys.exit(2)

for opt, arg in opts:
	if opt in ('-e', '--epoch'):
		hm_epoch = int(arg)
	elif opt in ('-s', '--steps'):
		hm_steps = int(arg)
	elif opt in ('-b', '--batch_size'):
		batch_size = int(arg)
	elif opt in ('-l', '--lr'):
		learning_rate = float(arg)
	else:
		sys.exit(2)

gen = yolo_utils.get_generator(batch_size)
val_data = yolo_utils.get_data(100)

model_obj = models.YOLO()
model = model_obj.TinyYolo()

opt = Adam(lr=learning_rate)
model.compile(optimizer=opt, loss=loss.yolo_loss)
model.fit_generator(gen, epochs=hm_epoch, steps_per_epoch=hm_steps, validation_data=val_data)

new_model.save('tiny_yolo.h5')
print('Done saving new transfered model.')
