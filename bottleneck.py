import getopt
import sys
from keras.models import load_model
import yolo_utils
import numpy as np
from keras.models import Model

wider_path = './wider_dataset'
image_path = wider_path + '/WIDER_train/images'
bbox_path = wider_path + '/wider_face_train_bbx_gt.txt'
quantity = 100

#create input from terminal
try:
    opts, args = getopt.getopt(sys.argv[1:], 'q:', ['qty='])
except getopt.GetoptError:
    sys.exit(2)

for opt, arg in opts:
	if opt in ('-q', '--qty'):
		quantity = int(arg)
	else:
		sys.exit(2)

data = yolo_utils.get_data(quantity)
model = load_model('yolo.h5')

#only take the first until the second last layer
intermediate_model = Model(inputs=model.input, outputs=model.get_layer('leaky_re_lu_8').output)
bottleneck_features = intermediate_model.predict_generator(data, steps=1)
np.save(open('bottleneck_features.npy', 'wb'), bottleneck_features)
print('Done saving bottleneck features.')