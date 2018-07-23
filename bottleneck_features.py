import getopt
import sys
import yolo_utils
import numpy as np
from tqdm import tqdm
from keras.models import load_model
from keras.models import Model

wider_path = './wider_dataset'
image_path = wider_path + '/WIDER_train/images'
bbox_path = wider_path + '/wider_face_train_bbx_gt.txt'
batch_size = 10
steps = 50

#take parameter from terminal
try:
    opts, args = getopt.getopt(sys.argv[1:], 'b:s:', ['batch_size=', 'steps='])
except getopt.GetoptError:
    sys.exit(2)

for opt, arg in opts:
	if opt in ('-b', '--batch_size'):
		batch_size = int(arg)
	elif opt in ('-s', '--steps'):
		steps = int(arg)
	else:
		sys.exit(2)

gen = yolo_utils.get_generator(batch_size, randomize=False, target=True)
bottleneck_feature = np.zeros((batch_size*steps, 13, 13, 1024))
bottleneck_target = np.zeros((batch_size*steps, 13, 13, 25))

model = load_model('yolo.h5')
#only take the first until the second last layer
intermediate_model = Model(inputs=model.input, outputs=model.get_layer('leaky_re_lu_8').output)

for i in tqdm(range(steps)):
	start_batch = i*batch_size
	end_batch = (i+1)*batch_size

	x, y = next(gen)
	batch_prediction = intermediate_model.predict_on_batch(x)
	
	#set values on feature
	bottleneck_feature[start_batch:end_batch] = batch_prediction
	bottleneck_target[start_batch:end_batch] = y

np.savez('bottleneck_data.npz', feature=bottleneck_feature, target=bottleneck_target)
print('Done saving array(s)')