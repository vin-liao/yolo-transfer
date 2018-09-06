import yolo_utils
from keras.models import load_model, Model
from keras.layers import Reshape, Conv2D, Input

class YOLO():
    def __init__(self):
        pass

    #def FullYolo():

    #    ...

    #    model.summary()
    #    return model

    def TinyYolo(self):
        base_model = load_model('./keras_models/yolo.h5')

        model_input = Input(shape=(416, 416, 3))

        #from layer 1 to last feature extractor (Leaky ReLU)
        base_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

        detection_layer = Conv2D(25,
                        (1, 1),
                        strides=(1, 1),
                        padding='same',
                        kernel_initializer='lecun_normal')

        intermediate_output = base_model(model_input)
        prediction = detection_layer(intermediate_output)
        prediction = Reshape((13, 13, 5, -1))(prediction)

        model = Model(inputs=model_input, outputs=prediction)

        model.summary()
        return model
