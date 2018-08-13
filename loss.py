import numpy as np
import keras.backend as K
import tensorflow as tf
import yolo_utils

def yolo_loss(y_true, y_pred):
    """
    there are 5 parts for this loss
    to put it in other words, these 5 things are penalized

    1. if there is an object in grid i, and anchor box j is responsible for predicting it,
        penalize the coordinate (x, y)
    2. if there is an object in grid i, and anchor box j is responsible for predicting it,
        penalize the size (h, w)
    3. if there is an object in grid i, and anchor box j is responsible for predicting it,
        penalize the confidence (slightly increase it)
    4. if there is NO object in grid i, and anchor box j is responsible for predicting it,
        penalize the confidence (slightly decrease it)
    5. if there is an object at grid i, penalize the wrong classification

    The word responsible in here means: the anchor box that has the highest IOU with the
    ground truth. To put it in other words, if the box's job is to predict the object
    in that grid, then penalize it--adjust the weight so that the prediction of that
    particular anchor box is better

    what to mse?
    penalize the bounding box value with the ground truth value
    e.g. to penalize the coordinate, take the best anchor box (let's say it's 3) then
        mse it to the ground truth
        pred x, y = 0.3, 0.5
        true x, y = 0.2, 0.9

        xy_loss = mse((0.2, 0.9), (0.3, 0.5))
    the rest of the loss is the same thing, just different position / index

    TODO: can you vectorize this instead of looping like a fucking noob?
    """
    #find only the x, y, w, h, c of the bbox and put it into a variable
    true_x = y_true[..., 0:20:4]
    true_y = y_true[..., 1:21:4]
    true_w = y_true[..., 2:22:4]
    true_h = y_true[..., 3:23:4]
    true_conf = y_true[..., 4:24:4]

    pred_x = y_pred[..., 0:20:4]
    pred_y = y_pred[..., 1:21:4]
    pred_w = y_pred[..., 2:22:4]
    pred_h = y_pred[..., 3:23:4]
    pred_conf = y_pred[..., 4:24:4]

    #set some variables
    lambda_coord = tf.constant(5, dtype=tf.float32)
    lambda_noobj = tf.constant(0.5, dtype=tf.float32)
    anchor_size = tf.constant(5, dtype=tf.int32)
    dim_size = tf.constant(4, dtype=tf.int32) # the input size of IOU, which is [top, left, bottom, right]
    const_zero = tf.constant(0.0, dtype=tf.float32)

    #find where the true_conf is 1 (has a face) and 0 (has no face)
    idx = tf.where(tf.equal(true_conf, 1))
    no_idx = tf.where(tf.equal(true_conf, 0))

    hm_obj = K.eval(idx).shape[0] #total ground truth
    hm_noobj = K.eval(no_idx).shape[0] #total anchor box with no ground truth
    hm_total = hm_obj + hm_noobj
    total_anchor = 5

    ground_truth_list = []
    prediction_list = []
    possible_prediction_list = []

    for i in range(hm_total):
        #assign value to ground_truth and possible prediction
        if i < hm_obj:
            indices = K.eval(idx[i])
        else:
            indices = K.eval(no_idx[i-hm_obj])

        temp_np_ground_truth = [true_x[indices[0], indices[1], indices[2]],\
                            true_y[indices[0], indices[1], indices[2]],\
                            true_w[indices[0], indices[1], indices[2]],\
                            true_h[indices[0], indices[1], indices[2]],\
                            true_conf[indices[0], indices[1], indices[2]]]

        #turn the np variables into top, left, bottom, right format
        ground_truth_tlbr_value = yolo_utils.unscale_bbox(temp_np_ground_truth, indices[0], indices[1], format='tlbr')
        ground_truth_tensor = tf.convert_to_tensor(ground_truth_tlbr_value)

        #put the xywhc format tensor instead of tlbr. tlbr is only used for calculating iou
        ground_truth_list.append(tf.convert_to_tensor(temp_np_ground_truth))

        #take all the anchor box for that ground truth, i.e. all the possible prediction
        #TODO: temp_np_possible value's aren't correct
        temp_np_possible = [list(pred_x[indices[0], indices[1]]),\
                            list(pred_y[indices[0], indices[1]]),\
                            list(pred_w[indices[0], indices[1]]),\
                            list(pred_h[indices[0], indices[1]]),\
                            list(pred_conf[indices[0], indices[1]])]

        temp_np_possible_tlbr = np.zeros((total_anchor, 4))
        for k in range(total_anchor):
            temp_np_possible_tlbr[k] = yolo_utils.unscale_bbox(temp_np_possible[k], indices[0], indices[1], format='tlbr')

        #flatten the list of tlbr values, a list with shape (5, 4) into a (20,)
        possible_prediction_tlbr_value = [item for sublist in temp_np_possible_tlbr for item in sublist]
        possible_prediction_tensor = tf.convert_to_tensor(possible_prediction_tlbr_value)
        #appending the tlbr value for calculating iou
        possible_prediction_list.append(possible_prediction_tensor)

        highest_IOU = const_zero
        highest_IOU_index = 0
        for j in range(total_anchor):
            begin = [j*4]
            size = [4]

            #slice the (20,) tensor into 5 parts, each corresponds to tlbr value of anchor box
            one_anchor_pred = tf.slice(possible_prediction_list[i], begin, size)
            current_IOU = yolo_utils.calculate_tensor_IOU(ground_truth_list[i], one_anchor_pred)

            #if curr_iou > highest_iou, then highest_iou_index = j
            highest_IOU_index = tf.cond(tf.greater(current_IOU, highest_IOU),\
                    true_fn=lambda: j,\
                    false_fn=lambda: highest_IOU_index)

        #append the prediction anchor box (xywhc) with highest iou
        highest_IOU_index = K.eval(highest_IOU_index)
        prediction_list.append(tf.convert_to_tensor(temp_np_possible[highest_IOU_index]))

    #stacking the list of tensor into one big tensor
    ground_truth = tf.cast(tf.stack(ground_truth_list), dtype=tf.float32)
    prediction = tf.cast(tf.stack(prediction_list), dtype=tf.float32)

    #slice the whole tensor into bb with and without object
    #error is raised if there is no bounding box -- i.e. 0 hm_obj.
    obj_prediction = tf.slice(prediction, [0, 0], [hm_obj, total_anchor])
    noobj_prediction = tf.slice(prediction, [hm_obj, 0], [hm_noobj, total_anchor])
    obj_truth = tf.slice(ground_truth, [0, 0], [hm_obj, total_anchor])
    noobj_truth = tf.slice(ground_truth, [hm_obj, 0], [hm_noobj, total_anchor])

    #tf.unstack will return a list of tensor, with each item on the list corresponds to the unstacked items
    #e.g. unstack_ground_truth_obj[0] = obj_truth[..., 0]
    #the tf.unstack can only the x, y, w, h, c, i.e. [..., 0], [..., 1], etc
    unstacked_ground_truth_obj = tf.unstack(obj_truth, axis=1)
    unstacked_ground_truth_noobj = tf.unstack(noobj_truth, axis=1)
    unstacked_prediction_obj = tf.unstack(obj_prediction, axis=1)
    unstacked_prediction_noobj = tf.unstack(noobj_prediction, axis=1)

    #calculate the loss of each part
    xy_loss = lambda_coord * (K.mean(K.square(unstacked_prediction_obj[0] - unstacked_ground_truth_obj[0]) + \
        K.square(unstacked_prediction_obj[1] - unstacked_ground_truth_obj[1]), axis=-1))

    wh_loss = lambda_coord * (K.mean(K.square(K.sqrt(unstacked_prediction_obj[2]) - K.sqrt(unstacked_ground_truth_obj[2])) + \
            K.square(K.sqrt(unstacked_prediction_obj[3]) - K.sqrt(unstacked_ground_truth_obj[3])), axis=-1))

    cobj_loss = K.mean(K.square(unstacked_prediction_obj[4] - unstacked_ground_truth_obj[4]), axis=-1)

    cnoobj_loss = lambda_noobj * K.mean(K.square(unstacked_prediction_noobj[4] - unstacked_ground_truth_noobj[4]), axis=-1)

    #if the bounding box is 0, (i.e. hm_obj == 0) then loss is nan, to prevent this from having 
    #an error, the nan value will be replaced with a 0 tensor
    xy_loss, wh_loss, cobj_loss = tf.cond(tf.is_nan(xy_loss),\
            true_fn = lambda: [const_zero, const_zero, const_zero],\
            false_fn = lambda: [xy_loss, wh_loss, cobj_loss])

    return xy_loss + wh_loss + cobj_loss + cnoobj_loss
