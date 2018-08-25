import numpy as np
import keras.backend as K
import tensorflow as tf
import yolo_utils

def yolo_loss(y_true, y_pred):
    #LOSS INPUT SHAPE: (?, 13, 13, 5, 5)

    #set some variables
    LAMBDA_COORD = tf.constant(5, dtype=tf.float32)
    LAMBDA_NOOBJ = tf.constant(0.5, dtype=tf.float32)
    epsilon = tf.constant(1e-6, dtype=tf.float32)
    two_const = tf.constant(2, dtype=tf.float32)

    #the confidence of y_true, represented by boolean value
    mask = tf.cast(y_true[..., 4], tf.bool)
    neg_mask = tf.logical_not(mask)

    #these masked things are the one that has already been filtered out based on the truth
    #the shape of these masked items are (?, 5), which is the item that is filtered through the mask
    #masked true is all the items (i.e. xywhc) that has ground truth confidence 1
    #neg masked is all the items that has conf 0
    masked_true = tf.cast(tf.boolean_mask(y_true, mask), tf.float32)
    masked_pred = tf.cast(tf.boolean_mask(y_pred, mask), tf.float32)
    neg_masked_pred = tf.cast(tf.boolean_mask(y_pred, neg_mask), tf.float32)

    #slice up true and pred tensor
    #for some reason, people apply sigmoid and exp here
    masked_pred_xy = tf.sigmoid(masked_pred[..., 0:2])
    #i'm not sure about the sigmoid on masked wh
    masked_pred_wh = tf.sigmoid(masked_pred[..., 2:4])
    masked_pred_o_conf = tf.sigmoid(masked_pred[..., 4:5])
    masked_pred_no_o_conf = tf.sigmoid(neg_masked_pred[..., 4:5])
    #masked_true_c = masked_true[..., 5:] #for class classification

    masked_true_xy = masked_true[..., 0:2]
    masked_true_wh = masked_true[..., 2:4]

    total_obj_mask = tf.cast(tf.shape(masked_true)[0], tf.float32)
    total_noobj_mask = tf.cast(tf.shape(neg_masked_pred)[0], tf.float32)

    #calculate the loss
    xy_loss = tf.reduce_sum(tf.square(masked_true_xy-masked_pred_xy)) / (total_obj_mask + epsilon) / two_const
    wh_loss = tf.reduce_sum(tf.square(masked_true_wh-masked_pred_wh)) / (total_obj_mask + epsilon) / two_const
    obj_loss = tf.reduce_sum(tf.square(1-masked_pred_o_conf)) / (total_obj_mask + epsilon) / two_const
    noobj_loss = tf.reduce_sum(tf.square(0-masked_pred_no_o_conf)) / (total_noobj_mask + epsilon) / two_const

    loss = LAMBDA_COORD * (xy_loss + wh_loss) + obj_loss + LAMBDA_NOOBJ * noobj_loss
    return loss

def old_yolo_loss(y_true, y_pred):
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
    """


    """
    #unstack tensor using tf ops
    #heh apparently, the unstack process can't be done if the shape is dynamic
    #unstack the tensors
    true_tensor_list_raw = tf.unstack(y_true, axis=3)
    pred_tensor_list_raw = tf.unstack(y_pred, axis=3)

    #list that contains true_x, true_y, ..., true_conf and pred_x, pred_y, ..., pred_conf
    #note: true_x means all the x values of the anchor box, in this case, 5
    true_tensor_list = []
    pred_tensor_list = []

    for i in range(5):
        start = i
        end = 20+i #temporary solution
        interval = 4

        true_tensor_list.append(true_tensor_list_raw[start:end:interval])
        pred_tensor_list.append(pred_tensor_list_raw[start:end:interval])
    """

    #set some variables
    lambda_coord = tf.constant(5, dtype=tf.float32)
    lambda_noobj = tf.constant(0.5, dtype=tf.float32)
    anchor_size = tf.constant(5, dtype=tf.int32)
    dim_size = tf.constant(4, dtype=tf.int32) # the input size of IOU, which is [top, left, bottom, right]
    const_zero = tf.constant(0.0, dtype=tf.float32)
    total_anchor = 5

    #find only the x, y, w, h, c of the bbox and put it into a variable
    true_tensor_list = []
    pred_tensor_list = []
    #true_x = y_true[..., 0:20:4]
    #true_y = y_true[..., 1:21:4]
    #true_w = y_true[..., 2:22:4]
    #true_h = y_true[..., 3:23:4]
    #true_conf = y_true[..., 4:24:4]

    #pred_x = y_pred[..., 0:20:4]
    #pred_y = y_pred[..., 1:21:4]
    #pred_w = y_pred[..., 2:22:4]
    #pred_h = y_pred[..., 3:23:4]
    #pred_conf = y_pred[..., 4:24:4]

    for i in range(total_anchor):
        start = i
        end = 20+i #temporary solution
        interval = 4

        true_tensor_list.append(y_true[..., start:end:interval])
        pred_tensor_list.append(y_pred[..., start:end:interval])

    #I don't think that I'm stacking correctly here.
    stacked_true_tensor = tf.stack(true_tensor_list, axis=-2)
    stacked_pred_tensor = tf.stack(pred_tensor_list, axis=-2)

    #stacked_true_tensor[..., -1] is the confidence of each anchor box
    raw_ground_truth = tf.where(stacked_true_tensor[..., -1] == 1, stacked_true_tensor, tf.zeros_like(stacked_true_tensor))
    raw_prediction = tf.where(stacked_pred_tensor[..., -1] == 1, stacked_pred_tensor, tf.zeros_like(stacked_pred_tensor))
    print(idxx)
    #find where the true_conf is 1 (has a face) and 0 (has no face)
    #in this case, tf.where will return the index of face/noface
    idx = tf.where(tf.equal(true_tensor_list[4], 1))
    no_idx = tf.where(tf.equal(true_tensor_list[4], 0))

    #TODO: since the tensor is symbolic (there is no input yet), doing this might cause an error
    #don't take/get values from anything
    hm_obj = tf.shape(idx)[0] #total ground truth
    hm_noobj = tf.shape(no_idx)[0] #total anchor box with no ground truth
    hm_total = tf.add(hm_obj, hm_noobj)

    ground_truth_list = []
    prediction_list = []
    possible_prediction_list = []

    #use tf.scan instead of for loops
    for i in range(hm_total):
        #assign value to ground_truth and possible prediction
        #this K.eval(idx[i]) might also be problematic, it's accessing an empty tensor on graph construction
        if i < hm_obj:
            #indices = K.eval(idx[i])
            #TODO: I think this is also problematic, idk bro, fucking find out
            indices = idx[i]
        else:
            #indices = K.eval(no_idx[i-hm_obj])
            indices = no_idx[i-hm_obj]

        #temp_np_ground_truth = [true_x[indices[0], indices[1], indices[2]],\
        #                    true_y[indices[0], indices[1], indices[2]],\
        #                    true_w[indices[0], indices[1], indices[2]],\
        #                    true_h[indices[0], indices[1], indices[2]],\
        #                    true_conf[indices[0], indices[1], indices[2]]]

        """
        ground truth
        """

        temp_ground_truth_list = []
        for idx, true_tensor in enumerate(true_tensor_list):
            #TODO: this is wrong, true_tensor_idx is NOT true_x. 
            #true_x is true_tensor[0:20:4], or true_x = true_tensor[0, 4, 9, 14, 19]
            one_truth_tensor = true_tensor[idx]

            temp_ground_truth_list.append(one_truth_tensor[indices[0], indices[1], indices[2]])
        #temp_ground_truth is a list of tensor with xywhc values
        temp_ground_truth = tf.stack(temp_ground_truth_list)

        ground_truth_tlbr_value = yolo_utils.unscale_tensor(temp_ground_truth, indices[0], indices[1])
        ground_truth_list.append(temp_ground_truth)

        """
        prediction
        """

        #take all the anchor box for that ground truth, i.e. all the possible prediction
        #TODO: temp_np_possible value's aren't correct
        #TODO: symbolic tensor
        #temp_np_possible = [list(pred_x[indices[0], indices[1]]),\
        #                    list(pred_y[indices[0], indices[1]]),\
        #                    list(pred_w[indices[0], indices[1]]),\
        #                    list(pred_h[indices[0], indices[1]]),\
        #                    list(pred_conf[indices[0], indices[1]])]

        temp_prediction_all_list = []
        for idx, pred_tensor in enumerate(pred_tensor_list):
            #one_pred_tensor is the pred_x, pred_y, ..., pred_conf, shape (13, 13, 25)
            one_pred_tensor = pred_tensor[idx]
            #one_pred_tensor[a, b] is a tensor with shape (1, 1, 25)
            temp_prediction_all_list.append(one_pred_tensor[indices[0], indices[1]])

            temp_prediction_tlbr = []
            for k in range(total_anchor):
                start = k*5
                end = (k+1)*5

                #temp_prediction_tlbr.append(yolo_utils.unscale_tensor[indices[0], indices[1]])
                #temp_prediction_tlbr = a list of anchor box, with tlbr values
                temp_prediction_tlbr.append(yolo.utils.unscale_tensor(\
                        one_pred_tensor[indices[0], indices[1], start:end],\
                        indices[0], indices[1]))

            #flatten the tensor from (4, 5) to (20,)
            temp_prediction_tlbr = tf.reshape(temp_prediction_tlbr, [-1])

        #temp_prediction = tf.stack(temp_prediction_list)

        #temp_np_possible_tlbr = np.zeros((total_anchor, 4))
        #for k in range(total_anchor):
        #    temp_np_possible_tlbr[k] = yolo_utils.unscale_bbox(temp_np_possible[k], indices[0], indices[1], format='tlbr')

        ##flatten the list of tlbr values, a list with shape (5, 4) into a (20,)
        #possible_prediction_tlbr_value = [item for sublist in temp_np_possible_tlbr for item in sublist]
        #possible_prediction_tensor = tf.convert_to_tensor(possible_prediction_tlbr_value)
        ##appending the tlbr value for calculating iou
        #possible_prediction_list.append(possible_prediction_tensor)

        highest_IOU = const_zero
        highest_IOU_index = 0
        for j in range(total_anchor):
            begin = [j*4]
            size = [4]

            #slice the (20,) tensor into 5 parts, each corresponds to tlbr value of anchor box
            one_anchor_pred = tf.slice(temp_prediction_tlbr[i], begin, size)
            current_IOU = yolo_utils.calculate_tensor_IOU(ground_truth_list[i], one_anchor_pred)

            #if curr_iou > highest_iou, then highest_iou_index = j
            highest_IOU_index = tf.cond(tf.greater(current_IOU, highest_IOU),\
                    true_fn=lambda: j,\
                    false_fn=lambda: highest_IOU_index)

        #append the prediction anchor box (xywhc) with highest iou
        #prediction_list.append(tf.convert_to_tensor(temp_np_possible[highest_IOU_index]))
        #temp_prediction_list.append(temp_prediction_all_list[highest_IOU_index])
        #prediction_list.append(temp_prediction_all_list[highest_IOU_index])
        prediction_list.append(tf.slice(temp_prediction_all_list, [highest_IOU_index], [4]))

    #stacking the list of tensor into one big tensor
    ground_truth = tf.cast(tf.stack(ground_truth_list), dtype=tf.float32)
    prediction = tf.cast(tf.stack(prediction_list), dtype=tf.float32)

    #slice the whole tensor into bb with and without object
    obj_prediction = tf.slice(prediction, [0, 0], [hm_obj, total_anchor])
    noobj_prediction = tf.slice(prediction, [hm_obj, 0], [hm_noobj, total_anchor])
    obj_truth = tf.slice(ground_truth, [0, 0], [hm_obj, total_anchor])
    noobj_truth = tf.slice(ground_truth, [hm_obj, 0], [hm_noobj, total_anchor])

    #tf.unstack will return a list of tensor, with each item on the list corresponds to the unstacked items
    #e.g. unstack_ground_truth_obj[0] = obj_truth[..., 0]
    #the tf.unstack can get the x, y, w, h, c, i.e. [..., 0], [..., 1], etc
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
