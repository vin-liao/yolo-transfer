import numpy as np
import keras.backend as K
import tensorflow as tf
import yolo_utils

def yolo_loss(y_true, y_pred):
    #turn this to true to print each loss values, useful for debugging
    debug_loss = False
    debug_value = False

    #LOSS INPUT SHAPE: (?, 13, 13, 5, 5)

    LAMBDA_COORD = tf.constant(5, dtype=tf.float32)
    LAMBDA_NOOBJ = tf.constant(0.5, dtype=tf.float32)
    epsilon = tf.constant(1e-6, dtype=tf.float32)

    #mask, a boolean tensor, which value depends on the y_true confidence
    mask = tf.equal(y_true[..., 4], 1)
    neg_mask = tf.equal(y_true[..., 4], 0)

    #FIXME: apparently, using boolean mask created a warning
    #https://stackoverflow.com/questions/44380727/get-userwarning-while-i-use-tf-boolean-mask

    masked_true = tf.cast(tf.boolean_mask(y_true, mask), tf.float32)
    masked_pred = tf.cast(tf.boolean_mask(y_pred, mask), tf.float32)
    neg_masked_pred = tf.cast(tf.boolean_mask(y_pred, neg_mask), tf.float32)


    """adjusting prediction mask"""
    masked_pred_xy = tf.sigmoid(masked_pred[..., 0:2])

    #logically, sigmoid works the best here because the ground truth of
    #wh values are a range between 0...1
    masked_pred_wh = tf.sigmoid(masked_pred[..., 2:4])

    masked_pred_o_conf = tf.sigmoid(masked_pred[..., 4:5])
    masked_pred_no_o_conf = tf.sigmoid(neg_masked_pred[..., 4:5])
    #masked_true_c = masked_true[..., 5:] #for class classification

    """adjusting ground truth mask"""
    masked_true_xy = masked_true[..., 0:2]
    masked_true_wh = masked_true[..., 2:4]

    #total object for each mask
    total_obj_mask = tf.cast(tf.shape(masked_true)[0], tf.float32)
    total_noobj_mask = tf.cast(tf.shape(neg_masked_pred)[0], tf.float32)

    #calculate the loss
    #each loss value is divided by a number, which is the total mask object. This is used for normalization
    #so that the loss doesn't get too large
    xy_loss = tf.reduce_sum(tf.square(masked_true_xy-masked_pred_xy)) / (total_obj_mask + epsilon)
    wh_loss = tf.reduce_sum(tf.square(masked_true_wh-masked_pred_wh)) / (total_obj_mask + epsilon)
    obj_loss = tf.reduce_sum(tf.square(1-masked_pred_o_conf)) / (total_obj_mask + epsilon)
    noobj_loss = tf.reduce_sum(tf.square(0-masked_pred_no_o_conf)) / (total_noobj_mask + epsilon)

    loss = LAMBDA_COORD * (xy_loss + wh_loss) + obj_loss + LAMBDA_NOOBJ * noobj_loss
    #debugging
    if debug_loss:
        loss = tf.Print(loss, [xy_loss], '\nxy loss:')
        loss = tf.Print(loss, [wh_loss], 'wh loss:')
        loss = tf.Print(loss, [obj_loss], 'obj loss:')
        loss = tf.Print(loss, [noobj_loss], 'noobj loss:')

    if debug_value:
        loss = tf.Print(loss, [y_pred[..., 0]], '\nThe value of prediction x:', summarize=10)
        loss = tf.Print(loss, [y_pred[..., 1]], 'The value of prediction y:', summarize=10)
        loss = tf.Print(loss, [y_pred[..., 2]], 'The value of prediction w:', summarize=10)
        loss = tf.Print(loss, [y_pred[..., 3]], 'The value of prediction h:', summarize=10)
        #loss = tf.Print(loss, [y_pred[..., 4]], 'The value of prediction conf:', summarize=10)

        flattened = tf.reshape(y_pred[..., 4], [-1])
        unique_values, _ = tf.unique(flattened)
        loss = tf.Print(loss, [unique_values], 'Unique value of conf', summarize=100)

        #prediction value
        loss = tf.Print(loss, [masked_true], 'True mas', summarize=15)
        loss = tf.Print(loss, [tf.sigmoid(masked_pred)], 'Pred mask', summarize=15)
        loss = tf.Print(loss, [tf.sigmoid(neg_masked_pred)], 'No mask', summarize=15)
        
    return loss
