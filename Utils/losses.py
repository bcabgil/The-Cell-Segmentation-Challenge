from tensorflow.keras import backend as K
import tensorflow as tf


def weighted_bce_loss(weight_map, weight_strength):
    def weighted_bce(y_true, y_pred):
        weight_f = weight_map * weight_strength +1.
        wy_true_f = weight_f * y_true
        wy_pred_f = weight_f * y_pred
        return K.mean(K.binary_crossentropy(wy_true_f, wy_pred_f))
    return weighted_bce


def jaccard_loss(y_true, y_pred, smooth=100):
  #   Calculates mean of Jaccard distance as a loss function 
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1,2))
    sum_ = tf.reduce_sum(y_true + y_pred, axis=(1,2))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    jd =  (1-jac) * smooth
    return tf.reduce_mean(jd)

def jaccard_acc(y_true, y_pred, smooth=100):
  #   Calculates mean of Jaccard distance as a loss function 

    intersection = tf.reduce_sum(y_true * y_pred, axis=(1,2))
    sum_ = tf.reduce_sum(y_true + y_pred, axis=(1,2))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    jd =  (jac) * smooth
    return tf.reduce_mean(jd)

