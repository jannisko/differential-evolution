import tensorflow as tf

def create_huber(model, input_vals_y):
    def huber(delta=14.0):
        # huber loss
        diff = tf.abs(input_vals_y - model())
        loss_list =  tf.where(
            diff < delta, 
            0.5 * tf.square(diff),
            delta * diff - 0.5 * tf.square(delta)
        )
        return tf.reduce_mean(loss_list)
    return huber
        
def create_mse(model, input_vals_y):
    def mse():
        return tf.reduce_mean(tf.square(input_vals_y - model()))
    return mse

