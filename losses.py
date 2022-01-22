
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import tensorflow as tf

def get_padding_mask(x):
    return tf.cast( tf.not_equal( x , 0 ) , tf.float32 )

def sparse_categorical_ce( predictions , targets ):
    padding_mask = get_padding_mask( targets )
    loss = SparseCategoricalCrossentropy()( targets , predictions )
    loss = loss * padding_mask
    return tf.reduce_sum( loss )


