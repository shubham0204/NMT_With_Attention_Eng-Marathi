
import tensorflow as tf

class DotProductAttention( tf.keras.layers.Layer ):

    def __call__( self , enc_outputs , dec_hidden_state ):
        alignment_vector = tf.matmul( dec_hidden_state, enc_outputs, transpose_b=True)
        alignment_scores = tf.nn.softmax(alignment_vector)
        context_vector = tf.reduce_sum(alignment_vector * alignment_scores, axis=1)
        context_vector = tf.expand_dims(context_vector, axis=1)
        return context_vector
