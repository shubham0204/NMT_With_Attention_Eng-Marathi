
import tensorflow as tf

class DotProductAttention( tf.keras.layers.Layer ):

    def __call__( self , enc_outputs , dec_hidden_state ):
        attention_scores = tf.matmul( enc_outputs , dec_hidden_state )
        alignment_scores = tf.nn.softmax( attention_scores )
        context_vector = tf.reduce_sum( enc_outputs * alignment_scores , axis=1 )
        return context_vector
