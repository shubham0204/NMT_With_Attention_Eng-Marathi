
from tensorflow.keras.layers import GRU , Embedding , Dense , Input
import tensorflow as tf

class Encoder( tf.keras.Model ):

    def __init__( self , embedding_dim , enc_units , vocab_size ):
        super(Encoder, self).__init__()
        self.embedding = Embedding( vocab_size + 1 , embedding_dim  )
        self.gru = GRU( units=enc_units ,
                        return_sequences=True ,
                        return_state=True ,
                        recurrent_initializer='glorot_uniform' )

    def call( self , x ):
        x = self.embedding( x )
        outputs , hidden_states = self.gru( x )
        return outputs , hidden_states


class Decoder( tf.keras.Model ):

    def __init__( self , embedding_dim , dec_units , vocab_size ):
        super(Decoder, self).__init__()
        self.embedding = Embedding(vocab_size + 1, embedding_dim)
        self.gru = GRU(units=dec_units,
                       return_sequences=True,
                       return_state=True,
                       recurrent_initializer='glorot_uniform')
        self.linear = Dense( vocab_size )


    def call(self , x , hidden_state , enc_outputs ):
        # enc_outputs -> ( None , timesteps , enc_units )
        # hidden_state -> ( None , dec_units )
        # x -> ()
        alignment_vector = tf.matmul( hidden_state , enc_outputs , transpose_b=True )
        alignment_scores = tf.nn.softmax( alignment_vector )
        context_vector = tf.reduce_sum( alignment_vector * alignment_scores , axis=1 )
        context_vector = tf.expand_dims( context_vector , axis=1 )
        x = self.embedding(x)
        x = tf.concat( [ context_vector , x ] , axis=-1 )
        output , state = self.gru( x )

        x = self.linear( output )
        return x , state

def build_encoder( embedding_dims , enc_units , vocab_size , input_max_len ):
    inputs = Input( shape=( input_max_len , ) )
    outputs , hidden_states = Encoder( embedding_dims , enc_units , vocab_size )( inputs )
    return tf.keras.Model( inputs , [ outputs , hidden_states ] )

def build_decoder( embedding_dims , dec_units , vocab_size , input_max_len ):
    decoder_inputs = Input( shape=( 1 , ) )
    decoder_hidden_state = Input( shape=( dec_units , ) )
    decoder_enc_outputs = Input( shape=( input_max_len , dec_units ) )
    outputs , hidden_state = Decoder( embedding_dims , dec_units , vocab_size )( decoder_inputs , decoder_hidden_state , decoder_enc_outputs )
    return tf.keras.Model( [ decoder_inputs , decoder_hidden_state , decoder_enc_outputs ] , [outputs,hidden_state] )