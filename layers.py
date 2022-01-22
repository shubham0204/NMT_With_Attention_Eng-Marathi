
from tensorflow.keras.layers import GRU , Embedding , Dense , Input
from attention import DotProductAttention
import tensorflow as tf

"""
An encoder model which transforms the input sequences into a intermediate representation.
The model takes a fixed length sequence of shape ( max_len , ) and produces two outputs,
-> enc_hidden_state: The hidden state of the GRU of shape ( enc_units , )
-> enc_outputs: The outputs of the GRU cell at each timestep, shape ( timesteps , enc_units )
"""
class Encoder( tf.keras.Model ):

    def __init__( self , embedding_dim , enc_units , vocab_size ):
        super(Encoder, self).__init__()
        # An embedding layer is used to convert indices to dense vectors.
        self.embedding = Embedding( vocab_size + 1 , embedding_dim  )
        # A Gated Recurrent Unit ( GRU )
        self.gru = GRU( units=enc_units ,
                        return_sequences=True ,
                        return_state=True ,
                        recurrent_initializer='glorot_uniform' )

    # Perform a forward pass for the encoder
    def call( self , x ):
        x = self.embedding( x )
        outputs , hidden_states = self.gru( x )
        return outputs , hidden_states


class Decoder( tf.keras.Model ):

    def __init__( self , embedding_dim , dec_units , vocab_size ):
        super(Decoder, self).__init__()
        self.embedding = Embedding(vocab_size + 1, embedding_dim )
        self.gru = GRU(units=dec_units,
                       return_sequences=True,
                       return_state=True,
                       recurrent_initializer='glorot_uniform')
        self.attention = DotProductAttention()
        self.linear = Dense( vocab_size , activation='softmax' )


    def call(self , x , hidden_state , enc_outputs ):
        # enc_outputs -> ( None , timesteps , enc_units )
        # hidden_state -> ( None , dec_units )
        # x -> ()
        context_vector = self.attention( enc_outputs , hidden_state )
        x = self.embedding(x)
        x = tf.concat( [ context_vector , x ] , axis=-1 )
        output , state = self.gru( x )

        x = self.linear( output )
        return x , state


# Building the encoder and decoder model with well-defined input shapes and outputs
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