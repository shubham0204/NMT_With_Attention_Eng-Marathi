
from layers import Encoder , Decoder , build_decoder
import numpy as np

batch_size = 32
input_max_len = 8
vocab_size = 1000

inputs = np.random.randint( 1 , vocab_size + 1 , size=( batch_size , input_max_len ) )
encoder = Encoder( 16 , 16 , vocab_size +1  )
decoder = build_decoder( 16 , 16 , vocab_size , input_max_len )
print( decoder.input , decoder.output )

enc_outputs , enc_hidden_state = encoder( inputs )
print( enc_outputs.shape )
print( enc_hidden_state.shape )

dec_hidden_state = enc_hidden_state
dec_input = np.expand_dims( [ 0 ] * batch_size , 1 )
for t in range( 1 , input_max_len ):
    dec_predictions , dec_hidden_state = decoder( [dec_input , dec_hidden_state , enc_outputs] )
    print( dec_predictions.shape )
    print( dec_hidden_state.shape )

