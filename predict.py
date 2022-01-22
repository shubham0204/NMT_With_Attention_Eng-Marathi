

from model_config import read_config
from layers import build_decoder , build_encoder
from preprocessing import load_corpus_processor
import tensorflow as tf
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser( 'Python script to train the NMT model.' )
parser.add_argument( '--config_path' , type=str )
args = parser.parse_args()

config = read_config( args.config_path )
enc_latest_model_path = tf.train.latest_checkpoint(
    os.path.join( config[ 'model_dir' ] , 'weights' , config[ 'run_name'] , 'encoder' ))
dec_latest_model_path = tf.train.latest_checkpoint(
    os.path.join( config[ 'model_dir' ] , 'weights' , config[ 'run_name'] , 'decoder' ))

encoder = build_encoder(
    config[ 'enc_embedding_dims' ] ,
    config[ 'enc_units' ] ,
    config[ 'eng_vocab_size' ] ,
    config[ 'eng_max_len' ]
)
decoder = build_decoder(
    config[ 'dec_embedding_dims' ] ,
    config[ 'dec_units' ] ,
    config[ 'marathi_vocab_size' ] ,
    config[ 'eng_max_len' ]
)

encoder.load_weights( enc_latest_model_path )
decoder.load_weights( dec_latest_model_path )

eng_processor = load_corpus_processor( config[ 'eng_processor_path' ] )
marathi_processor = load_corpus_processor( config[ 'marathi_processor_path' ] )

input_sent = input( 'Enter sentence :' )
input_sent = eng_processor.texts_to_sequences( [ input_sent ] )

enc_outputs, enc_hidden_state = encoder( np.array( input_sent ) )
dec_input = np.array( [ [ 1.0 ] ] )
dec_hidden_state = enc_hidden_state
for t in range(1, marathi_processor.max_len):
    predictions, dec_hidden_state = decoder( [dec_input, dec_hidden_state, enc_outputs] )
    word = marathi_processor.index2word[ np.argmax( predictions ) ]
    print( word )
    dec_input = np.array( [ [ np.argmax( predictions ) ] ] )
