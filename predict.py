

from model_config import read_config
from layers import build_decoder , build_encoder
import tensorflow as tf
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




