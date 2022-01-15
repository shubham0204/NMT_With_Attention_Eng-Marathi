
from preprocessing import CorpusProcessor
from layers import build_decoder , build_encoder
from model_config import write_config
from losses import sparse_categorical_ce
from losses import get_padding_mask
from preprocessing import read_txt
from preprocessing import create_train_test_ds
import tensorflow as tf
import argparse
import os
import wandb

parser = argparse.ArgumentParser( 'app_description' )
parser.add_argument( '--num_lines' , type=int , nargs='?' , const=10000 , default=10000 )
parser.add_argument( '--batch_size', type=int, nargs='?' , const=32, default=32 )
parser.add_argument( '--epochs', type=int, nargs='?' , const=100, default=100 )
parser.add_argument( '--model_dir' , type=str , nargs='?' , const='models/', default='models/' )
parser.add_argument( '--enc_embedding_dims', nargs='?' , type=int , const=128, default=128 )
parser.add_argument( '--dec_embedding_dims', nargs='?' , type=int , const=128, default=128 )
parser.add_argument( '--enc_units', type=int , nargs='?' , const=128, default=128 )
parser.add_argument( '--dec_units', type=int , nargs='?' , const=128, default=128 )

args = parser.parse_args()
num_lines = args.num_lines
batch_size = args.batch_size
epochs = args.epochs
model_dir = args.model_dir
enc_embedding_dims = args.enc_embedding_dims
dec_embedding_dims = args.dec_embedding_dims
enc_units = args.enc_units
dec_units = args.dec_units

#wandb.init( project='my-test-project' )
#wandb.config.update( args )

if not os.path.exists( model_dir ):
    os.mkdir( model_dir )
    os.mkdir( os.path.join( model_dir , 'weights' ) )
    os.mkdir( os.path.join( model_dir , 'config' ) )

eng_sentences , marathi_sentences = read_txt( 'tests/mar.txt' , num_lines )

eng_processor = CorpusProcessor( eng_sentences , lang='eng' )
eng_vocab_size = len( eng_processor.vocab )
marathi_processor = CorpusProcessor( marathi_sentences , lang='mar' )
marathi_vocab_size = len( marathi_processor.vocab )

eng_sentences = eng_processor.texts_to_sequences( eng_sentences )
marathi_sentences = marathi_processor.texts_to_sequences( marathi_sentences )

encoder = build_encoder( enc_embedding_dims , enc_units , eng_vocab_size , eng_processor.max_len )
decoder = build_decoder( dec_embedding_dims , dec_units , marathi_vocab_size , eng_processor.max_len )
write_config( model_dir , 'encoder_config.json' , enc_embedding_dims , eng_processor.max_len ,
              eng_vocab_size , enc_units )
write_config( model_dir , 'decoder_config.json' , dec_embedding_dims , eng_processor.max_len ,
              marathi_vocab_size , dec_units )

train_ds , test_ds = create_train_test_ds( eng_sentences , marathi_sentences )
train_ds = train_ds.batch( batch_size , drop_remainder=True ).repeat( epochs )

optimizer = tf.keras.optimizers.Adam( learning_rate=0.01 )

def forward_pass( batch_inputs , batch_outputs , output_mask ):
    batch_loss = 0.0
    enc_outputs, enc_hidden_state = encoder(batch_inputs)
    dec_hidden_state = enc_hidden_state
    dec_input = tf.expand_dims( [eng_processor.word2index[eng_processor.START_TAG]] * batch_size, axis=1)
    for t in range(1, marathi_processor.max_len):
        predictions, dec_hidden_state = decoder( [dec_input, dec_hidden_state, enc_outputs] )
        predictions = tf.squeeze( predictions , axis=1 )

        batch_loss += sparse_categorical_ce(predictions, batch_outputs[ : , t : t+1 ] )
    return batch_loss / tf.reduce_sum( tf.cast( output_mask , dtype=tf.float32 ) )


@tf.function
def train_step( inputs , outputs ):
    batch_loss = 0.0
    with tf.GradientTape() as tape:
        output_mask = get_padding_mask( outputs )
        batch_loss = forward_pass( inputs , outputs , output_mask )
    enc_grads, dec_grads = tape.gradient(batch_loss, [encoder.trainable_weights, decoder.trainable_weights])
    optimizer.apply_gradients(zip(enc_grads, encoder.trainable_weights))
    optimizer.apply_gradients(zip(dec_grads, decoder.trainable_weights))
    batch_loss = (batch_loss / batch_size)
    return batch_loss



for e in range( epochs ):
    print('Epoch {} -----------------------------------------'.format(e + 1))
    epoch_min_loss = 0.0

    for ( step , ( inputs , outputs ) ) in enumerate( train_ds ):
        outputs = tf.cast( outputs , tf.float32 )
        batch_loss = train_step( inputs , outputs )
        epoch_min_loss = batch_loss
        # wandb.log( { 'loss' : batch_loss } )
        print( 'step {} --------------------------------- loss={}'.format( step + 1 , batch_loss ) )

    if e % 5 == 0:
        encoder.save_weights( os.path.join( model_dir , 'weights/encoder_{}_{}'.format( e+1 , epoch_min_loss ) ) )
        decoder.save_weights( os.path.join( model_dir , 'weights/decoder_{}_{}'.format( e+1 , epoch_min_loss ) ))
        print( 'Model weights saved.' )



