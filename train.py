
from preprocessing import CorpusProcessor
from preprocessing import read_txt
from preprocessing import create_train_test_ds
from losses import sparse_categorical_ce
from losses import get_padding_mask
from layers import build_decoder , build_encoder
from model_config import write_config
import tensorflow as tf
import argparse
import os
import wandb

tf.get_logger().setLevel( 2 )

parser = argparse.ArgumentParser( 'Python script to train the NMT model.' )
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

train_config = vars( args )

wandb.init( project='my-test-project' )
run_name = wandb.run.name

# Make sure that all directories exist, before saving model weights
if not os.path.exists( model_dir ):
    os.mkdir( model_dir )
os.makedirs( os.path.join( model_dir , 'weights' , run_name , 'encoder' ) , exist_ok=True )
os.makedirs( os.path.join( model_dir , 'weights' , run_name , 'decoder' ) , exist_ok=True )
os.makedirs( os.path.join( model_dir , 'config' , run_name ) , exist_ok=True )

# Read the (eng-marathi) sentences pairs from the text file.
eng_sentences , marathi_sentences = read_txt('mar.txt', num_lines)

# Process eng and marathi sentences also,
# Compute the vocabulary sizes for both languages
eng_processor = CorpusProcessor( eng_sentences , lang='eng' )
eng_vocab_size = len( eng_processor.vocab )
marathi_processor = CorpusProcessor( marathi_sentences , lang='mar' )
marathi_vocab_size = len( marathi_processor.vocab )

eng_sentences = eng_processor.texts_to_sequences( eng_sentences )
marathi_sentences = marathi_processor.texts_to_sequences( marathi_sentences )

train_config[ 'eng_vocab_size' ]  = eng_vocab_size
train_config[ 'marathi_vocab_size' ]  = marathi_vocab_size
train_config[ 'eng_max_len' ]  = eng_processor.max_len
train_config[ 'marathi_max_len' ]  = marathi_processor.max_len
train_config[ 'run_name' ] = run_name

eng_processor_path = os.path.join( model_dir , 'config' , run_name , 'eng_processor' )
marathi_processor_path = os.path.join( model_dir , 'config' , run_name , 'marathi_processor' )

train_config[ 'eng_processor_path' ] = eng_processor_path
train_config[ 'marathi_processor_path' ] = marathi_processor_path

eng_processor.save( eng_processor_path )
marathi_processor.save( marathi_processor_path )

write_config( os.path.join( model_dir , 'config' , run_name , 'config.json' ) , train_config )

# Build encoder and decoder models
# The decoder module contains the attention mechanism
encoder = build_encoder( enc_embedding_dims , enc_units , eng_vocab_size , eng_processor.max_len )
decoder = build_decoder( dec_embedding_dims , dec_units , marathi_vocab_size , eng_processor.max_len )

# Create tf.data.Dataset for the training examples
train_ds , test_ds = create_train_test_ds( eng_sentences , marathi_sentences )
train_ds = train_ds.batch( batch_size , drop_remainder=True ).prefetch( tf.data.AUTOTUNE )
test_ds = test_ds.batch( batch_size , drop_remainder=True ).prefetch( tf.data.AUTOTUNE )

# Adam optimizer for optimizing the parameters of the model
optimizer = tf.keras.optimizers.Adam( learning_rate=0.01 )

# A `tf.function` which performs a forward pass given an input and output batch of training data.
# It returns the mean loss ( cross entropy loss ) for the batch
@tf.function
def forward_pass( batch_inputs , batch_outputs , output_mask ):
    batch_loss = 0.0
    enc_outputs, enc_hidden_state = encoder(batch_inputs)
    dec_hidden_state = enc_hidden_state
    dec_input = tf.expand_dims( [eng_processor.word2index[eng_processor.START_TAG]] * batch_size, axis=1)
    for t in range(1, marathi_processor.max_len):
        predictions, dec_hidden_state = decoder( [dec_input, dec_hidden_state, enc_outputs] )
        predictions = tf.squeeze( predictions , axis=1 )
        dec_input = tf.expand_dims( batch_outputs[ : , t ] , 1 )
        batch_loss += sparse_categorical_ce(predictions, batch_outputs[ : , t : t+1 ] )
    return batch_loss / tf.reduce_sum( output_mask )


# A `tf.function` performing a training step ( forward pass + loss calculation + backpropagation ) on a batch
# of the training dataset.
@tf.function
def train_step( inputs , outputs ):
    with tf.GradientTape() as tape:
        output_mask = get_padding_mask( outputs )
        batch_loss = forward_pass( inputs , outputs , output_mask )
    enc_grads, dec_grads = tape.gradient( batch_loss, [encoder.trainable_weights, decoder.trainable_weights])
    optimizer.apply_gradients( zip(enc_grads, encoder.trainable_weights) )
    optimizer.apply_gradients( zip(dec_grads, decoder.trainable_weights) )
    return batch_loss

@tf.function
def test_step( test_ds ):
    test_batch_loss = 0.0
    for ( step , ( test_inputs , test_outputs ) ) in enumerate( test_ds ):
        test_outputs_mask = get_padding_mask( test_outputs )
        test_batch_loss += forward_pass( test_inputs , test_outputs , test_outputs_mask )
    return test_batch_loss / tf.cast( test_ds.cardinality() , tf.float32 )


# Start the training of the NMT model. Train the model for a certain number of `epochs`
# Log the loss values at each step and also save the model weights periodically.
for e in range( epochs ):
    print('Epoch {} -----------------------------------------'.format(e + 1))
    epoch_min_loss = 0.0

    for ( step , ( inputs , outputs ) ) in enumerate( train_ds ):
        outputs = tf.cast( outputs , tf.float32 )
        train_loss = train_step(inputs, outputs)
        val_loss = test_step( test_ds )
        epoch_min_loss = train_loss
        wandb.log({ 'loss' : train_loss , 'val_loss' : val_loss } )
        print( 'step {} --------------------------------- loss={} val_loss {}'.format(step + 1, train_loss, val_loss))

    if e % 5 == 0:
        encoder.save_weights( os.path.join( model_dir , 'weights' , run_name , 'encoder' , 'encoder_{}'.format( e+1 ) ))
        decoder.save_weights( os.path.join( model_dir , 'weights' , run_name , 'decoder' , 'decoder_{}'.format( e+1 ) ))
        print( 'Model weights saved.' )



