
from preprocessing import CorpusProcessor
from preprocessing import read_txt
from preprocessing import create_train_test_ds
import time

t1 = time.time()

num_lines = None
batch_size = 256

eng_sentences , marathi_sentences = read_txt( 'mar.txt' , num_lines )

eng_processor = CorpusProcessor( eng_sentences , lang='eng' )
marathi_processor = CorpusProcessor( marathi_sentences , lang='mar' )

eng_sentences = eng_processor.texts_to_sequences( eng_sentences )
marathi_sentences = marathi_processor.texts_to_sequences( marathi_sentences )

print( 'Sample input {}'.format( eng_sentences[0] ) )
print( 'Sample output {}'.format( marathi_sentences[0] ) )

print( 'input maxlen {} and vocab size {}'.format( eng_processor.max_len , len( eng_processor.vocab) ))
print( 'output maxlen {} and vocab size {}'.format( marathi_processor.max_len , len( marathi_processor.vocab) ))

train_ds , test_ds = create_train_test_ds( eng_sentences , marathi_sentences )
train_ds = train_ds.batch( batch_size )

print( 'Processing completed in {} secs'.format( (time.time() - t1) ) )
