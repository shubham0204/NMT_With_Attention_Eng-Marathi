

from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import re
import pickle


"""
Helper class to process the given language corpus.
We perform various operations: 
1. text cleaning
2. tokenization
3. transforming text tokens -> integers
4. padding of integer sequences
"""
class CorpusProcessor:

    """
    Args:
        corpus: list[str]:
        lang: str ( can be 'eng' or 'mar' ): The language of the corpus. This is required as English and Marathi
        sentences have different processing methods
    """
    def __init__( self , corpus , lang ):
        self.START_TAG , self.START_TAG_INDEX = '<start>' , 1
        self.END_TAG , self.END_TAG_INDEX = '<end>' , 2
        self.PADDING = '<pad>'
        self.lang = lang
        self.max_len = None
        self.vocab = [ self.PADDING , self.START_TAG , self.END_TAG ]
        self.word2index = { self.PADDING : 0 , self.START_TAG : 1 , self.END_TAG : 2  }
        self.index2word = { 0 : self.PADDING , 1 : self.START_TAG , 2 : self.END_TAG }
        self.word2index , self.index2word = self.build_vocab( corpus )

    def texts_to_sequences( self , eng_sentences , add_tags=True ):
        out = []
        for sent in eng_sentences:
            if add_tags:
                out.append( self.tokens_to_indices( [ self.START_TAG ] + self.process( sent ) + [ self.END_TAG ] ) )
            else:
                out.append( self.tokens_to_indices( self.process( sent ) ) )
        if self.max_len is None:
            self.max_len = max( len(s) for s in out )
        out = pad_sequences( out , maxlen=self.max_len , padding='post' )
        return out

    def merge_two_dicts(self, x, y):
        z = x.copy()
        z.update(y)
        return z

    def build_vocab( self , corpus ):
        sent_tokens = self.tokenize(corpus)
        flattened_sent_tokens = [ token for sent in sent_tokens for token in sent ]
        vocab = sorted( list( set( flattened_sent_tokens ) ) )
        self.vocab += vocab
        indices = range( 3 , len( vocab ) + 3 )

        self.word2index = self.merge_two_dicts( self.word2index , {word: index for word, index in zip( vocab, indices)} )
        self.index2word = self.merge_two_dicts( self.index2word , {index: word for index, word in zip(indices, vocab)} )
        return self.word2index , self.index2word

    def tokenize( self , sentences ):
        sent_tokens = []
        for sent in sentences:
            sent_tokens.append( self.process( sent ) )
        return sent_tokens

    def save(self, filename):
        with open(filename, 'wb') as fp:
            pickle.dump(self, fp)

    def process( self , sentence ):
        if self.lang == 'eng':
            sentence = sentence.strip().lower()
            sentence = re.sub( r'[^A-Za-z ]+' , '' , sentence )
        else:
            sentence = sentence.strip()
            sentence = re.sub(r'[२३०८१५७९४६!.?,;]', '', sentence)
        return sentence.split()

    def tokens_to_indices( self , sent_tokens ):
        try:
            return [ self.word2index[ token] for token in sent_tokens if token in self.word2index ]
        except KeyError:
            return



    def indices_to_tokens( self , sent_indices ):
        return [ self.index2word[ index ] for index in sent_indices ]


def load_corpus_processor( filename):
    with open(filename, 'rb') as fp:
        return pickle.load(fp)

def read_txt( filename , num_lines=None ):
    sentences = pd.read_csv( filename , header=None , encoding='utf8' , sep='\t' ).sample( frac=1. ).values
    if num_lines == None:
        return sentences[ : , 0 ] , sentences[ : , 1 ]
    else:
        return sentences[ 0 : num_lines , 0 ] , sentences[ 0 : num_lines , 1 ]

def create_train_test_ds( eng_sentences , marathi_sentences , test_frac=0.3 ):
    ds = tf.data.Dataset.from_tensor_slices( ( eng_sentences , marathi_sentences ) )
    num_test_samples = int( ds.cardinality().numpy() * test_frac )
    test_ds = ds.take( num_test_samples )
    train_ds = ds.skip( num_test_samples )
    return train_ds , test_ds
