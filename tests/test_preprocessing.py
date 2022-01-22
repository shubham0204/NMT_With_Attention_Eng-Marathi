
from preprocessing import CorpusProcessor , load_corpus_processor
import pandas as pd

sentences = pd.read_csv('../mar.txt', sep='\t', encoding='utf8', header=None).sample(frac=1.).values
eng_sentences = sentences[ : , 0 ]
marathi_sentences = sentences[ : , 1 ]

eng_processor = CorpusProcessor( eng_sentences , lang='eng' )
marathi_processor = CorpusProcessor( marathi_sentences , lang='mar' )

out = eng_processor.texts_to_sequences( eng_sentences )
print( out )
out = marathi_processor.texts_to_sequences( marathi_sentences )
print( out )