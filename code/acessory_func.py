# %%
import numpy as np
import pandas as pd

import re
import string
import html

import nltk
nltk.download('all')
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

# %%
def remove_url(text):
    '''Remove URLs, HTML tags, and HTML Characters Entities from text.'''
    
    # Remove URLs
    text = text.apply(lambda x: re.sub(r'http\S+|www.\S+', '', x))

    # Remove HTML tags
    text = text.apply(lambda x: re.sub(r'<.*?>', '', x))

    # Remove HTML Characters Entities
    text = text.apply(lambda x: html.unescape(x).encode('ascii', 'ignore').decode('ascii', 'ignore'))

    return text


# %%
def cleaning(text, stopwords_to_exclude=None, punc_to_exclude=None):
    '''Convert text to lower case and remove numbers, punctuations, and stop words.
    Args:
        text: pd.Series
        stopwords_to_exclude: stop word or list of stop words to exclude from NLTK stop words
                              default is None
        punc_to_exclude: puntuation or list of puntuations to exclude from string.punctuation
                         default is None'''
    
    # Remove numbers and words containing numbers
    text = text.apply(lambda x: re.sub('\w*\d\w*', '', x))

    # Remove punctuations from the text
    if punc_to_exclude is not None:
        punc = string.punctuation
        for i in punc_to_exclude:
            punc = punc.replace(i, '')
        text = text.apply(lambda x: re.sub('[%s]' % re.escape(punc), ' ', x.lower()))
    else:
        text = text.apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower()))

    # Remove stop words from the text
    stop_words = set(stopwords.words('english'))
    if stopwords_to_exclude is not None:
        stop_words = stop_words - set(stopwords_to_exclude)
        text = text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    else:
        text = text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    return text

# %%
def n_grams_word_count(text, ngrams):
    '''Get the frequency count of n-grams words in the text.
    Args:
        text: pd.Series
        ngrams: int, the n-grams words to count'''
    
    vec = CountVectorizer(ngram_range=(ngrams, ngrams))
    word_cnt = np.ravel(vec.fit_transform(text).sum(axis=0))
    
    return pd.DataFrame([(word_cnt[i], k) for k, i in vec.vocabulary_.items()], columns=['Count', 'Words'])

# %%
# Create a fuction to get the POS tag to use it in lemmatization
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return "a"
    elif treebank_tag.startswith('V'):
        return "v"
    elif treebank_tag.startswith('N'):
        return "n"
    elif treebank_tag.startswith('R'):
        return "r"
    else:
        return "n"

# %%
def preprocessing(text, lemma=False):
    '''Tokenize the text. If lemma is True, then POS tagging and lemmatization will be performed.
    Args:
        text: pd.Series
        lemma: bool, default is False
               If True, then POS tagging and lemmatization will be performed. If False, then only tokenization will be performed.'''
    # Tokenize the text
    text = text.str.split()

    # POS tagging
    if lemma:
        tokens_word_postag = [nltk.pos_tag(i) for i in text]
        pos_text = [[nltk.WordNetLemmatizer().lemmatize(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in i] for i in tokens_word_postag]
    else:
        pos_text = text
    
    return pos_text


