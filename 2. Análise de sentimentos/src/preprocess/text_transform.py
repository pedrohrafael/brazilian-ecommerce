import os
import re
import spacy
import demoji
from string import punctuation
from unidecode import unidecode
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
    
LANGS_DOWNLOAD = {'pt':'pt_core_news_sm', 
                  'en':'en_core_web_sm', 
                  'es':'es_core_news_sm'}

class TextCleaning(BaseEstimator, TransformerMixin):
        
    def __init__(self, transformers):
        """
        """
        self.transformers = transformers
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        for name, transformer in self.transformers.items():
            X = transformer(X)
        return X    
    
class Lemmatizer(BaseEstimator, TransformerMixin):
    
    def __init__(self, language='pt'):
        """
        language : str, Default 'pt'
            Options: pt, en, es
        """
        self.language = language
        package = LANGS_DOWNLOAD[self.language]
        # os.system(f"python -m spacy download {package} --silent")
        self.nlp = spacy.load(f"{package}")
        
    def fit(self, X, y=None):
        return self    
    
    def transform(self, X):
        return [' '.join(i.lemma_ for i in self.nlp(string)) for string in X]

class PartOfSpeechTagFilter(BaseEstimator, TransformerMixin):

    def __init__(self, tags, language='pt'):
        """
        Params
        --------        
        tags : list (optional)
            Tags list to filter sentences.
            Options: ADJ, ADP, ADV, AUX, CONJ, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X, SPACE
        language : str, Default 'pt'
            Options: pt, en, es
        """
        self.language = language
        self.tags = tags
        package = LANGS_DOWNLOAD[self.language]
        # os.system(f"python -m spacy download {package} --silent")        
        self.nlp = spacy.load(f"{package}")
        
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = [' '.join(str(i) for i in self.nlp(string) if i.pos_ in self.tags) for string in X]
        return X

def remove_accents(X):
    return [unidecode(string) for string in X]

def remove_digits(X):
    return [''.join([i for i in string if not i.isdigit()]) for string in X]

def remove_punctuation(X):
    return [''.join(i for i in string.strip() if i not in set(punctuation)) for string in X]

def remove_stopwords(X):
    return [' '.join([i for i in string.split() if i not in stopwords.words('portuguese')]) for string in X]
    
def remove_emoji(X):
    return [demoji.replace(string, '') for string in X]

def remove_hiperlink(X):
    return [re.sub('http[s]?:[/]?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', string) for string in X]

def replace_negation(X):
    return [' '.join(re.sub('([nN][ãÃaA][oO]|[ñÑ]|[nN] )', ' negativa ', string).split()) for string in X]

def lowercase(X):
    return [string.lower() for string in X]