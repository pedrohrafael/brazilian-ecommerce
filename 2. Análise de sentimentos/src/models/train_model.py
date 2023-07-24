# Pip
import os
import joblib
import numpy as np
import pandas as pd
import logging as log
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Project
from ml_logwriter.logger import LogArtifacts, LogWriter
from dataset.make_dataset import load_reviews
from preprocess.text_transform import TextCleaning, Lemmatizer, PartOfSpeechTagFilter
from preprocess.text_transform import remove_hiperlink, remove_emoji, remove_digits, remove_punctuation, \
                                      remove_accents, remove_stopwords, lowercase, replace_negation

# Global variables
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'raw')
CSV_PATH = os.path.join(DATA_DIR, "olist_order_reviews_dataset.csv")
ARTIF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'models')
FILE_LOGS = 'file_logs.txt'
SEED = 42

# start logArtifacts
logAtfcs = LogArtifacts(ARTIF_DIR)
logAtfcs.create()

# start logWriter
log = LogWriter().logger(os.path.join(logAtfcs.artifacts_path, FILE_LOGS))

log.info("Iniciando treino do modelo.")
log.info(f"Armazenando Logs e Artefatos em: {logAtfcs.name}.")

try:
    log.info("Carregando dados.")
    reviews = load_reviews(CSV_PATH)
except Exception as e:
    log.error(f"Falha ao carregar dados - {e}")
    sys.exit('Encerrando.')

try:    
    log.info("Dividindo dados de treino e teste.")
    X = np.asarray(reviews.review_comments)
    y = np.asarray(reviews.target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=SEED)
except Exception as e:
    log.error(f"Falha ao dividir os dados - {e}")
    sys.exit('Encerrando.')

try:
    log.info(f"Armazenando datasets de treino e teste.")
    dataset_train = pd.concat([pd.Series(X_train), pd.Series(y_train)], axis=1)
    dataset_test = pd.concat([pd.Series(X_test), pd.Series(y_test)], axis=1)
    logAtfcs.log_dataset(dataset_train, 'dataset_train')
    logAtfcs.log_dataset(dataset_test, 'dataset_test')
except Exception as e:
    log.error(f"Falha ao armazenar os datasets - {e}")
    
text_transformers = {'remove_hiperlink':remove_hiperlink,
                     'remove_emoji':remove_emoji,
                     'remove_digits':remove_digits,
                     'remove_punctuation':remove_punctuation,
                     'remove_accents':remove_accents,
                     'remove_stopwords':remove_stopwords,
                     'lowercase':lowercase,
                     'replace_negation':replace_negation}

try:
    log.info(f"Construindo pipeline.")
    pipeline = Pipeline([
        ('TextPreprocessing', Pipeline([
            ('TextCleaning', TextCleaning(text_transformers)),
            ('Lemmatizer', Lemmatizer()),
            ('PartOfSpeechTagFilter', PartOfSpeechTagFilter(tags=['VERB', 'ADJ', 'ADV', 'NOUN', 'PRON'])),
            ('FeatureExtraction', TfidfVectorizer())
        ])),
        ('Model', LogisticRegression())
    ])
except Exception as e:
    log.error(f"Falha ao construir o pipeline - {e}")
    sys.exit('Encerrando.')

params = {
    'TextPreprocessing__FeatureExtraction__max_features' : [250, 500, 750, 1000],
    'TextPreprocessing__FeatureExtraction__min_df' : [4, 8, 12],
    'TextPreprocessing__FeatureExtraction__max_df' : [0.8, 0.85, 0.90],
    'Model__solver' : ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag'],
    'Model__class_weight' : ['balanced', None],
    'Model__random_state' : [SEED]
}

try:
    log.info(f"Ajustando modelo com melhores hiperparametros [RandomizedSearchCV].")
    RSCV = RandomizedSearchCV(pipeline, params, scoring='accuracy', n_jobs= 1)
    RSCV.fit(X_train, y_train)
except Exception as e:
    log.error(f"Falha ao ajustar modelo - {e}")
    sys.exit('Encerrando.')
    
try:
    log.info(f"Armazenando modelo com melhores hiperparametros.")
    model = RSCV.best_estimator_
    logAtfcs.log_model(model)
    logAtfcs.log_parameters(model.get_params())
except Exception as e:
    log.error(f"Falha ao armazenar modelo - {e}")
    sys.exit('Encerrando.')
try:
    log.info(f"Avaliando modelo.")
    y_test_pred = model.predict(X_test)
    metrics = {
        "accuracy_score" : accuracy_score(y_test_pred, y_test),
        "precision_score" : precision_score(y_test_pred, y_test),
        "recall_score" : recall_score(y_test_pred, y_test),
        "f1_score" : f1_score(y_test_pred, y_test)
    }
    logAtfcs.log_metrics(metrics)
except Exception as e:
    log.error(f"Falha ao avaliar modelo - {e}")
    sys.exit('Encerrando.')
    
log.info(f"Modelo treinado com SUCESSO!")