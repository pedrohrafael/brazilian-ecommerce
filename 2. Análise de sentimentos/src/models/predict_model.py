import os
import joblib
import numpy as np

class AnalyzeSentiment:
    
    def __init__(self, filename):
        self.__model = self.__load_model(filename)

    def predict(self, string):
        pred = self.__model.predict([string])
        proba = self.__model.predict_proba([string]).max()
        sentiment = 'Positivo' if pred==1 else 'Negativo'
        result = {'string' : string, 'sentiment' : sentiment, 'proba' : proba}        
        return result
    
    def __load_model(self, filename):
        return joblib.load(filename)