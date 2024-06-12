# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 20:18:38 2024

@author: Nicolas, Marwan, Doriane, Aurore, Quentin
Project Explain
"""

#%% Imports
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import re
from cleantext import clean 
from nltk.tokenize import (
    TweetTokenizer  
)
from nltk.stem import WordNetLemmatizer 
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from wordfreq import word_frequency

import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

#%% Chargement de la base de données avec tri et nettoyage

try:
    # Lecture du fichier CSV
    data = pd.read_csv("C:\\Users\\nicol\\OneDrive\\Bureau\\Documents\\L3\\s6\\Master Camp\\Project-Explain\\EFREI - LIPSTIP - 50k elements EPO.csv")
    #data = pd.read_csv(r"C:\Users\nicol\OneDrive\Bureau\Documents\L3\s6\Master Camp\Project Explain\base_de_donnees.csv")
    
    # Afficher les premières lignes du DataFrame pour vérifier la lecture
    print(data.head(5))
    print(data.shape)
    print(data.info())
except Exception as e:
    # En cas d'erreur, afficher un message et l'erreur
    print("Erreur lors de la lecture du fichier CSV :", e)
    data = "error..."

#%% Traitement des données
#Suppression des doublons
print("Taille avant :",data.shape)
data = data.drop_duplicates()
print("Taille après :",data.shape)

#identification de possibles valeurs manquantes
print(data.isnull().all())

#Retirer les 4 premières colonnes
cols_to_mantain = ['CPC', 'description']
data = data[cols_to_mantain]
print(data.shape)

#%%¨Nettoyage des données (balises):
    
# Combined regex for faster processing
combined_regex = re.compile(
    r'<!--.*?-->|<[^>]+>|\(([0-9]*[a-z]*|[ivxlcdm]*)\)|\(\s*(\d+[a-z]*)(\s*,\s*\d+[a-z]*)*(\s*to\s*\d+[a-z]*)?\s*\)|\d+[a-z]*|[^a-zA-Z0-9\s<>]'
)

# Function to clean descriptions
def clean_description(desc):
    desc = clean(desc, lower=True, no_currency_symbols=True, replace_with_currency_symbol="<CUR>", lang="en")
    desc = combined_regex.sub(' ', desc)
    return desc
    

#Test sur les 4 premières ligens car c'est bcp trop long sur les 50000
data.loc[:4, 'description'] = data.loc[:4, 'description'].apply(clean_description)

print(data['description'][3])

#%%¨Tokenize :
vocab = set()
tokenizer = TweetTokenizer()

for _, rev in data.loc[:4, 'description'].items():
    vocab.update(tokenizer.tokenize(rev))
print(len(vocab))

# for 1 description
desc = set()
tokenizer = TweetTokenizer()

for _, rev in data.loc[:1, 'description'].items():
    desc.update(tokenizer.tokenize(rev))
print(len(desc))

vocab = desc -set(stopwords.words('english')) 
print(len(desc))


vocab = vocab -set(stopwords.words('english')) 
print(len(vocab))

#%% Stemming (enlève la terminaison) and Lemmatization (renvoie à la forme radicale):
stemmer = PorterStemmer() #shorter

stem_vocab = [stemmer.stem(word) for word in vocab]
print(stem_vocab)    

stem_vocab = set(stem_vocab)
print(len(stem_vocab))

# Stemming d'une description
stem_desc = [stemmer.stem(word) for word in desc]
print(stem_desc)    

stem_desc = set(stem_desc)
print(len(stem_desc))
"""
lemmatizer = WordNetLemmatizer()
lemmatizer_vocab = [lemmatizer.lemmatize(word) for word in vocab]
print(lemmatizer_vocab)

lemmatizer_vocab = set(lemmatizer_vocab)
print(len(lemmatizer_vocab))
"""

#%% Vectorisation
vocabulary = dict()
vocab_index=0

for word in stem_vocab:
    if word not in vocabulary:
        vocabulary[word] = vocab_index
        vocab_index +=1
print(vocabulary)



def vectorize_with_freq(text,vocab):
    vect = np.zeros(len(vocab))
    cpt=0
    for word in text:
        vect[vocab[word]] +=1
        cpt +=1
    vect /=cpt
    return vect
print(vectorize_with_freq(stem_desc,vocabulary))

#%% Algorithme supervisé : Classification

#%% Main
#if __name__=='__main__':

