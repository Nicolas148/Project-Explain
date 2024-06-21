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
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from wordfreq import word_frequency
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Attention, Embedding, Layer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import ast



#%% Chargement de la base de données avec tri et nettoyage

try:
    # Lecture du fichier CSV
    #data = pd.read_csv("C:\\Users\\nicol\\OneDrive\\Bureau\\Documents\\L3\\s6\\Master Camp\\Project-Explain\\EFREI - LIPSTIP - 50k elements EPO.csv")
    data = pd.read_csv(r"C:\EXPLAIN\Project-Explain\EFREI - LIPSTIP - 50k elements EPO.csv")
    data = data.head(5000)
    
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
cols_to_mantain = ['CPC', 'claim']
data = data[cols_to_mantain]
print(data.shape)

#%%Nettoyage des données

#Nettoyage des claims (balises, chiffres, références):
    
# Regex combinée pour ne parcourir qu'une seule fois chaque description
combined_regex = re.compile(
    r'<!--.*?-->|<[^>]+>|\(([0-9]*[a-z]*|[ivxlcdm]*)\)|\(\s*(\d+[a-z]*)(\s*,\s*\d+[a-z]*)*(\s*to\s*\d+[a-z]*)?\s*\)|\d+[a-z]*|[^a-zA-Z0-9\s<>]'
)

# Function to clean claims
def clean_claims(claim):
    claim = clean(claim, lower=True, lang="en")
    claim = combined_regex.sub(' ', claim)
    return claim
    
data.loc[:, 'claim'] = data.loc[:, 'claim'].apply(clean_claims)


#Nettoyage des codes CPC pour 2 niveaux
data['CPC'] = data['CPC'].apply(ast.literal_eval)#Evalue la liste de code comme réellement une liste    
def get_unique_prefixes(code_list):
    prefixes = set(code[:3] for code in code_list)
    return list(prefixes)

data['CPC'] = data['CPC'].apply(get_unique_prefixes)
#%%¨Tokenize :
"""
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

lemmatizer = WordNetLemmatizer()
lemmatizer_vocab = [lemmatizer.lemmatize(word) for word in vocab]
print(lemmatizer_vocab)

lemmatizer_vocab = set(lemmatizer_vocab)
print(len(lemmatizer_vocab))

"""
#%% Vectorisation
"""
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
"""

#%% Algorithme supervisé : Classification
# Paramètres
max_words = 20000
max_len = 100
embedding_dim = 100

# Prétraitement des textes
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(data['claim'])
sequences = tokenizer.texts_to_sequences(data['claim'])
word_index = tokenizer.word_index
données = pad_sequences(sequences, maxlen=max_len)
print("sequences : \n", sequences, "\nword_index :\n", word_index, "\ndonnées :\n", données)

# Multi-label binarizer for labels
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(data['CPC'])

# Stratified Split using MultilabelStratifiedKFold
mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_index, test_index = next(mskf.split(données, labels))

X_train, X_test = données[train_index], données[test_index]
y_train, y_test = labels[train_index], labels[test_index]

#%% Custom Layer for ReduceSum
class ReduceSumLayer(Layer):
    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)

#%% Model Definition
inputs = Input(shape=(max_len,))
embedding_layer = Embedding(input_dim=max_words, output_dim=embedding_dim)(inputs)
lstm_out, state_h, state_c = LSTM(128, return_sequences=True, return_state=True)(embedding_layer)

# Attention Mechanism
attention = Attention()([lstm_out, lstm_out])
attention = ReduceSumLayer()(attention)

# Output Layer
outputs = Dense(labels.shape[1], activation='sigmoid')(attention)  # Use sigmoid for multi-label

# Compile Model
model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='auc')])
model.summary()

#%% Training
history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2)

#%% Evaluation
predictions = model.predict(X_test)

# Convert predictions to binary values
threshold = 0.5
predictions_binary = (predictions > threshold).astype(int)

# Inspect some predictions
print("Some predictions:\n", predictions_binary[:5])
print("True labels:\n", y_test[:5])

# Calcul des métriques avec une attention particulière aux dimensions
y_test_binary = y_test

# Calcul des métriques
f1 = f1_score(y_test_binary, predictions_binary, average='micro')
precision = precision_score(y_test_binary, predictions_binary, average='micro')
recall = recall_score(y_test_binary, predictions_binary, average='micro')

print(f'F1 Score: {f1}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')

print("Predictions (binary):\n", predictions_binary[:5])
print("True labels (binary):\n", y_test_binary[:5])

# Comparaison des formes
print("Shape of y_test_binary:", y_test_binary.shape)
print("Shape of predictions_binary:", predictions_binary.shape)

# Vérification des valeurs uniques dans les prédictions
print("Unique values in predictions_binary:", np.unique(predictions_binary))

# Vérification des valeurs uniques dans les étiquettes
print("Unique values in y_test_binary:", np.unique(y_test_binary))

print(classification_report(y_test_binary, predictions_binary))

#%% Final Evaluation
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')



