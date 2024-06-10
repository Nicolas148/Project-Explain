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

#%% Chargement de la base de données avec tri et nettoyage

try:
    # Lecture du fichier CSV
    data = pd.read_csv("EFREI - LIPSTIP - 50k elements EPO.csv")
    #data = pd.read_csv(r"C:\Users\nicol\OneDrive\Bureau\Documents\L3\s6\Master Camp\Project Explain\base_de_donnees.csv")
    
    # Afficher les premières lignes du DataFrame pour vérifier la lecture
    print(data.head(5))
    print(data.shape)
    print(data.info())
except Exception as e:
    # En cas d'erreur, afficher un message et l'erreur
    print("Erreur lors de la lecture du fichier CSV :", e)
    data = "error..."

#%%
#Suppression des doublons
print("Taille avant :",data.shape)
data = data.drop_duplicates()
print("Taille après :",data.shape)

#identification de possibles valeurs manquantes
print(data.isnull().all())

#Retirer les 4 premières colonnes
cols_to_mantain = ['CPC', 'IPC', 'claim', 'description']
data = data[cols_to_mantain]
print(data.shape)

#%%¨Nettoyage des données (balises):
    
balise_EPO_regex = r'<!--.*?-->'
balise_xml_regex = r'<[^>]+>'
number_regex = r'\d+[a-z]*'
reference_regex1 = r'\(([0-9]*[a-z]*|[ivxlcdm]*)\)'
reference_regex2 = r'\(\s*(\d+[a-z]*)(\s*,\s*\d+[a-z]*)*(\s*to\s*\d+[a-z]*)?\s*\)'
special_characters_regex = r'[^a-zA-Z0-9\s<>]'


# Function to clean descriptions
def clean_description(desc):
    desc = clean(desc, lower=True, no_currency_symbols = True, replace_with_currency_symbol = "<CUR>", lang="en") 
    desc = re.sub(balise_EPO_regex, '', desc)  # Remove comment tags
    desc = re.sub(balise_xml_regex, '', desc)  # Remove XML tags
    desc = re.sub(reference_regex1, '<REF>', desc) 
    desc = re.sub(reference_regex2, '<REF>', desc) 
    desc = re.sub(number_regex, ' <NUMBER>', desc) 
    desc = re.sub(special_characters_regex, '', desc)
    return desc
    

#Test sur les 4 premières ligens car c'est bcp trop long sur les 50000
data.loc[:4, 'description'] = data.loc[:4, 'description'].apply(clean_description)
    
print(data['description'][3])

  
#%% Algorithme supervisé : Classification

#%% Main
#if __name__=='__main__':

