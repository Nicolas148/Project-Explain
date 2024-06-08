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

#%% Chargement de la base de données avec tri et nettoyage



try:
    # Lecture du fichier CSV
    data = pd.read_csv(r"C:\Users\nicol\OneDrive\Bureau\Documents\L3\s6\Master Camp\Project Explain\base_de_donnees.csv")
    # Afficher les premières lignes du DataFrame pour vérifier la lecture
    print(data.head(5))
    print(data.shape)
    print(data.info())
except Exception as e:
    # En cas d'erreur, afficher un message et l'erreur
    print("Erreur lors de la lecture du fichier CSV :", e)
    data = "error..."
 
#Suppression des doublons

print("Taille avant :",data.shape)
data = data.drop_duplicates()
print("Taille après :",data.shape)

#Remplir les valeurs manquantes
#fillna(0)
#%% Algorithme supervisé : Classification

#%% Main
#if __name__=='__main__':

