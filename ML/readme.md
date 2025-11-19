## ESSALHI Hayat 
## apogee 25007996 
# Interprétation du code du notebook

## 1. Importation et installation des bibliothèques

Le code commence par installer et importer *ucimlrepo*, une
bibliothèque permettant de télécharger des jeux de données UCI.

## 2. Chargement du dataset Wine Quality

Le dataset wine_quality est téléchargé, puis ses caractéristiques
(features), sa cible (targets) et les informations sur les variables
sont affichées.\
Ce dataset est utilisé pour des problèmes de classification ou
régression liés à la qualité du vin.

## 3. Utilisation du modèle KNN

Le code importe *KNeighborsClassifier* pour construire un modèle de
machine learning basé sur les plus proches voisins.

## 4. Normalisation des données

Un *StandardScaler* est utilisé pour normaliser les données
d'entraînement, de validation et de test.\
Le code vérifie également les dimensions des matrices normalisées.
