# Essalhi hayat
<img src="essalhi.jpg" style="height:300px;margin-right:300px; float:left; border-radius:10px;"/>

Numéro d'étudiant : 25007996
Classe : CAC2
Compte rendu
Analyse Prédictive de Régression des Prix de Voitures
Date : 3 Décembre 2025​

Table des Matières
Introduction et Contexte

Analyse Exploratoire des Données (Data Analysis)

Chargement et Structure du Dataset

Distribution de la Variable Cible

Préparation pour la Régression

Analyse Statistique et Visuelle

Méthodologie de Régression

Séparation des Données (Data Split)

Algorithmes de Régression

Résultats et Comparaison des Modèles

Modèles Linéaires

Modèles Non-Linéaires

Comparaison des Performances

Conclusion

1. Introduction et Contexte
Ce rapport présente une analyse détaillée d'un jeu de données réel concernant les caractéristiques des voitures et leurs prix de vente, importé depuis Kaggle, réalisée dans le cadre d'une analyse prédictive de régression. En suivant le cycle de vie des données, nous avons mené une exploration (EDA), un prétraitement et une modélisation avec plusieurs algorithmes de régression.

L'objectif est de construire des modèles capables de prédire le prix de vente (sellingprice) en fonction des caractéristiques des voitures, et d'évaluer les performances relatives des modèles linéaires et non-linéaires via les métriques R², MSE et RMSE.​

2. Analyse Exploratoire des Données (Data Analysis)
2.1 Chargement et Structure du Dataset
Le jeu de données "CAR DETAILS FROM CAR DEKHO.csv" contient les caractéristiques des voitures et leurs prix de vente.

Nombre d'échantillons ($N$) : 8128 observations (inféré des splits 70/30).

Nombre de variables ($d$) : Multiples colonnes après encodage (features + target).

Variables d'entrée ($X$) : year, km_driven, fuel, seller_type, transmission, owner (catégorielles encodées via one-hot : fuel, seller_type, transmission, owner), autres numériques.
Variable de sortie ($Y$) : Prix de vente (sellingprice).                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
```python
import pandas as pd
# Chargement des données
df = pd.read_csv('content/drive/MyDrive/pro/analyse/CAR DETAILS FROM CAR DEKHO.csv')
df.info()
print(df.head())
```
2.2 Distribution de la Variable Cible
La variable sellingprice montre une distribution continue avec des valeurs élevées, nécessitant des modèles de régression pour la prédiction précise. Les visualisations scatter (actual vs predicted) révèlent des écarts importants pour les prix extrêmes.​

2.3 Préparation pour la Régression
Aucune valeur manquante détectée. La colonne 'name' est supprimée, et les variables catégorielles (fuel, seller_type, transmission, owner) sont encodées via one-hot encoding avec drop_first=True.
```python
df_processed = df.copy()
df_processed = df_processed.drop('name', axis=1)
categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']
df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
```
2.4 Analyse Statistique et Visuelle
Les corrélations et visualisations montrent des relations non-linéaires complexes entre features et prix. Les modèles linéaires peinent face aux ensembles d'arbres
```python
# Visualisations générées pour chaque modèle (scatter plots actual vs predicted)
```
3. Méthodologie de Régression
3.1 Séparation des Données (Data Split)
Les données sont divisées en ensembles d'entraînement (70%) et de test (30%) avec stratification via random_state=42 pour évaluer la généralisation.
```python
from sklearn.model_selection import train_test_split
X = df_processed.drop('selling_price', axis=1)
y = df_processed['selling_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
3.2 Algorithmes de Régression
Six modèles testés : Régression Linéaire, Ridge (alpha=1.0), Lasso (alpha=1.0), Arbre de Décision, Forêt Aléatoire, Gradient Boosting. Tous entraînés sur X_train, évalués sur X_test avec MSE, RMSE, R²
4. Résultats et Comparaison des Modèles
4.1 Modèles Linéaires
Les modèles linéaires (Linéaire, Ridge, Lasso) montrent des performances modérées avec R² ≈ 0.45 et MSE ≈ 1.63e11, limités par les relations non-linéaires.​

4.2 Modèles Non-Linéaires
Les ensembles surpassent : Forêt Aléatoire (R²=0.52, MSE=1.43e11), Gradient Boosting meilleur (R²=0.53, MSE=1.38e11, RMSE=372123.87). Arbre unique faible (R²=0.43).
4.3 Comparaison des Performances
| Modèle              | R²   | MSE     | RMSE      |
| ------------------- | ---- | ------- | --------- |
| Gradient Boosting   | 0.53 | 1.38e11 | 372123.87 |
| Forêt Aléatoire     | 0.52 | 1.43e11 | 378316.89 |
| Régression Linéaire | 0.45 | 1.63e11 | 403877.53 |
| Ridge               | 0.45 | 1.63e11 | 403819.44 |
| Lasso               | 0.45 | 1.63e11 | 403877.04 |
| Arbre de Décision   | 0.43 | 1.69e11 | 410683.06 |
5. Conclusion
Cette analyse valide l'importance des modèles non-linéaires pour prédire les prix de voitures : les ensembles (Gradient Boosting, Forêt Aléatoire) capturent les complexités mieux que les linéaires. Le prétraitement (one-hot encoding) et split 70/30 assurent robustesse. Optimisation hyperparamètres recommandée pour Gradient Boosting
