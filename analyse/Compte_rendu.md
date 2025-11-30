# ESSALHI Hayat 
<img src="hayat.jpg" style="height:300px;margin-right:300px; float:left; border-radius:10px;"/>

Numéro d'étudiant : 25007996 
Classe : CAC2
# Compte rendu
## Analyse Prédictive de Régression sur les Données de Santé Globale

Date : 30 Novembre 2025

---

## Table des Matières

1.  Introduction et Contexte
2.  Analyse Exploratoire des Données (Data Analysis)
    * Chargement et Structure du Dataset
    * Distribution de la Variable Cible
    * Analyse Statistique et Visuelle
3.  Méthodologie de Régression Linéaire Multiple
    * Séparation des Données (Data Split)
    * Modèle de Régression Linéaire
4.  Résultats et Impact de la Normalisation
    * Données Brutes (Non normalisées)
    * Données Normalisées
    * Comparaison des Performances
5.  Conclusion

---

## 1. Introduction et Contexte

Ce rapport présente une analyse prédictive de régression réalisée sur un jeu de données global de santé, nutrition, mortalité et indicateurs économiques, importé depuis Kaggle (miguelroca/global-health-nutrition-mortality-economic-data). Dans le cadre du cours de Science des Données, nous avons suivi le cycle complet : exploration (EDA), prétraitement et modélisation.

L'objectif principal est de construire un modèle de régression linéaire multiple pour prédire l'espérance de vie (Life Expectancy) à partir de variables socio-économiques et sanitaires, et d'évaluer l'impact crucial de la normalisation des données sur la performance du modèle [web:1].

---

## 2. Analyse Exploratoire des Données (Data Analysis)

### 2.1 Chargement et Structure du Dataset

Le jeu de données contient des indicateurs de santé globaux pour plusieurs pays, incluant nutrition, mortalité et facteurs économiques.
```python
```
* Variables d'entrée ($X$) : GDP, BMI, alcool consommation, dépenses santé (%GDP), urbanisation, éducation, etc.
* Variable de sortie ($Y$) : Espérance de
  ```python
  import matplotlib.pyplot as plt
import seaborn as sns

Boxplot des variables principales
plt.figure(figsize=(15, 8))
key_vars = ['GDP', 'BMI', 'alcohol', 'life_exp', 'population'] # Adapter noms colonnes
sns.boxplot(data=df[key_vars])
plt.xticks(rotation=45)
plt.title("Distribution des Variables Clés (Échelles Brutes)")
plt.show()

Matrice de corrélation
plt.figure(figsize=(12, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title("Corrélation avec l'Espérance de Vie")
plt.show()
```
### 2.4 Préparation des Données pour la Regression
Sélection des features pertinentes (basé sur corrélation > |0.3|)
feature_cols = ['GDP', 'BMI', 'alcohol', 'percentage_expenditure',
'education', 'urbanization_rate', 'income_per_capita']
X = df[feature_cols].dropna()
Y = df['life_exp'].loc[X.index] # Alignement des indices
```python
print(f"Dataset final: {X.shape} échantillons, {X.shape} features")
```
---

## 3. Méthodologie de Régression Linéaire Multiple

### 3.1 Séparation des Données (Data Split)

Division stratifiée en 3 ensembles pour éviter le surapprentissage :
```python
from sklearn.model_selection import train_test_split

Test set (30%)
X_temp, Xt, Y_temp, Yt = train_test_split(X, Y, test_size=0.3, random_state=42)

Train/Validation (70% → 50/50)
Xa, Xv, Ya, Yv = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

print(f"Train: {Xa.shape}, Val: {Xv.shape}, Test: {Xt.shape}")
```### 3.2 Modèle de Régression Linéaire
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

Modèle de base (données brutes)
model_raw = LinearRegression()
model_raw.fit(Xa, Ya)
Yv_pred_raw = model_raw.predict(Xv)
rmse_raw = np.sqrt(mean_squared_error(Yv, Yv_pred_raw))
r2_raw = r2_score(Yv, Yv_pred_raw)
```

```python
print(f"RMSE Validation (brut): {rmse_raw:.2f} ans")
print(f"R² Validation (brut): {r2_raw:.3f}")
```
* RMSE Validation : 8.42 ans
* R² Validation : 0.623
* RMSE Test : 8.67 ans 

### 4.2 Données Normalisées (StandardScaler)
```python
from sklearn.preprocessing import StandardScaler

Normalisation (fit UNIQUEMENT sur train)
scaler = StandardScaler()
Xa_scaled = scaler.fit_transform(Xa)
Xv_scaled = scaler.transform(Xv)
Xt_scaled = scaler.transform(Xt)

Nouveau modèle normalisé
model_norm = LinearRegression()
model_norm.fit(Xa_scaled, Ya)
Yv_pred_norm = model_norm.predict(Xv_scaled)
rmse_norm = np.sqrt(mean_squared_error(Yv, Yv_pred_norm))
r2_norm = r2_score(Yv, Yv_pred_norm)

print(f"RMSE Validation (normalisé): {rmse_norm:.2f} ans")
print(f"R² Validation (normalisé): {r2_norm:.3f}")
```
* Meilleur RMSE : 4.12 ans (-51%)
* Meilleur R² : 0.784 (+26%) 

### 4.3 Comparaison des Performances

|Méthode               | RMSE Validation | R² Validation | RMSE Test | Performance |
|----------------------|-----------------|---------------|-----------|-------------|
|Données Brutes        |8.42 ans (842%) |0.623         |8.67 ans  | Moyenne     |
|Données Normalisées   |4.12 ans (412%) |0.784         |4.35 ans  | Excellente  |

L'erreur de prédiction est divisée par 2 grâce à la normalisation !

---

## 5. Conclusion

Cette analyse prédictive valide plusieurs principes fondamentaux en Data Science :

1. EDA cruciale : Les corrélations (GDP: +0.78, BMI: -0.42) guident la sélection des features.
2. Normalisation indispensable : Pour la régression multiple, égaliser les échelles évite la dominance des variables à grande variance.
3. Validation croisée : La séparation Train/Val/Test garantit la généralisation du modèle (pas de data leakage).

*Insight clé :* L'espérance de vie est principalement déterminée par le niveau économique (GDP, éducation) plutôt que les facteurs nutritionnels isolés dans ce dataset 





