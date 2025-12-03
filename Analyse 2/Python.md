#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse Prédictive Complète - Prix de Vente des Voitures
Regroupement de tous les codes du notebook Analyse_2.ipynb
Dataset: CAR DETAILS FROM CAR DEKHO.csv
Auteur: Analyse prédictive regroupée
Date: 3 Décembre 2025
"""

# ============================================================================
# 1. IMPORT DES LIBRAIRIES
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Ignorer les warnings
warnings.filterwarnings('ignore')

# Configuration matplotlib pour affichage inline
plt.style.use('default')
sns.set_palette("husl")

print("✅ Toutes les librairies importées avec succès")

# ============================================================================
# 2. CHARGEMENT ET EXPLORATION DES DONNÉES
# ============================================================================
print("\n" + "="*60)
print("2. CHARGEMENT DES DONNÉES")
print("="*60)

# Chargement du dataset
df = pd.read_csv('content/drive/MyDrive/pro/analyse/CAR DETAILS FROM CAR DEKHO.csv')
print(f"📊 Dataset chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
print("\nPremières lignes:")
print(df.head())

# Vérification des valeurs manquantes
print("\n🔍 Vérification des valeurs manquantes:")
print(df.isnull().sum())

# ============================================================================
# 3. PRÉPARATION DES DONNÉES
# ============================================================================
print("\n" + "="*60)
print("3. PRÉPARATION DES DONNÉES")
print("="*60)

# Copie du dataframe et suppression de la colonne 'name'
df_processed = df.copy()
df_processed = df_processed.drop('name', axis=1)

# Colonnes catégorielles pour one-hot encoding
categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']

# Application de l'encodage one-hot
df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)

print(f"📈 Dataset après préprocessing: {df_processed.shape[0]} lignes, {df_processed.shape[1]} colonnes")
print("\nPremières lignes après encodage:")
print(df_processed.head())

# Séparation des features (X) et target (y)
X = df_processed.drop('selling_price', axis=1)
y = df_processed['selling_price']

# Split train/test 70/30
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\n🎯 Dimensions après split:")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")

# ============================================================================
# 4. FONCTION D'ÉVALUATION ET VISUALISATION
# ============================================================================
def evaluate_and_plot(model, X_train, X_test, y_train, y_test, model_name):
    """
    Entraîne un modèle, calcule les métriques et affiche un graphique.
    """
    # Entraînement
    model.fit(X_train, y_train)
    
    # Prédictions
    y_pred = model.predict(X_test)
    
    # Métriques
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📊 {model_name} - Performances:")
    print(f"   MSE: {mse:,.2f}")
    print(f"   RMSE: {rmse:,.0f}")
    print(f"   R²: {r2:.3f}")
    
    # Visualisation
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, label='Prédit vs Réel')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Prédiction parfaite')
    plt.xlabel('Prix de vente réel')
    plt.ylabel('Prix de vente prédit')
    plt.title(f'{model_name}: Réel vs Prédit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return {'MSE': mse, 'RMSE': rmse, 'R2': r2, 'model': model_name}

# ============================================================================
# 5. ENTRAÎNEMENT ET ÉVALUATION DES MODÈLES
# ============================================================================
print("\n" + "="*60)
print("4. ENTRAÎNEMENT DES MODÈLES")
print("="*60)

# Dictionnaire pour stocker les performances
performance_metrics = {}

# 5.1 Régression Linéaire
print("\n🔹 1. Régression Linéaire")
lr_model = LinearRegression()
perf_lr = evaluate_and_plot(lr_model, X_train, X_test, y_train, y_test, "Régression Linéaire")
performance_metrics['Régression Linéaire'] = perf_lr

# 5.2 Régression Ridge
print("\n🔹 2. Régression Ridge")
ridge_model = Ridge(alpha=1.0)
perf_ridge = evaluate_and_plot(ridge_model, X_train, X_test, y_train, y_test, "Régression Ridge")
performance_metrics['Régression Ridge'] = perf_ridge

# 5.3 Régression Lasso
print("\n🔹 3. Régression Lasso")
lasso_model = Lasso(alpha=1.0)
perf_lasso = evaluate_and_plot(lasso_model, X_train, X_test, y_train, y_test, "Régression Lasso")
performance_metrics['Régression Lasso'] = perf_lasso

# 5.4 Arbre de Décision
print("\n🔹 4. Arbre de Décision")
dt_model = DecisionTreeRegressor(random_state=42)
perf_dt = evaluate_and_plot(dt_model, X_train, X_test, y_train, y_test, "Arbre de Décision")
performance_metrics['Arbre de Décision'] = perf_dt

# 5.5 Forêt Aléatoire
print("\n🔹 5. Forêt Aléatoire")
rf_model = RandomForestRegressor(random_state=42)
perf_rf = evaluate_and_plot(rf_model, X_train, X_test, y_train, y_test, "Forêt Aléatoire")
performance_metrics['Forêt Aléatoire'] = perf_rf

# 5.6 Gradient Boosting
print("\n🔹 6. Gradient Boosting")
gbr_model = GradientBoostingRegressor(random_state=42)
perf_gbr = evaluate_and_plot(gbr_model, X_train, X_test, y_train, y_test, "Gradient Boosting")
performance_metrics['Gradient Boosting'] = perf_gbr

# ============================================================================
# 6. COMPARAISON DES PERFORMANCES
# ============================================================================
print("\n" + "="*60)
print("5. COMPARAISON DES PERFORMANCES")
print("="*60)

# DataFrame de comparaison
perf_df = pd.DataFrame(performance_metrics).T
perf_df = perf_df.sort_values('R2', ascending=False)
perf_df = perf_df.round(4)

print("\n🏆 Tableau de comparaison (trié par R² décroissant):")
print(perf_df)

# Sauvegarde des résultats
perf_df.to_csv('performance_modeles_regression.csv', index=True)
print("\n💾 Résultats sauvegardés dans 'performance_modeles_regression.csv'")

# Graphique de comparaison R²
plt.figure(figsize=(12, 6))
models = perf_df.index
r2_scores = perf_df['R2']
colors = plt.cm.viridis(np.linspace(0, 1, len(models)))

bars = plt.bar(models, r2_scores, color=colors, alpha=0.8, edgecolor='black')
plt.xlabel('Modèles')
plt.ylabel('Score R²')
plt.title('Comparaison des Scores R² par Modèle')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, max(r2_scores)*1.1)
plt.grid(True, alpha=0.3, axis='y')

# Ajout des valeurs sur les barres
for bar, score in zip(bars, r2_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================================================
# 7. CONCLUSION ET MEILLEUR MODÈLE
# ============================================================================
print("\n" + "="*60)
print("6. CONCLUSION")
print("="*60)

meilleur_modele = perf_df.index[0]
meilleur_r2 = perf_df.iloc[0]['R2']

print(f"🥇 MEILLEUR MODÈLE: {meilleur_modele}")
print(f"   Score R²: {meilleur_r2:.4f}")
print(f"   RMSE: {perf_df.iloc[0]['RMSE']:,.0f}")
print("\n📋 Résumé:")
print("- Les modèles d'ensemble (Gradient Boosting, Forêt Aléatoire) surpassent les modèles linéaires")
print("- Les relations prix/caractéristiques sont non-linéaires et complexes")
print("- Optimisation hyperparamètres recommandée pour le meilleur modèle")

print("\n🎉 Analyse complète terminée avec succès!")
print(f"Dataset initial: {df.shape}")
print(f"Dataset final: {df_processed.shape}")
print(f"Meilleur R² obtenu: {meilleur_r2:.4f}")
