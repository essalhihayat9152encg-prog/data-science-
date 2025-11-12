## Cours de science des donnees 
## Encg 4 annee - CAC2 
## Hayat Essalhi 
<img src="image 1.jpg" style="height:464px;margin-right:432px"/>
## code apogee 25007996
## nom du jeu : HEART DISEASE 
## Descriptive de la base de donnee 
Le jeu de données *Heart Disease* provient du *UCI Machine Learning Repository, une base de référence pour les projets d’apprentissage automatique. Il a été créé par **Robert Detrano* et ses collaborateurs de la *Cleveland Clinic Foundation, avec la contribution de plusieurs institutions médicales telles que **Hungarian Institute of Cardiology, **V.A. Medical Center (Long Beach, Californie)* et *University Hospital (Zurich, Suisse)*.  

L’objectif initial de la collecte de ces données était de *développer un modèle prédictif permettant de diagnostiquer la présence d’une maladie cardiaque* chez un patient, à partir de ses caractéristiques cliniques et de résultats d’examens médicaux simples. Les chercheurs souhaitaient faciliter le diagnostic précoce des maladies cardiovasculaires sans recourir à des méthodes invasives.  

Ce jeu de données est devenu depuis l’un des ensembles les plus célèbres et les plus utilisés pour l’apprentissage automatique, notamment pour la *classification binaire* (maladie cardiaque présente = 1, absente = 0).  

Il contient *303 observations* représentant des patients, chacun décrit par *14 variables* (attributs) incluant des facteurs démographiques, cliniques et biologiques. Les variables principales sont : l’âge, le sexe, la tension artérielle au repos, le cholestérol, la fréquence cardiaque maximale atteinte, la présence d’angine de poitrine induite par l’effort, la glycémie à jeun, ainsi que d’autres indicateurs électrocardiographiques.  

La *variable cible*, notée généralement target, indique si le patient présente une maladie cardiaque. Une valeur de 1 signifie la présence d’une affection cardiaque, tandis qu’une valeur de 0 indique l’absence de maladie.  

Ce jeu de données a plusieurs avantages : il est bien documenté, de taille modérée, et équilibré, ce qui le rend idéal pour les travaux d’expérimentation et d’apprentissage. Il est souvent utilisé pour tester des algorithmes de *régression logistique, **arbres de décision, **k-plus proches voisins, **réseaux de neurones, ou **SVM*.  

Sur le plan méthodologique, ce jeu de données est également intéressant pour étudier les *corrélations entre les facteurs de risque cardiovasculaires* (comme l’âge, la tension, ou le cholestérol) et la probabilité de développer une maladie cardiaque.  

En résumé, le jeu de données *Heart Disease* a été créé dans le but d’aider les médecins et les chercheurs à mieux comprendre les déterminants de la santé cardiaque et à construire des modèles de prédiction fiables. Il reste aujourd’hui un *jeu de référence pour l’enseignement, la recherche et l’expérimentation en data science médicale.*
``` python
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipython-input-83676083.py in <cell line: 0>()
----> 1 from ucimlrepo import fetch_ucirepo
      2 
      3 # fetch dataset
      4 heart_disease = fetch_ucirepo(id=45)
      5 

ModuleNotFoundError: No module named 'ucimlrepo'

