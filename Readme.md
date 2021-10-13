# Parcours Data Scientist - OpenClassrooms  

                                    Projet N° 7 : Implémentez un modèle de scoring
                                    ----------------------------------------------
## Mission

* Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.
* Construire un dashboard interactif à destination des gestionnaires de la relation client permettant d'interpréter les prédictions faites par le modèle et d’améliorer la connaissance client des chargés de relation client.

## Spécifications du dashboard

* Permettre de visualiser le score et l’interprétation de ce score pour chaque client de façon intelligible pour une personne non experte en data science.
* Permettre de visualiser des informations descriptives relatives à un client (via un système de filtre).
* Permettre de comparer les informations descriptives relatives à un client à l’ensemble des clients ou à un groupe de clients similaires.

## Contenu du dossier Notebooks_Code_Dashboard

* **P7_01_Analyse_EDA :** Analyse exploratoire des données.
* **P7_02_Feature Engineering :** Feature Engineering.
* **P7_03_Modelisation_1 :** Modélisation avec l'ensemble des 1242 features obtenues au terme du 'feature engineering'. =>LDA, XGBOOST et Lightgbm.
* **P7_04_Modelisation_2 :** Modélisation avec les 50 principales features. => Lightgbm et Logistic Regression.
* **P7_05_Modelisation_3 :** Modélisation avec les 350 principales features. => Lightgbm et Linear Discriminant Analysis (LDA).
* **P7_06_Modelisation_4 :** Modélisation XGBOOST avec les 300 principales features - Modèle finalement retenu.
* **P7_07_Annexe1_data :** Annexe 1 - Préparation des jeux de données pour l'entraînement des modèles de classification et pour le tableau de bord Streamlit.
* **P7_08_Annexe2_data_xgb_300 :** Annexe 2 - Préparation des jeux de données pour l'affichage des shap values pour le modèle Xgboost.
* **P7_09_Code_Dashboard_API_Streamlit :** Code pour l'affichage du dashboard et le déploiement en ligne sur la plateforme Streamlit.
* **P7_10_Note_Méthodologique**.
* **P7_11_Présentation**.

## Utilisation de la bibliothèque Pycaret pour les travaux de modélisation

* PyCaret est une bibliothèque d'apprentissage automatique open source en Python qui automatise les flux de travail d'apprentissage automatique. Il s'agit d'un outil d'apprentissage automatique et de gestion de modèles de bout en bout qui accélère le cycle d'expérimentation de manière exponentielle et le rend plus productif.

* En comparaison avec les autres bibliothèques d'apprentissage automatique open source, PyCaret est une bibliothèque alternative qui peut être utilisée pour remplacer des centaines de lignes de code par quelques lignes de code seulement. Cela rend les expériences exponentiellement rapides et efficaces. 

* PyCaret est essentiellement un wrapper Python construit autour de plusieurs bibliothèques et frameworks d'apprentissage automatique tels que scikit-learn, XGBoost, LightGBM, CatBoost, spaCy, Optuna, Hyperopt, Ray et bien d'autres.

## Méthodologie mise en oeuvre lors des travaux de modélisation

* **Etape 1 : choix des modèles**

Présélection de quelques modèles candidats => entraînement de tous les modèles disponibles dans la bibliothèque de modèles avec notation à l'aide de la validation croisée Kfold (10 folds) pour l'évaluation des métriques (AUC, F1-Score, Log-Loss) et du temps d’entraînement.

* **Etape 2 : création et notation du modèle choisi**

A l'aide de la validation croisée K-fold (10 plis).

* **Etape 3 : Optimisation du modèle et réglage des hyperparamètres**

La fonction tune_model  règle automatiquement les hyperparamètres d'un modèle à l'aide de la recherche par grille aléatoire sur un espace de recherche prédéfini. Possibilité de passer le paramètre custom_grid dans la fonction tune_model. Utilisation possible de fonctions de recherche optimisées avec Hyperopt et Optuna.

* **Etape 4 : Prédiction sur le jeu de données entraîné avec une validation croisée**

Prédire l'ensemble test/hold-out (30 %) du jeu d’entraînement initial.

* **Etape 5 : Finaliser le modèle pour le déploiement en production**

La fonction finalize_model() ajuste le modèle sur l'ensemble de données d’entraînement complet, y compris l'échantillon test/hold-out (30% ).

* **Etape 6 : Prédictions sur le 'unseen df'**

Jeu de test non traité et représentant 20 % du jeu de données complet.

## Utilisation de Streamlit pour la création du Dashboard et son déploiement en ligne

Streamlit est un framework d'application en python dédié à la science des données. Il est livré avec un serveur Web intégré.

Le choix a été fait de déployer l'application sur la plateforme Streamlit en mode partage.

Cela constitue une excellente alternative à Heroku car, en accès gratuit, ce dernier limite la taille du slug à 500 Mo. Dés lors que la bibliothèque PyCaret possède elle-même beaucoup de dépendances, la taille de l'application Web dépasse rapidement la limite de taille de slug de Heroku. Streamlit présente de ce point de vue un avantage intéressant puisqu'il n'est pas contraint par cette limite.

## Lien Streamlit vers le dashboard en ligne

https://share.streamlit.io/henrique-dacosta/p7scoring/main/app/app.py

## Exécution de l'application en mode local

Aprés clonage du dossier Github, vous placer dans le dossier 'app' et tapez la commande : streamlit run app.py.

## Lien Github du répo

https://github.com/henrique-dacosta/p7scoring


