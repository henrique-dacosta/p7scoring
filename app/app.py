import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

import seaborn as sns
import plotly.express as px
plt.style.use('fivethirtyeight')
# Classifieur Xgboost
import xgboost
# Librairie Pycaret et scikit-learn
import pycaret
from pycaret.classification import *
from pycaret.utils import check_metric
from sklearn.metrics import log_loss
from pycaret.classification import load_model, predict_model
from sklearn.model_selection import train_test_split
# SMOTE
# import imblearn
# from imblearn.over_sampling import SMOTE
# SHAP
import shap
shap.initjs()
# Chargement et traitement d'image image
from PIL import Image


### Programme de traitement et d'affichage des données client ###

def main() :

    # Chargement des données
    @st.cache
    def load_data():
        
        # Informations sur le client choisi dans la base Test sans Target
        test_info_client = pd.read_csv('../data_tableau_300_xgb/test_info_client_300_sample.csv', index_col='SK_ID_CURR', encoding ='utf-8').drop('Unnamed: 0', axis=1)
        selection_clients = pd.read_csv('../data_tableau_300_xgb/selection_clients.csv').drop('Unnamed: 0', axis=1)
        
        # Jeu de données pour les comparaisons dans la base Train avec Target
        train_compare = pd.read_csv('../data_tableau_300_xgb/train_df_300_sample.csv', index_col='SK_ID_CURR', encoding ='utf-8').drop('Unnamed: 0', axis=1)
        compare_client = pd.read_csv('../data_tableau_300_xgb/train_df_300_sample.csv', encoding ='utf-8').drop('Unnamed: 0', axis=1)
        # train_df_std_300_sample = pd.read_csv('../data_tableau_300_xgb/train_df_std_300_sample.csv',encoding ='utf-8').drop('Unnamed: 0', axis=1)
        
        # Jeu de données pour la prédiction sur la base Test avec le classifieur Final Xhboost Model
        # test_predict_pycaret = pd.read_csv('../data_tableau_300_xgb/test_df_std_300_sample.csv').drop('Unnamed: 0', axis=1) 
        test_df_std_sample = pd.read_csv('../data_tableau_300_xgb/test_df_std_300_sample.csv').drop('Unnamed: 0', axis=1)
        
        # Jeux de données pour les features importance (SHAP Values)
        train_shap = pd.read_csv('../data_tableau_300_xgb/train_shape.csv').drop('Unnamed: 0', axis=1)
        test_shap = pd.read_csv('../data_tableau_300_xgb/test_shape.csv').drop('Unnamed: 0', axis=1)
        y_shap = pd.read_csv('../data_tableau_300_xgb/y_shape.csv').drop('Unnamed: 0', axis=1)
        
        
        target = train_compare.iloc[:, -1:]

        return test_info_client, selection_clients, train_compare, compare_client, test_df_std_sample, train_shap, test_shap, y_shap, target


    @st.cache
    def load_infos_gen(data):
        lst_infos = [data.shape[0],
                     round(data["AMT_INCOME_TOTAL"].mean(), 2),
                     round(data["AMT_CREDIT"].mean(), 2)]

        nb_credits = lst_infos[0]
        rev_moy = lst_infos[1]
        credits_moy = lst_infos[2]

        targets = data.TARGET.value_counts()

        return nb_credits, rev_moy, credits_moy, targets


    def identite_client(data, id):
        data_client = data[data.index == int(id)]
        return data_client

    @st.cache
    def load_age_population(data):
        data_age = round((data["AGE"]), 2)
        return data_age

    @st.cache
    def load_income_population(sample):
        df_income = pd.DataFrame(sample["AMT_INCOME_TOTAL"])
        df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
        return df_income
    
    @st.cache
    def load_prediction(predict_test, id):
        Score = float(predict_test[predict_test['SK_ID_CURR'] == int(id)].Score.values)
        Label = int(predict_test[predict_test['SK_ID_CURR'] == int(id)].Label.values)
        return Score, Label
    

    # Chargement des données ……
    test_info_client, selection_clients, train_compare, compare_client, test_df_std_sample, train_shap, test_shap, y_shap, target = load_data()
    id_client = selection_clients['ID'].values
    clf  =  load_model ( '../Save_Model/Final XGBOOST Model 06oct2021' )
    predict_test = predict_model ( clf ,  probability_threshold = 0.74, data = test_df_std_sample )
     
    # Renommer colonne 'DAYS_BIRTH' => 'AGE' et convertir en integer
    test_info_client = test_info_client.rename({'DAYS_BIRTH':'AGE'}, axis=1)
    test_info_client['AGE'] = test_info_client['AGE'].astype(int)

    #######################################
    # SIDEBAR
    #######################################

    # Titre
    html_temp = """
    <div style="background-color: green; padding:10px; border-radius:10px">
    <h1 style="color: white; text-align:center">Tableau de bord de scoring crédit</h1>
    </div>
    <p style="font-size: 20px; font-weight: bold; text-align:center">Aide à la décision d'octroi du prêt …</p>
    """
    st.write(html_temp, unsafe_allow_html=True)

   ### Affichage logo Prêt à dépenser ###

    im = Image.open("../logo/logo_pad_circle.png")
    
    col1, col2, col3 = st.sidebar.columns([30,250,60])
    with col1:
        st.write("")
    with col2:
        st.image(im, width=250)
    with col3:
        st.write("")
    
    # Sélection identifiant client
    st.sidebar.header("**Informations générales**")

    # Choix de l'identifiant
    chk_id = st.sidebar.selectbox("Identifiant Client", id_client)

    # Chargement des informations générales
    nb_credits, rev_moy, credits_moy, targets = load_infos_gen(train_compare)


    ### Affichage des informations sur la sidebar ###
    # Nombre de prêts dans l'échantillon
    st.sidebar.write("<u>Nombre de prêts dans l'échantillon :</u>", unsafe_allow_html=True)
    st.sidebar.text(nb_credits)

    # Revenu moyen
    st.sidebar.write("<u>Revenu moyen (USD) :</u>", unsafe_allow_html=True)
    st.sidebar.text(rev_moy)

    # Montant du crédit moyen
    st.sidebar.write("<u>Montant du crédit moyen (USD) :</u>", unsafe_allow_html=True)
    st.sidebar.text(credits_moy)
    
    #PieChart
    #st.sidebar.markdown("<u>......</u>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5,5))
    plt.pie(targets, explode=[0, 0.1], labels=['Non défaillant', 'Défaillant'], autopct='%1.1f%%', startangle=90)
    st.sidebar.pyplot(fig)
        

    #######################################
    # CONTENU DE LA PAGE PRINCIPALE
    #######################################
    # Affichage de l'identifiant client depuis la Sidebar
    st.write("Sélection identifiant client :", chk_id)


    # Affichage des informations client : Sexe, Age, Status familial, Enfants, …
    st.header("**Informations du client**")

    if st.checkbox("Informations relatives au client."):

        infos_client = identite_client(test_info_client, chk_id)
        st.write("**Sexe : **", infos_client["CODE_GENDER"].values[0])
        st.write("**Age : **{:.0f} ans".format(int(infos_client["AGE"].values[0])))
        st.write("**Status familial : **", infos_client["NAME_FAMILY_STATUS"].values[0])
        st.write("**Nombre d'enfants : **{:.0f}".format(infos_client["CNT_CHILDREN"].values[0]))
        
        # Affecter 'WEIGHTED_EXT_SOURCE' à une avariable pour pie chart comparatif en fin de dashboard
        wscore_client = round(infos_client['WEIGHTED_EXT_SOURCE'].values[0], 2)
    
        # Histogramme des âges
        data_age = load_age_population(test_info_client)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_age, edgecolor = 'k', color="blue", bins=20)
        ax.axvline(int(infos_client["AGE"].values), color="green", linestyle='--')
        ax.set(title="Position du client dans l'histogramme des âges", xlabel='Age(Années)', ylabel='')
        st.pyplot(fig)
    
        
        st.subheader("*Revenu (USD)*")
        st.write("**Revenu total : **{:.0f}".format(infos_client["AMT_INCOME_TOTAL"].values[0]))
        st.write("**Montant du crédit : **{:.0f}".format(infos_client["AMT_CREDIT"].values[0]))
        st.write("**Annuités de crédit : **{:.0f}".format(infos_client["AMT_ANNUITY"].values[0]))
        st.write("**Valeur du bien financé : **{:.0f}".format(infos_client["AMT_GOODS_PRICE"].values[0]))
        
        # Histogramme des revenus
        data_income = load_income_population(test_info_client)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_income["AMT_INCOME_TOTAL"], edgecolor = 'k', color="blue", bins=10)
        ax.axvline(int(infos_client["AMT_INCOME_TOTAL"].values[0]), color="green", linestyle='--')
        ax.set(title="Position du client dans l'histogramme des revenus", xlabel='Revenu (USD)', ylabel='')
        st.pyplot(fig)
    
    else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)

    # Solvabilité du client
    st.header("**Analyse du dossier client**")
    Score, Label = load_prediction(predict_test, chk_id)
    st.write(chk_id)
    if Label == 1:
        st.write("**Défaillant avec un probabilité de : **{:.0f} %".format(round(float(Score)*100, 2)))
    else:
        st.write("**Non Défaillant avec un probabilité de : **{:.0f} %".format(round(float(Score)*100, 2)))
        
    # Dataframe avec l'ensemble des caractérisques du client
    col_affiche = ['CODE_GENDER', 'AGE', 'NAME_FAMILY_STATUS', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL','AMT_CREDIT', 'AMT_ANNUITY', 'CREDIT_EXT_RATIO', 'WEIGHTED_EXT_SOURCE']
    st.markdown("<u>Données du client :</u>", unsafe_allow_html=True)
    st.write(identite_client(test_info_client[col_affiche], chk_id))

    # Feature importance / SHAP Values
    
    if st.checkbox("Identifiant client {:.0f} : caractéristiques importantes.".format(chk_id)):
        shap.initjs()
        X = train_shap
        y = y_shap
        # create a train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
        d_train = xgboost.DMatrix(X_train, label=y_train)
        d_test = xgboost.DMatrix(X_test, label=y_test)
        # Former le modele
        params = {
            "eta": 0.01,
            "objective": "binary:logistic",
            "subsample": 0.5,
            "base_score": float(np.mean(y_train)),
            "eval_metric": "logloss"
        }
        model = xgboost.train(params, d_train, 10000, evals = [(d_test, "test")], verbose_eval=100, early_stopping_rounds=20)
        
        client_shap = test_shap[test_shap['SK_ID_CURR'] == chk_id]
            
        # Interprétation et Affichage du bar plot des features importances
        fig, ax = plt.subplots(figsize=(10, 10))
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        number = st.slider("Choix du nombre de caratéristiques du client …", 0, 20, 8)
        shap.summary_plot(shap_values, client_shap, max_display=number, plot_type ="bar", color_bar=False)
        st.pyplot(fig)
        
    else:
        st.write("<i>…</i>", unsafe_allow_html=True)
    
    # Afficahe des principales caractéristiques des clients similaires défaillants et non défaillants
    
    if st.checkbox("Prinicipales caractéristiques de clients similaires selon les critères de : sexe, status familial, âge, revenu, montant du crédit."):
        
        # Masques de sélection
        sexe = infos_client['CODE_GENDER'].values[0]
        age = infos_client['AGE'].values[0]
        revenu = infos_client['AMT_INCOME_TOTAL'].values[0]
        credit = infos_client['AMT_CREDIT'].values[0]
        status = infos_client['NAME_FAMILY_STATUS'].values[0]
        child = infos_client['CNT_CHILDREN'].values[0]
        
        mask_1 = compare_client['CODE_GENDER'] == sexe
        mask_2 = compare_client['NAME_FAMILY_STATUS'] == status
        mask_3 = (compare_client['DAYS_BIRTH'] > 0.90 * age) & (compare_client['DAYS_BIRTH'] < 1.10 * age)
        mask_4 = (compare_client['AMT_INCOME_TOTAL'] > 0.70 * revenu) & (compare_client['AMT_INCOME_TOTAL'] < 1.3 * revenu)
        mask_5 = (compare_client['AMT_CREDIT'] > 0.50 * credit) & (compare_client['AMT_CREDIT'] < 1.50 * credit)
        
        # Clients avec un profil similaire défaillants
        st.write("**Clients avec un profil similaire défaillants**")
        mask_0 = compare_client['TARGET'] == 1
        df_compare = compare_client[mask_0 & mask_1 & mask_2 & mask_3 & mask_4 & mask_5 ]
        df_compare = df_compare[[ 'SK_ID_CURR','CODE_GENDER', 'DAYS_BIRTH', 'NAME_FAMILY_STATUS', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', \
                                            'AMT_CREDIT', 'AMT_ANNUITY', 'CREDIT_EXT_RATIO', 'WEIGHTED_EXT_SOURCE', 'TARGET']]
        df_compare = df_compare.rename({'DAYS_BIRTH':'AGE'}, axis=1)
        df_compare['AGE'] = df_compare['AGE'].astype(int)
        
        # Affecter 'WEIGHTED_EXT_SOURCE' à une avariable pour pie chart comparatif en fin de dashboard
        wscore_default = round(df_compare['WEIGHTED_EXT_SOURCE'].values[0], 2)
        
        st.write(df_compare)
        
        # Clients avec un profil similaire non défaillants
        st.write("**Clients avec un profil similaire non défaillants**")
        mask_0 = compare_client['TARGET'] == 0
        df_compare = compare_client[mask_0 & mask_1 & mask_2 & mask_3 & mask_4 & mask_5 ]
        df_compare = df_compare[[ 'SK_ID_CURR','CODE_GENDER', 'DAYS_BIRTH', 'NAME_FAMILY_STATUS', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', \
                                            'AMT_CREDIT', 'AMT_ANNUITY', 'CREDIT_EXT_RATIO', 'WEIGHTED_EXT_SOURCE', 'TARGET']]
        df_compare = df_compare.rename({'DAYS_BIRTH':'AGE'}, axis=1)
        df_compare['AGE'] = df_compare['AGE'].astype(int)
        
        # Affecter 'WEIGHTED_EXT_SOURCE' à une avariable pour pie chart comparatif en fin de dashboard
        wscore_regular = round(df_compare['WEIGHTED_EXT_SOURCE'].values[0], 2)
        
        st.write(df_compare)
        
        st.write("**Score normalisé comparatif sur une échelle de 0 à 9**")
        
        fig, ax = plt.subplots(figsize=(1,1))
        scores_normal = [wscore_client, wscore_default, wscore_regular]
        plt.pie(scores_normal, labels=['Client', 'Défaillant', 'Non défaillant'], autopct='%1.1f%%', textprops={'fontsize': 5}, startangle=90)
        st.pyplot(fig)
        
        st.write("Score normalisé du client : ", wscore_client)
        st.write("Score normalisé moyen des défaillants : ", wscore_default)
        st.write("Score normalisé moyen des non défaillants : ", wscore_regular)
        
    st.markdown('***')
    st.markdown("**Outil d'aide à la décision développé par la groupe Prêt à dépenser**.")


if __name__ == '__main__':
    main()