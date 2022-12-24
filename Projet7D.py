# Construire un dashboard interactif à destination des gestionnaires de la relation client permettant :
# d'interpréter les prédictions faites par le modèle
# d’améliorer la connaissance client des chargés de relation client

#Permettre de visualiser le score et l’interprétation de ce score pour chaque client de façon intelligible pour une personne non experte en data science.
#Permettre de visualiser des informations descriptives relatives à un client (via un système de filtre).
#Permettre de comparer les informations descriptives relatives à un client à l’ensemble des clients ou à un groupe de clients similaires.

import streamlit as st
import pandas as pd
import numpy as np
import cv2
import math
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
from matplotlib.pyplot import figure
import joblib
import pickle
import shap
import json
import requests
import streamlit.components.v1 as components


rows = 250

df_base= pd.read_csv('application_train_sorted.csv', sep=',', nrows =rows)

#Création du volet d'affichage à gauche
st.sidebar.image("Logo.png", use_column_width=True)

select_id = st.sidebar.selectbox( "Select Loan Application ID", df_base.SK_ID_CURR.tolist())  # sélection du client
st.sidebar.write('Afficher informations client :')
check_client = st.sidebar.checkbox("Personnelles & financières")

#CREATION DES EN-TETES
new_title = '<p style="font-family:tahoma; color:#1e4046; font-size: 45px;">DASHBOARD CLIENT</p>'
st.markdown (new_title, unsafe_allow_html=True)
scd_title = '<p style="font-family:calibri; color:Grey; font-size: 36px;">------ Demande de prêt ------</p>'
st.markdown(scd_title, unsafe_allow_html=True)
st.caption ("Usage exclusif des conseillers clientèles - Informations confidentielles")
st.write ("Loan Application ID = ", select_id )  # Rappel du code client

#Modification df
df_base = df_base.rename({'DAYS_BIRTH':'Age (ans)','CODE_GENDER' :'Sexe',
                            'CNT_CHILDREN' : 'Nombre Enfant','NAME_EDUCATION_TYPE' :'Niveau d\'éducation', 'NAME_HOUSING_TYPE' : 'Logement Actuel',
                            'NAME_FAMILY_STATUS' :'Statut Familial', 'NAME_INCOME_TYPE':'Situation Prof.','DAYS_EMPLOYED':'Ancienneté Prof. (ans)',
                             'NAME_CONTRACT_TYPE':'Type emprunt sollicité', 'AMT_CREDIT' : 'Montant Emprunt ($)',
                             'AMT_ANNUITY': 'Montant Annuité ($)','AMT_INCOME_TOTAL' : 'Salaire Annuel ($)', 'AMT_GOODS_PRICE':'Montant du bien financé ($)'}, axis=1)

df_base['Ancienneté Prof. (ans)'] = abs((df_base['Ancienneté Prof. (ans)']/365).astype(np.int64, errors='ignore')).round(1)
df_base['Age (ans)'] = abs((df_base['Age (ans)']/365).astype(np.int64, errors='ignore'))
df_base['Sexe'] = df_base['Sexe'].replace(['M', 'F'],['Male', 'Female'])
df_base['Taux endettement (%)']=(df_base['Montant Annuité ($)']/df_base['Salaire Annuel ($)']*100).round(1)

scd_info_cli= '<p style="background-color:#50a28c;font-size:24px;border-radius:2%;">--- Informations situation personnelle & financière ---</p>'
st.markdown(scd_info_cli, unsafe_allow_html=True)
#CREATION TABLEAU SITUATION PERSONNELLE
df_client = df_base[['SK_ID_CURR', 'Age (ans)','Sexe', 'Nombre Enfant','Niveau d\'éducation', 'Logement Actuel',
                      'Statut Familial', 'Situation Prof.','Ancienneté Prof. (ans)']]
df_cli_select = df_client.loc[df_client['SK_ID_CURR']==select_id]
m = df_cli_select.select_dtypes(np.number)
df_cli_select[m.columns]= m.round().astype('Int64')
#CREATION TABLEAU SITUATION FINANCIERE
df_finance = df_base[['SK_ID_CURR', 'Type emprunt sollicité', 'Montant Emprunt ($)','Montant Annuité ($)',  
                        'Salaire Annuel ($)','Montant du bien financé ($)','Taux endettement (%)']]
df_fi_select = df_finance.loc[df_finance['SK_ID_CURR']==select_id]
df_fi_select['Montant du bien financé ($)']=df_fi_select['Montant du bien financé ($)'].map('{:,} $'.format)
df_fi_select['Montant Emprunt ($)']=df_fi_select['Montant Emprunt ($)'].map('{:,} $'.format)
df_fi_select['Montant Annuité ($)']=df_fi_select['Montant Annuité ($)'].map('{:,} $'.format)
df_fi_select['Salaire Annuel ($)']=df_fi_select['Salaire Annuel ($)'].map('{:,} $'.format)
df_fi_select['Taux endettement (%)']=df_fi_select['Taux endettement (%)'].astype(str) + '%'
# Masquer index
hide_dataframe_row_index = """ <style>.row_heading.level0 {display:none}.blank {display:none}</style>"""
st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
# Affichage détails client
if check_client:
    st.table(df_cli_select)
    st.table(df_fi_select)

#CREATION TABLEAU SITUATION COMPARATIVE
check_compare = st.sidebar.checkbox("Comparatives")
scd_info_comp= '<p style="background-color:#43818c;font-size:24px;border-radius:2%;">--- Informations comparatives clientèle ---</p>'
st.markdown(scd_info_comp, unsafe_allow_html=True)

if check_compare:
    check_compare_quali = st.checkbox("Comparatives Qualitatives")
#CREATION TABLEAU SITUATION COMPARATIVE QUALI

    if check_compare_quali:
        select_display1 = st.selectbox( 'Mode Affichage', ('Affichage basique', 'Affichage avec répartition Prêt Accepté (0) ou Refusé (1)'))
        select_variable = st.selectbox('Variable Qualitative', ('Logement Actuel', 'Statut Familial', 'Niveau d\'éducation', 'Sexe',
                                    'Situation Prof.','Type emprunt sollicité'))
        
        #valeur de la variable sélectionnée pour le client concerné
        spec_quali =df_base.loc[df_base['SK_ID_CURR']==select_id][select_variable].values[0]
        
        # affichage simple : 
        if select_display1 == 'Affichage basique':
            #liste des valeurs uniques de la variable sélectionnée (select_variable)
            fig = plt.figure(figsize=(8, 4))
            var_list = list(df_base[select_variable].unique())
            ax = sns.histplot(df_base[select_variable])
            #couleur spécifique pour la bin dans laquelle le client se trouve
            for i in range(len(ax.patches)):
                p= ax.patches[i] #PATCH , rectangle n°i
                j=var_list[i]
                if j == spec_quali:
                    p.set_facecolor('#50a28c')
            legend = patches.Patch(color='#50a28c', label='Client\'s bracket')
            plt.legend(handles=[legend], title='Valeur client = '+str(spec_quali),loc=1, fontsize='small', fancybox=True)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # affichage variable 'TARGET' : 
        if select_display1 == 'Affichage avec répartition Prêt Accepté (0) ou Refusé (1)':
            st.write('Valeur client : ' + str(spec_quali))
            fig = plt.figure(figsize=(8, 4))
            ax = sns.histplot(data=df_base, x=df_base[select_variable], hue='TARGET', multiple="stack")
            plt.xticks(rotation=45)
            st.pyplot(fig)

#CREATION TABLEAU SITUATION COMPARATIVE QUANTITATIVE
    check_compare_quanti = st.checkbox("Comparatives Quantitatives")
    if check_compare_quanti:
        select_display2 = st.selectbox( 'Mode Affichage', ('Affichage basique', 'Affichage avec répartition Prêt Accepté (0) ou Refusé (1)'))
        select_variable = st.selectbox('Variable Quantitative', ('Age (ans)', 'Nombre Enfant','Ancienneté Prof. (ans)','Montant Emprunt ($)',
                                    'Montant Annuité ($)', 'Salaire Annuel ($)', 'Montant du bien financé ($)', 'Taux endettement (%)'))
        spec_quanti =df_base.loc[df_base['SK_ID_CURR']==select_id][select_variable].values[0]

        # affichage simple :
        if select_display2 == 'Affichage basique':
            var_list = list(df_base[select_variable].unique())
            var_list.sort()
            if len(var_list) > 5:
                var_list = np.linspace(min(var_list), max(var_list), num=25)
            else : 
                var_list = var_list  
            fig = plt.figure(figsize=(8, 4))
            patch_index = np.digitize([spec_quanti], var_list)[0]-1
            ax = sns.histplot(df_base[select_variable] ,bins=var_list)
            plt.axvline(x=df_base[select_variable].median(),color='#a95f6f',ls='--')
            ax.patches[patch_index].set_color('#50a28c')
            #légende
            legend1 = patches.Patch(color='#50a28c', label='Client\'s bracket')
            legend2 = Line2D([0], [0], color='#a95f6f', label='Median',ls='--')
            if select_variable in ['Age (ans)','Nombre Enfant','Ancienneté Prof. (ans)']:
                titre = 'Valeur client = '+str(spec_quanti)
            elif select_variable == 'Taux endettement (%)':
                titre = 'Valeur client = '+str(spec_quanti) +' %'
            else : 
                titre = 'Valeur client = $'+str(spec_quanti) 
            plt.legend(handles=[legend1, legend2], title=titre,loc=1, fontsize='small', fancybox=True)
            st.pyplot(fig)

        # affichage variable 'TARGET' : 
        if select_display2 == 'Affichage avec répartition Prêt Accepté (0) ou Refusé (1)':
            st.write('Valeur client : ' + str(spec_quanti))
            fig = plt.figure(figsize=(8, 4))
            ax = sns.histplot(data=df_base, x=df_base[select_variable], hue='TARGET', multiple="stack")
            plt.xticks(rotation=45)
            st.pyplot(fig)

# SCORE
check_score = st.sidebar.checkbox("Score Client")
scd_info_score= '<p style="background-color:#50a28c;font-size:24px;border-radius:2%;">--- Score Client ---</p>'
st.markdown(scd_info_score, unsafe_allow_html=True)
 # df_final a déjà été trié par ordre croissant de SK_ID_CURR avant enregistrement
df_final = pd.read_csv('df_final.csv', sep=',', nrows=rows)  
df_1 = df_final.drop('TARGET', axis=1)

def get_score_from_pickle():
    model = pickle.load(open('model.pkl', 'rb'))
    prediction = model.predict_proba(df_1[df_1['SK_ID_CURR']== select_id])
    P = np.array(prediction).flatten()
    score = P[1]
    return score

def get_score_from_api():
    url = 'http://127.0.0.1:5000/prediction'
    r = requests.post(url,json={'SK_ID_CURR': select_id})
    score = float(r.json())
    return score

if check_score:
    score = get_score_from_api()
    st.write('Prédiction de la **probabilité de faillite**. ')
    score_text = f'Risque de défaut de paiement de {str(int(score*100))} %.'
    st.write(score_text, unsafe_allow_html=True)
    if score>0.5:
        st.write('Demande de prêt susceptible d\'être **REFUSEE**.')
    else:
        st.write('Demande de prêt susceptible d\'être **ACCEPTEE**.')
    import plotly.graph_objects as go
    fig1 = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = score,
        mode = "gauge+number",
        title = {'text': "Score"},
        gauge = {'axis': {'range': [None, 1]},
                'bar': {'color': "darkblue"},
                'steps' : [
                    {'range': [0, 0.5], 'color': "green"},
                    {'range': [0.5, 1], 'color': "red"}]}))
    st.plotly_chart(fig1)

# POIDS DES FEATURES
check_FI = st.sidebar.checkbox("Importance des variables")
scd_info_comp= '<p style="background-color:#43818c;font-size:24px;border-radius:2%;">--- Affichage Importance des Variables ---</p>'
st.markdown(scd_info_comp, unsafe_allow_html=True)
if check_FI:
    
    # FEATURES GLOBALES
    check_FI_Global = st.checkbox("Importance des variables globales")
    if check_FI_Global:

        st.write('Importance des features pour tous les clients:')
        
        data = [['EXT_SOURCE_2', 'Score normalisé n°2 d\'une base de donnée externe'], 
                ['EXT_SOURCE_3', 'Score normalisé n°3 d\'une base de donnée externe'], 
                ['PAYMENT_RATE', '% remboursé annuellement sur montant total du prêt demandé'], 
                ['AMT_GOODS_PRICE', 'Montant du bien financé ($)'],
                ['AMT_ANNUITY', 'Montant Annuité ($)'],
                ['DAYS_EMPLOYED', 'Ancienneté Prof. (ans)'],
                ['INSTAL_DPD_MEAN', 'Moyenne du Nombre d\'échéance en retard'],
                ['DAYS_BIRTH', 'Age (ans)']]
  
        col_names = ["Feature Code", "Traduction"]
        df_légende = pd.DataFrame(data, columns=col_names)

        tab1, tab2 = st.tabs(["Chart", "Data"])
        with tab1:
            st.image("grafic.png", use_column_width=True)
        with tab2:
            st.table(df_légende)

        st.write('Répartition des Features Principales non visibles dans "Données Comparatives": ')
        df_boxplot = df_final [['EXT_SOURCE_2','EXT_SOURCE_3','PAYMENT_RATE', 'INSTAL_DPD_MEAN']]
        for column in df_boxplot :
            fig3 = plt.figure(figsize=(17,1))
            sns.set_theme(style="whitegrid")
            sns.boxplot(data = df_boxplot, x=column, palette = 'Greens', showfliers=False)        
            st.pyplot(fig3)

    # FEATURES LOCALES
    check_FI_Local = st.checkbox("Importance des variables locales")
    if check_FI_Local:
        
        model = pickle.load(open('model.pkl', 'rb'))
        st.write('Importance des features pour ce client:')  
        shap.initjs()
        X = df_final.loc[:, df_final.columns != 'TARGET']
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        index = df_final.SK_ID_CURR[df_final.SK_ID_CURR == select_id].index.tolist()

        figure(figsize=(8, 6), dpi=80)
        fig2=shap.plots.bar(shap_values[index])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(fig2)
        
        def st_shap(plot, height=None):
            shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
            components.html(shap_html, height=height)

        shap_v = explainer.shap_values(X)
        i = index[0]
        # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
        st_shap(shap.force_plot(explainer.expected_value, shap_v[i,:], X.iloc[i,:]))