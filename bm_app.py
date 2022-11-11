import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
df=pd.read_csv("bank.csv", sep=';')
df_cat = df.select_dtypes(include='object').columns.drop(["deposit"])
df_num = df.select_dtypes(exclude='object').columns















pages = ["Le projet Bank Marketing", "Dataviz", "Modélisation", "Conclusion et perspective"]
st.sidebar.title("Bank Marketing Project")

page= st.sidebar.radio('Menu', pages)


if page=='Le projet Bank Marketing':
    st.image('image9.jpg')


    
    st.title("Le projet Bank Marketing")
    st.write("Le projet **Bank Marketing** se concentre sur l'analyse d'un ensemble de données de \nmarketing bancaire qui contient des données ou des informations sur les clients et vise à \nobtenir des informations utiles à partir de ces données et à prédire si un nouveau \nclient acceptera ou non une offre de dépôt à terme.")
    st.write("Le **compte dépôt à terme** désigne une somme d'argent mise en dépôt et bloquée sur un compte bancaire. Productive d'intérêts, cette somme ne peut être retirée par son propriétaire qu'après un certain laps de temps fixé à l'avance dans un contrat, signé par les personnes concernées par le dépôt.")
    
    st.write("Pour ce projet, il faudra d’abord effectuer une analyse visuelle et statistique des facteurs pouvant expliquer le lien entre les données personnelles du client (âge, statut marital, quantité d’argent placé dans la banque, nombre de fois que le client a été contacté, etc.) et la variable cible “Est-ce que le client a souscrit au dépôt à terme ?”. ")
    
    
    
    
    
    data = st.checkbox('Afficher le dataset')
    if data :
        st.dataframe(df)
    
    var = st.checkbox("Afficher la description des variables numeriques du dataset")
    if var:
        st.dataframe(df.describe())
    
    val = st.checkbox("Afficher le nombre de valeurs manquantes pour chaque colonne ")
    if val:
        st.dataframe(df.isna().sum())
    
        
        
    
  





if page == 'Dataviz':
    st.title('Visualisation des données')

    st.image('image5.jpg')
    graphique = ["Visualisation des différentes variables catégorielles", "Countplots des variables catégorielles en fonction de la variable « deposit »", 
                 "Analyse des valeurs aberrantes" ]
    radio2 = st.radio("Visualisation des données", graphique)
    if radio2=="Visualisation des différentes variables catégorielles":
        st.subheader("Visualisation des variables catégorielles")
        fig = plt.figure(figsize=(13,80))
        

        
        plotnumber =1
        for categorical_feature in df_cat:
            ax = plt.subplot(12,3,plotnumber)
            sns.countplot(y=categorical_feature,data=df)
            plt.xlabel(categorical_feature)
            plt.title(categorical_feature)
            plotnumber+=1
        st.pyplot(fig)
        
        st.subheader("Observations : ")
        st.write("-Le nombre de clients ayant un poste de travail en management est très élevé, tandis que le nombre de femmes de ménage est très bas.")
        st.write("-Le nombre de clients mariés est élevé, en revanche, les divorcés sont moins nombreux.")
        st.write("-La plupart des clients ont un niveau scolaire secondaire.")
        st.markdown("-La variable **default** n'est pas importante dans notre dataset car le nombre de **non** est très élevé (à plus de 97%) par rapport au **oui**.")
        st.write("-On remarque que le nombre de données du mois de mai sont élevées et basses en décembre.")
    
    if radio2=="Countplots des variables catégorielles en fonction de la variable « deposit »":
        st.subheader("Countplots des variables catégorielles en fonction de la variable « deposit »")
        df_cat1 = df_cat.drop(["job"])
        fig1 = plt.figure(figsize=(13,15))
        
        for i,cat_fea in enumerate(df_cat1):
            
            plt.subplot(4,2,i+1)
            sns.countplot(x=cat_fea,hue='deposit',data=df,edgecolor="black")
            plt.title("Countplot of {}  by deposit".format(cat_fea))
            plt.tight_layout()    
            plt.show()
        
        
        st.pyplot(fig1)
        fig2 = plt.figure(figsize=[14,5])
        sns.countplot(x='job', hue='deposit',edgecolor="black",data=df)
        plt.title("Countplot of job by deposit");
        st.pyplot(fig2)
        st.subheader("Observations : ")
        
        st.write("-Un client célibataire est plus susceptible de souscrire au dépôt qu'une personne mariée.")
        st.write("-Les clients sans prêt immobilier montre plus d’intérêt à souscrire au dépôt à terme.")
        st.write("-Lorsque le contact est effectué via le **cellulaire** le client a plus de chances de souscrire au  **dépôt à terme** alors que le **unknown** l'ait moins.")
        st.write("-La plupart des appels ont été effectués au mois de mai (environ 2500), tandis qu'au mois de décembre, c'est beaucoup moins (moins de 200)")
        st.write("-Les retraités ont plus tendance à souscrire au dépôt à terme.")
        
        
        
        
        
        
    if radio2=="Analyse des valeurs aberrantes":
        st.subheader("Distribution des variables numeriques continues")
        st.write("On continue notre analyse de datavisualisation avec les variables numériques en affichant leurs distributions : ")
        fig3= plt.figure(figsize=(18,80))
        plotnumber =1
        for continuous_feat in df_num:
            ax = plt.subplot(12,3,plotnumber)
            sns.distplot(df[continuous_feat])
            plt.xlabel(continuous_feat)
            plotnumber+=1
        st.pyplot(fig3)
        st.subheader("Observations : ")
        st.write("-La variable **age** et **day** ont des distributions normales.")
        st.write("-Tandis que les variables **balance**, **duration**, **campaign**, **pdays** et **previous** sont fortement biaisés vers la gauche et semblent avoir des valeurs aberrantes.")
        
        
        
        st.subheader("Boxplot des variables numeriques")
        st.write("Et pour justement déterminer les valeurs aberrantes, on va tracer les boxplots pour les variables numériques :")
        fig4= plt.figure(figsize=(18,80))
        plotnumber =1
        for numerical_feat in df_num:
            ax = plt.subplot(12,3,plotnumber)
            sns.boxplot(df[numerical_feat])
            plt.xlabel(numerical_feat)            
            plotnumber+=1
        
        st.pyplot(fig4)   
        st.subheader("Observation : ")
        
        st.write("On constate que les variables **age**, **balance**, **duration**, **campaign**, **pdays** et **previous** ont bien des valeurs aberrantes qu’on va devoir traiter dans la partie suivante.")
        
        
        
        
        
        
        
        
        
        
        

df= df.drop('default',axis=1)
df= df.drop('pdays',axis=1)
df = df[df['campaign'] < 33]

df = df[df['previous'] < 31]

cat_columns = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
for col in  cat_columns:
    df = pd.concat([df.drop(col, axis=1),pd.get_dummies(df[col], prefix=col, prefix_sep='_',drop_first=True, dummy_na=False)], axis=1)

df = df.replace({"yes":1, "no":0})

target=df["deposit"]
data=df.drop("deposit", axis=1)





X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")
 
if page == 'Modélisation':
    
    st.title('Modélisation')
    st.image('image6.jpg')
    selectbox1 = st.selectbox('Choix du modèle', ('DecisionTreeClassifier', 'RandomForestClassifier', 'AdaBoostClassifier'))

    if selectbox1=='DecisionTreeClassifier':
        st.title('DecisionTreeClassifier')
        from sklearn.tree import DecisionTreeClassifier
        dt_clf = load('dtc.joblib')
        y_pred = dt_clf.predict(X_test)
        
        st.write("Le score du model est",dt_clf.score(X_test, y_test))
        st.subheader('Matrice de confusion')
        st.image('dtc.jpg')

        
        
    if selectbox1=='RandomForestClassifier':
        st.title('RandomForestClassifier')
        from sklearn.ensemble import RandomForestClassifier

         

        clf = load('rfc.joblib') 

        y_pred = clf.predict(X_test)
        

        st.write("Le score du model est", clf.score(X_test, y_test))
        st.subheader('Matrice de confusion')

        st.image('rfc.jpg')
        ms = st.checkbox("Afficher le model score")
         
        if ms:
            ms = load('ms.joblib') 
            st.write(ms)
            st.write("Le moyenne des scores est ", ms.mean())
        gr = st.checkbox("Afficher la grille de recherche")
        if gr:
            st.header("Grille de recherche")
            grid_clf=load('grid_clf.joblib')
            grille = load('grille.joblib')
            st.write(pd.DataFrame.from_dict(grille.cv_results_).loc[:,['params', 'mean_test_score']]) 
            gp= st.checkbox("La meilleure combinaison de paramètres")
            if gp :
                
                st.write(grid_clf.best_params_)
       
        
    if selectbox1=='AdaBoostClassifier':
        st.title('AdaBoostClassifier')
        

        ac = load('adc.joblib') 
        st.write("Le score du model est",ac.score(X_test, y_test))
        st.subheader('Matrice de confusion')

        st.image('ac.jpg')
        
        
       
        
        
        
if page == 'Conclusion et perspective': 
    st.title('Conclusion et perspective')
    st.image('image8.jpg')
    st.subheader("***Conclusion***")
    st.write("On constate que pour rendre les données utiles il faut passer par des étapes : l’exploration, la visualisation, le nettoyage, le preprocessing, la modélisation.")
    st.write("Dans notre projet avant d’en arriver aux modèles de machine learning nous avons en effet passé par l’exploration des données notamment les différentes variables numériques et catégorielle. Nous avons visualisé ces données pour avoir des idées bien précises sur les données de notre projet. Enfin grâce au nettoyage nous avons améliorer la qualité de nos data.")
    
    st.subheader("***Perspective***")
    st.write("Et pour aller encore plus loin, il serait intéressant de procéder à l'interprétabilité des modèles de machine learning testés dans le but de démontrer et comprendre comment ces derniers fonctionnent et fournir des informations sur les données utilisées.")
    
    
    
    
    
    
    
st.sidebar.subheader("Projet Data Analyst - Promotion Bootcamp Septembre 2022")    
        
st.sidebar.subheader("Participants:")
st.sidebar.text("Agbetoho SILETE \nStephane N'da \nYanis KADRI")
