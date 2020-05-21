#!/usr/bin/env python
# coding: utf-8

# <hr style="height: 4px; color: #839D2D; width: 100%; ">
# 
# # <font color='#61210B'>Formation OpenClassRooms   -   Parcours DATA ANALYST</font>
# 
# <hr style="height: 2px; color: #839D2D; width: 100%; ">
# 
# ## <font color='#38610B'>Projet V - Etude de Marché Internationale</font>
# 
# ### Partie 1 - Constitution Jeu de Données & Nettoyage 
# <u>Les données sont issues des sites web suivants</u> :
# - Organisation des Nations Unies pour l'alimentation et l'agriculture (<a href="http://www.fao.org/faostat/en/#data">FAO</a>)  
# - Banque Mondiale - World Bank Open Data (<a href="https://databank.banquemondiale.org/data/home.aspx">WBD</a>)  
# - Wikipedia (données ISO-3166 Codes pays)
# 
# <u>DataFrames pandas utilisés et critères de téléchargement</u> :  
# * <font color='#8A0808'>DataFrame <strong>pays</strong></font> : Table de correspondance entre les codes pays issus de la FAO, de la WBD, et la norme Iso-3166  
# Constitué avec Excel en consolidant toutes les données pays issus de la FAO, WBD et de Wikipedia pour la norme ISO-3166  
# 
# 
# * <font color='#8A0808'>DataFrame <strong>fao</strong></font> : Année 2013 - Population & Bilans Alimentaires Volaille & Viande (source FAO)   
# Critères de selection : (**Pays** = tous, **Eléments** = tous, **Année** = 2013, **Produits** = Volailles, **Groupe Produits** = Viande (total) )  
# 
# 
# * <font color='#8A0808'>DataFrame <strong>wbdxch</strong></font> : Année 2013 - Achat Vente de Volailles entre France & Autres Pays (en $)   
# (**Pays Acheteur** = tous, **Pays Vendeur** = France, **Année** = 2013, **Produits** = Volailles vivante & Viande Volailles )
# 
# 
# * <font color='#8A0808'>DataFrame <strong>wbdeco</strong></font> Année 2013 - Données Macro-economiques (source WBD)
# 
#   
# * <font color='#013ADF'>DataFrame <strong>gen</strong></font> : Fichier enrichi, nettoyé pour etude de marché
# 
# 
# **<font color='#38610B'>- Date : 15 Avr 2019</font>**  
# Auteur : Frédéric Boissy
# <hr style="height: 4px; color: #839D2D; width: 100%; ">
# 

# In[1]:


# -*- coding: utf8 -*-
import numpy as np
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format   # Nombres avec sepa milliers "," et 2décimales après "."
pd.options.mode.use_inf_as_na = True

import seaborn as sns
import matplotlib as matplt
import matplotlib.pyplot as plt
import scipy as sc
import scipy.stats as scst
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import statsmodels as st
from sklearn import decomposition, preprocessing
from functions import *

from IPython.display import display, Markdown, HTML  # pour gérer un affichage plus joli que la fonction "print"

import time   # Librairie temps pour calculs durée par exemple
trt_start_time = time.time()


# In[2]:


pays = pd.read_excel("DATA/0-Table_Corresp_Pays_ISO_FAO_WBD.xlsx", sheet_name='data_pj5') 
fao = pd.read_excel("DATA/1-FAO_Data_Principales.xlsx", sheet_name='data_pj5') 
wbdxch = pd.read_excel("DATA/2-WBD_Data_CNUCED_FRA_2013.xlsx", sheet_name='data_pj5')
wbdeco = pd.read_excel("DATA/3-WBD_Data_Indic_Dev_PIB_2013.xlsx", sheet_name='data_pj5')


# <hr style="height: 3px; color: #839D2D; width: 100%; ">
# 
# ###  <font color='#61210B'><u>Table de Correspondance Code Pays</u></font> : Dataframe 'pays'
# #### Détails des zones
# 
# - <b>cpays_fao :</b> Code Pays (Source FAO)   
# - <b>pays_fao :</b> Nom Pays (Source FAO)   
# - <b>cpays_wbd :</b> Code Pays (Source WBD)   
# - <b>pays_wbd :</b> Nom Pays (Source WBD)   
# - <b>cpays_iso_n2 :</b> Code Pays Norme ISO - 2 digits numeriques (Source Wikipedia)   
# - <b>cpays_iso_a2 :</b> Code Pays Norme ISO - 2 digits alphanum (Source Wikipedia)   
# - <b>cpays_iso_a3 :</b> Code Pays Norme ISO - 3 digits alphanum (Source Wikipedia)   
# - <b>pays_iso :</b> Nom Pays Norme ISO (Source Wikipedia)   
# - <b>pays_iso_nom_francais :</b> Nom Pays Français Norme ISO (Source Wikipedia)   

# In[3]:


pays.head()


# Pas de nettoyage ici, table brute pour ne perdre aucune informations.

# <hr style="height: 3px; color: #839D2D; width: 100%; ">
# 
# ###  <font color='#61210B'><u>Données Requises Projets</u></font> : Dataframe 'fao'
# #### Détails des zones
# 
# - <b>cpays_fao :</b> Code Pays   
# - <b>pays_fao :</b> nom Pays   
# - <b>pop_2010 :</b> Population en 2010   
# - <b>pop_2013 :</b> Population en 2013   
# - <b>evo_2010_2013 :</b> (%) Evolution population entre 2010 & 2013   
# - <b>qt_disp_(kcal/p/j) :</b> Quantité dispo alimentaire totale pays (en Kcal/pers/jour)   
# - <b>qt_disp_prot_(gr/p/j) :</b> Quantité dispo alim. protéines totale pays (en Grammes/pers/jour)   
# - <b>qt_prot_via_(gr/p/j) :</b> Quantité dispo alim. protéines viande (volaille incluse) pays (en Grammes/pers/jour)   
# - <b>qt_prot_vol_(gr/p/j) :</b> Quantité dispo alim. protéines volaille pays (en Grammes/pers/jour)   
# - <b>ratio_prot_viande :</b> Ratio Qté Proteines Viande (volaille incluse) / Qté Proteines Totale du Pays   
# - <b>ratio_prot_volaille :</b> Ratio Qté Proteines Volaille / Qté Proteines Totale du Pays   
# - <b>qt_exp_via_(t) :</b> Quantité Export viande (volaille incluse) pays (en tonnes/an)   
# - <b>qt_exp_vol_(t) :</b> Quantité Export volaille pays (en tonnes/an)   
# - <b>qt_imp_via_(t) :</b> Quantité Import viande (volaille incluse) pays (en tonnes/an)   
# - <b>qt_imp_vol_(t) :</b> Quantité Import volaille pays (en tonnes/an)   
# - <b>qt_prod_via_(t) :</b> Quantité Production viande (volaille incluse) pays (en tonnes/an)   
# - <b>qt_prod_vol_(t) :</b> Quantité Production volaille pays (en tonnes/an)   
# 

# In[4]:


fao.describe(include='all')


# #### Nettoyage & mise en Forme

# In[5]:


# cas du Soudan => Après 2010, le Soudan s'est divisé en deux parties.
fao[fao['pays_fao'].str.contains('oudan', na=False) == True ]


# Choix de ne garder que les données de 2013.   
# - Donc suppression de la ligne codePays=206
# - Modification des valeurs de populations et d'évolution par application de regle de 3 avec valeur de 2010 (code 206)

# In[6]:


pop1 = fao[fao['cpays_fao'] == 276]['pop_2013'].values
pop2 = fao[fao['cpays_fao'] == 277]['pop_2013'].values
r = pop1/pop2
fao.loc[(fao['cpays_fao'] == 277), 'pop_2010'] = fao.loc[(fao['cpays_fao'] == 206), 'pop_2010'].values / (r+1)
fao.loc[(fao['cpays_fao'] == 276), 'pop_2010'] = fao.loc[(fao['cpays_fao'] == 206), 'pop_2010'].values * r / (r+1)
fao.drop(fao[fao.cpays_fao == 206].index, inplace=True)
fao.loc[(fao['cpays_fao'] == 276), 'evo_2010_2013'] = ((fao[fao['cpays_fao'] == 276]['pop_2013'].values / fao[fao['cpays_fao'] == 276]['pop_2010'].values)-1)*100
fao.loc[(fao['cpays_fao'] == 277), 'evo_2010_2013'] = ((fao[fao['cpays_fao'] == 277]['pop_2013'].values / fao[fao['cpays_fao'] == 277]['pop_2010'].values)-1)*100


# In[7]:


# cas du Soudan => Après 2010, le Soudan s'est divisé en deux parties.
fao[fao['pays_fao'].str.contains('oudan', na=False) == True ]


# In[8]:


fao[fao['qt_disp_(kcal/p/j)'].isna() == True].head()


# In[9]:


# Choix d'ecarter les pays n'ayant pas de valeurs renseignées sur cette variable imposée
fao = fao.dropna(subset=['qt_disp_(kcal/p/j)'])


# In[10]:


fao.describe(include='all')


# Controle du cas de la Chine (vu Etude PJ3)  
# - Données FAO :  Le Code Pays "351" est une agrégation des Codes Pays (41, 96, 128, 214)

# In[11]:


fao[fao['pays_fao'].str.contains("hine")]


# Pas de soucis ici dans notre DataSet

# <hr style="height: 3px; color: #839D2D; width: 100%; ">
# 
# ###  <font color='#61210B'><u>Données Complémentaires - Echanges Commerciaux InterPays</u></font> : Dataframe 'wbdxch'
# #### Détails des zones
# 
# ICI POSTULAT ENTREPRISE FRANCAISE
# 
# - <b>cpays_vnd :</b> Code Pays Vendeur   
# - <b>pays_vnd :</b> Nom Pays Vendeur   
# - <b>cpays_ach :</b> Code Pays Acheteur   
# - <b>pays_ach :</b> Nom Pays Acheteur   
# - <b>live_poultry_tr_value :</b> Valeur Marchande de Volaille vivante (US Dollars) Année 2013   
# - <b>poultry_meat_tr_value :</b> Valeur Marchande de Viande de Volaille (US Dollars) Année 2013   
# - <b>tot_poultry_tr_value :</b> Valeur Marchande de Totale de Volaille (US Dollars) Année 2013   
# - <b>cpays_fao :</b> Code Pays FAO   
# - <b>cpays_wbd :</b> Code Pays WBD   
# 

# In[12]:


wbdxch.head()


# In[13]:


wbdxch.describe(include='all')


# In[14]:


# On ne conserve que les données pour les code_pays de la FAO
wbdxch = wbdxch.dropna(subset=['cpays_fao'])


# <hr style="height: 3px; color: #839D2D; width: 100%; ">
# 
# ###  <font color='#61210B'><u>Données Complémentaires - Macro Economiques</u></font> : Dataframe 'wbdeco'
# #### Détails des zones
# 
# - <b>cpays_wbd :</b> Code Pays (WBD)   
# - <b>pays_wbd :</b> Nom Pays (WBD)   
# - <b>PIB_h_2013($US) :</b> PIB / habitant année 2013 (en US Dollars)   
# - <b>evo_PIB_h_(%annu)_2013 :</b> Evolution / Croissance du PIB en 2013 (%)   
# 

# In[15]:


wbdeco.head()


# In[16]:


wbdeco.describe(include='all')


# In[17]:


# Choix : On ne conserve que les pays contenant des Informations PIB renseignées dans les deux colonnes
wbdeco = wbdeco.dropna()


# <hr style="height: 3px; color: #839D2D; width: 100%; ">
# 
# ###  <font color='#61210B'><u>Constitution du jeu de données Consolidé pour Analyse</u></font> : Dataframe 'gen'
# #### Détails des zones
# 
# <u>A partir du Dataframe "fao" </u> :
# - <b>pays_fao :</b> Nom Pays FAO   
# - <b>evo_2010_2013 :</b> (%) Evolution population entre 2010 & 2013   
# - <b>qt_disp_(kcal/p/j) :</b> Quantité dispo alimentaire totale pays (en Kcal/pers/jour)   
# - <b>qt_disp_prot_(gr/p/j) :</b> Quantité dispo alim. protéines totale pays (en Grammes/pers/jour)   
# - <b>ratio_prot_viande :</b> Ratio Qté Proteines Viande (volaille incluse) / Qté Proteines Totale du Pays   
# 
# 
# <u>A partir du Dataframe "xch" </u> :
# - <b>tot_poultry_tr_value :</b> Valeur Marchande de Totale de Volaille (US Dollars) Année 2013   
# 
# <u>A partir du Dataframe "eco" </u> :
# - <b>PIB_h_2013($US) :</b> PIB / habitant année 2013 (en US Dollars)   
# 
# <u>Zone Calculée</u> :
# - <b>qt_vol_(kg/h/an) :</b> Quantité Volailles Disponible (en Kg/Hab/An)   
# Formule -->   qt_vol_(kg/h/an) = [qt_prod_vol_(t) + qt_imp_vol_(t) - qt_exp_vol_(t)] / pop_2013
# 
# qt_exp_vol_(t) : Quantité Export volaille pays (en tonnes/an)  
# qt_imp_vol_(t) : Quantité Import volaille pays (en tonnes/an)  
# qt_prod_vol_(t) : Quantité Production volaille pays (en tonnes/an)  

# In[18]:


gen = fao[['cpays_fao', 'pays_fao', 'pop_2013', 'evo_2010_2013', 'qt_disp_(kcal/p/j)', 'qt_disp_prot_(gr/p/j)', 
           'ratio_prot_viande', 'qt_prod_vol_(t)', 'qt_imp_vol_(t)', 'qt_exp_vol_(t)']].copy()
# On remplace les Nan par O pour les colonnes qt_prod, qt_imp, qt_exp, car il faut conserver ces lignes
gen = gen.fillna(0)
# Création de la nouvelle colonne
gen['qt_vol_(kg/h/an)'] = (gen['qt_prod_vol_(t)'] + gen['qt_imp_vol_(t)'] - gen ['qt_exp_vol_(t)'])*1000 / gen['pop_2013']
# Suppression des colonnes inutiles (pop et qté utilisées pour le calcul)
gen.drop(columns=['qt_prod_vol_(t)', 'qt_imp_vol_(t)', 'qt_exp_vol_(t)'], inplace=True)
# Renommage des Colonnes
gen.head()


# In[19]:


# Jointure Table Pays pour Recupérer le code pays wbd dans la table de transco
temp = pd.merge(gen,pays[['cpays_fao','cpays_wbd']], on='cpays_fao', how='left')
# Jointure Table eco pour Recupérer les données PIB des pays en 2013
temp = pd.merge(temp,wbdeco[['cpays_wbd','PIB_h_2013($US)']], on='cpays_wbd', how='left')
# Jointure Table echange pour Recupérer les données valeurs de volailles echangees avec la france en 2013
temp = pd.merge(temp,wbdxch[['cpays_wbd', 'tot_poultry_tr_value']], on='cpays_wbd', how='left')
temp.head()


# In[20]:


gen = temp.copy()
# On supprime les colonnes inutiles et on renomme les autres pour faire plus synthétique / propre
gen.drop(columns=['cpays_fao', 'cpays_wbd'], inplace=True)
gen.columns = ['pays', 'population', 'evo_pop_%', 'qt_kcal/h', 'qt_prot_gr/h', 'ratio_prot_viande_%', 'qt_volaille_Kg/h', 
               'PIBh_$', 'tot_poultry_exch_$']
gen.head()


# In[21]:


# On conserve les lignes données nan our live_poultry et poultry_meat car ce ne sont pas des erreurs  (on force à = 0 )
gen['tot_poultry_exch_$'] = gen['tot_poultry_exch_$'].fillna(0)

# On peut supprimer de notre analyse les pays n'ayant pas d'informations de PIB
gen = gen.dropna(subset=['PIBh_$'])
gen.describe(include='all')


# In[22]:


# On supprime aussi la France de notre Jeu de données car nous partons du postulat "entreprise Française"
gen.drop(gen[gen.pays == 'France'].index, inplace=True)


# <hr style="height: 3px; color: #839D2D; width: 100%; ">
# 
# ###  <font color='#61210B'><u>Export du Dataframe Consolidé</u></font> : Dataframe 'gen'
# #### Dans un fichier csv - Pour analyse

# In[23]:


gen.to_csv('DATA/selection_pays.csv', sep=',', encoding='utf-8', index=False)


# In[24]:


dureetotale = round(time.time() - trt_start_time, 5)
print("--- Durée TOTALE du Notebook PJ5 -1- Load&Clean --- ", "%s seconds" % dureetotale)

