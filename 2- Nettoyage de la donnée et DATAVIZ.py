# Databricks notebook source
import pandas as pd

df = pd.read_csv('/dbfs/mnt/mntproff/train.csv')


# COMMAND ----------


import pandas as pd
!pip install prettytable

from prettytable import PrettyTable

# Créer une table pour afficher les noms des colonnes
table_columns = PrettyTable()
table_columns.field_names = ["Index", "Nom de la Colonne"]
for i, column in enumerate(df.columns):
    table_columns.add_row([i, column])
print("Noms des Colonnes :")
print(table_columns)

# Afficher les colonnes numériques et qualitatives
# Créer des tables pour afficher les colonnes numériques et qualitatives
numeric_columns_table = PrettyTable()
numeric_columns_table.field_names = ["Index", "Colonne Numérique"]
numeric_columns = df.select_dtypes(include='number').columns
for i, column in enumerate(numeric_columns):
    numeric_columns_table.add_row([i, column])

qualitative_columns_table = PrettyTable()
qualitative_columns_table.field_names = ["Index", "Colonne Qualitative"]
qualitative_columns = df.select_dtypes(exclude='number').columns
for i, column in enumerate(qualitative_columns):
    qualitative_columns_table.add_row([i, column])

# Afficher les tables
print("Colonnes Numériques :")
print(numeric_columns_table)
print("\nColonnes Qualitatives :")
print(qualitative_columns_table)

# Nombre de valeurs manquantes par colonnes
missing_values_table = PrettyTable()
missing_values_table.field_names = ["Nom de la Colonne", "Valeurs Manquantes"]
for column in df.columns:
    missing_values_table.add_row([column, df[column].isnull().sum()])
print("\nValeurs Manquantes par Colonne :")
print(missing_values_table)

# Nombre de variables uniques par colonnes
unique_values_table = PrettyTable()
unique_values_table.field_names = ["Nom de la Colonne", "Valeurs Uniques"]
for column in df.columns:
    unique_values_table.add_row([column, df[column].nunique()])
print("\nValeurs Uniques par Colonne :")
print(unique_values_table)

# COMMAND ----------

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Sélection des colonnes pertinentes, y compris la colonne cible pour la prédiction
colonnes_selectionnees = [
    'Année_construction',
    'Qualité_isolation_enveloppe',
    'Qualité_isolation_menuiseries',
    'Qualité_isolation_murs',
    'Qualité_isolation_plancher_bas',
    'Surface_habitable_logement',
    'Conso_5_usages_é_finale',
    'Type_bâtiment',
    'Etiquette_DPE'  # Cible pour la prédiction
]
# Filtrage du dataset pour ne garder que les colonnes sélectionnées
df_model = df[colonnes_selectionnees]

# Traitement des valeurs manquantes pour les colonnes numériques
for col in df_model.select_dtypes(include='number').columns:
    if df_model[col].isnull().mean() < 0.20:
        df_model[col].fillna(0, inplace=True)
    else:
        df_model[col].dropna(inplace=True)



# COMMAND ----------

# MAGIC %md
# MAGIC DATAVIZ

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.histplot(df_model['Année_construction'], kde=False, bins=30)
plt.title('Distribution des Années de Construction')
plt.xlabel('Année de Construction')
plt.ylabel('Nombre de Bâtiments')
plt.show()

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

colonnes_selectionnees = [
    'Année_construction',
    'Qualité_isolation_enveloppe',
    'Qualité_isolation_menuiseries',
    'Qualité_isolation_murs',
    'Qualité_isolation_plancher_bas',
    'Surface_habitable_logement',
    'Conso_5_usages_é_finale',
    'Type_bâtiment',
    'Etiquette_DPE'  # Cible pour la prédiction
]

# Filtrer les données pour ces colonnes
filtered_data = df[colonnes_selectionnees]

# Distinguer les variables numériques des variables catégorielles
variables_numeriques = filtered_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
variables_categorielles = [col for col in colonnes_selectionnees if col not in variables_numeriques]

# Tracer des histogrammes pour les variables numériques
for col in variables_numeriques:
    plt.figure(figsize=(10, 6))
    sns.histplot(filtered_data[col], kde=True)
    plt.title(f'Histogramme de {col}')
    plt.show()

# Tracer des diagrammes en barres pour les variables catégorielles
for col in variables_categorielles:
    plt.figure(figsize=(10, 6))
    sns.countplot(y=col, data=filtered_data)
    plt.title(f'Diagramme en barres de {col}')
    plt.show()

# COMMAND ----------

import plotly.express as px

# df_model est votre DataFrame
fig = px.histogram(df_model, x='Année_construction')
fig.show()


# COMMAND ----------



