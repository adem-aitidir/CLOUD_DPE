# Databricks notebook source
import pandas as pd
df = pd.read_csv('/dbfs/mnt/mntproff/train.csv')

# COMMAND ----------

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

from sklearn.preprocessing import LabelEncoder

# Suppression des lignes où la colonne cible 'Etiquette_DPE' est nulle
df_model.dropna(subset=['Etiquette_DPE'], inplace=True)

# Encodage par étiquettes pour les variables catégorielles
label_encoder = LabelEncoder()
for col in df_model.select_dtypes(include=['object', 'category']).columns:
    df_model[col] = label_encoder.fit_transform(df_model[col])

# Affichage des premières lignes du DataFrame nettoyé
# Affichage des premières lignes du DataFrame nettoyé
df_model.head()

# COMMAND ----------

sdf = spark.createDataFrame(df_model)

# COMMAND ----------

sdf.write.saveAsTable("Data_EDF")
