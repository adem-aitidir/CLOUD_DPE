# Databricks notebook source
dbutils.fs.mount(
source = "wasbs://containergroupe6@comptestockagegroupe6.blob.core.windows.net",
mount_point = "/mnt/mntproff",
extra_configs = {"fs.azure.account.key.comptestockagegroupe6.blob.core.windows.net":dbutils.secrets.get(scope = "key-vault-secret-databricks", key = "coffre-groupe6")}) 


# COMMAND ----------

display(dbutils.fs.ls("/mnt/mntproff"))

# COMMAND ----------

import pandas as pd
df = pd.read_csv('/dbfs/mnt/mntproff/train.csv')
df.head()

