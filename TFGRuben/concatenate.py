import pandas as pd

# Cargar los dos CSVs
df_completo = pd.read_csv("../Bayes/filtered_uncertainty/uncertainty_train.csv")  # El que tiene 'Grado_ISUP'
df_parcial = pd.read_csv("./filtered_uncertainty/uncertainty_train.csv")        # El que quieres completar

# Hacer merge por 'Image Name'
df_merged = df_parcial.merge(df_completo[['Image Name', 'Grado_ISUP']], on='Image Name', how='left')

# Guardar el nuevo CSV
df_merged.to_csv("./filtered_uncertainty/uncertainty_train.csv", index=False)
