import pandas as pd

df = pd.read_csv('teldata.csv')

# Dimensiones (filas, columnas)
print(df.shape)

# Primeras filas para visualizar datos
print(df.head())

# Tipos de datos de cada columna
print(df.dtypes)