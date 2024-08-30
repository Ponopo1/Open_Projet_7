import uvicorn
from fastapi import FastAPI
import pandas as pd
import joblib
from pydantic import BaseModel

# Import model 
loaded_model = joblib.load('C:/Users/alex3/Desktop/Openclassrooms/Projet_7/best_Random Forest_2024-07-26.joblib')

# 2. Lire le fichier CSV dans un DataFrame en ignorant la colonne d'index si nécessaire
csv_path = 'C:/Users/alex3/Desktop/Openclassrooms/Projet_7/X_test.csv'
df_api= pd.read_csv(csv_path, index_col="ID_CLIENT")
df_api.index = df_api.index.astype(int)

# Instance API
app = FastAPI()
@app.get("/acceuil")
def great():
   return {"message":"Bonjour"}


@app.get("/predict")
def predict(ID_CLIENT) :
   ID_CLIENT = int(ID_CLIENT)
   # Vérifier si l'ID_CLIENT existe dans le DataFrame
   if ID_CLIENT in df_api.index:
      # Extraire les données du client
      client_data = df_api.loc[ID_CLIENT].values.reshape(1, -1)
      
      # Tester le modèle sur ce ID_CLIENT
      prediction = loaded_model.predict(client_data)
      prediction = prediction.tolist()[0]
      # Afficher la prédiction
      return {'prediction': prediction}
   else:
      return 'Manquant'
   


print(predict(232250))

if __name__ == "__main__":
   uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)