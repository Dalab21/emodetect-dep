import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pickle
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import os
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status
#from fastapi.openapi.utils import OpenAPIContact, OpenAPILicense
#from fastapi.openapi import CustomOpenAPI
from fastapi.openapi.models import Contact, License
from fastapi.staticfiles import StaticFiles
#
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from tempfile import NamedTemporaryFile
from typing import Optional
from datetime import timedelta
######
from schemas import Token, User, User
from auth import authenticate_user, create_access_token, get_current_user, hash_password, verify_password
from database import get_db
from models import User as DBUser
from auth import ACCESS_TOKEN_EXPIRE_MINUTES 
########



# Instanciation de FastAPI
app = FastAPI(
    title="detectEmoAPI",
    description="""
    API de détection d'émotions basée sur des images, detecEmoAPI utilise des techniques de vision par ordinateur et de machine learning. 
    Elle permet de prédire les émotions d'une personne en analysant les traits de son visage. 
    DetectEmoAPI intègre la fonctionnalité de gestion des utilisateurs avec authentification par tokens.
    
    Tâches de l'API
    detectEmoAPI propose les fonctionnalités suivantes :
    
    1. Authentification des utilisateurs
       - /token: Endpoint pour la génération de tokens d'accès après authentification.
         Génération de tokens d'accès (JWT) sécurisés pour les sessions
         Vérification des tokens en s'assurant que seules les demandes authentifiées peuvent accéder à l'endpoint protégé de detectEmoAPI.
    
    2. Endpoints de l'API
       - /predict: Endpoint protégé pour prédire l'émotion à partir d'une image uploadée par un utilisateur authentifié.
       - /predict_with_image: Endpoint pour prédire l'humeur de l'utilisateur et lui renvoyer l'image taguée des informations liées à la détection.
    """,
   version="1.0.1",
   terms_of_service="#",
   contact=Contact(
        name="Seydou DAOU",
        url="http://www.dalab21.com/contact/",
        email="daouseydou@gmail.com",
    ),
    license_info=License(
        name="Apache 2.0",
        url="https://www.apache.org/licenses/LICENSE-2.0.html",
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)


# Définition du schéma OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Si le folder 'images_pred' n'existe pas il est créé
if not os.path.exists('image_pred'):
    os.makedirs('image_pred')

# Chargement du modèle
with open('best_model/lr_rang_2.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialisation des outils de MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


# Route pour la génération des tokens
@app.post("/token", response_model=Token, tags=["Authentification : permettre à l'utilisateur de s'authentifier"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Nom d'utilisateur ou mot de passe incorrect",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}



# Fonction pour prédire l'émotion
def predict_emotion(image_path: str):
    """
     Endpoint de l'API
       - /predict: Endpoint protégé pour prédire l'émotion à partir d'une image uploadée par un utilisateur authentifié.
    """
    # Lecture de l'image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erreur : l'image n'a pas été correctement lue depuis {image_path}")
        return None, None, None, None

    # Conversion image en RGB pour MediaPipe
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        results = face_mesh.process(image)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                face = face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
                X = pd.DataFrame([face_row])
                face_class = model.predict(X)[0]
                if hasattr(model, 'predict_proba'):
                    face_prob = model.predict_proba(X)[0]
                else:
                    face_prob = None
                return face_class, face_prob, face_landmarks, img
    return None, None, None, None

# Route protégée par JWT
@app.post("/predict", tags=["Prédiction simple : Obtenir l'humeur de l'utilisateur"])
async def predict(file: UploadFile = File(...)): #async def predict(file: UploadFile = File(...), current_user: User = Depends(get_current_user)): 
    """
       Endpoint de l'API
#        - /predict: Endpoint protégé pour prédire l'émotion à partir d'une image uploadée par un utilisateur authentifié.
    """
    with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    print(f"Chemin du fichier temporaire : {temp_file_path}")
    
    face_class, face_prob, _, _ = predict_emotion(temp_file_path)
    if face_class is None:
        return JSONResponse(content={"erreur": "Pas de visage détecté"}, status_code=400)
    return {"emotion": face_class.split(' ')[0]}


# Route nécessitant un token pour acceder à ce service
@app.post("/predict_ep", tags=["Prédiction détaillée: Obtenir l'humeur de l'utilisateur et les détails de prédiction"])
async def predict_ep(file: UploadFile = File(...)): 
    """
       Endpoint de l'API
       - /predict_ep: Endpoint pour obtenir une prédiction détaillée de l'émotion.Une authentification est nécessaire pour obtenir ce service.
    """
    with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    face_class, face_prob, _, _ = predict_emotion(temp_file_path)
    if face_class is None:
        return JSONResponse(content={"erreur": "Pas de visage détecté"}, status_code=400)
    
    # Formatage de la probabilité
    prob_dict = {f"{i}": float(prob) for i, prob in enumerate(face_prob)} if face_prob is not None else None

    return {"emotion": face_class, "prob": prob_dict}


# Route de prédiction avec image taguée des information de prédiction. Nécessite une authentification
@app.post("/predict_with_image", tags=["Prédiction avancée : Obtenir l'humeur et les détails sur image de l'utilisateur"])
async def predict_with_image(file: UploadFile = File(...)):
    """
        Endpoint de l'API
       - /predict_with_image: Endpoint pour prédire l'émotion et retourner l'image annotée avec les informations sur l'émotion détectée. Elle est ensuite retournée par l'API et peut etre téléchargée par l'utilisateur.
    """
    with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name
    
    face_class, face_prob, face_landmarks, img = predict_emotion(temp_file_path)
    
    if face_class is None:
        return JSONResponse(content={"erreur": "Pas de visage détecté"}, status_code=400)
    
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if face_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 150, 25), thickness=1, circle_radius=1),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(155, 250, 40), thickness=1, circle_radius=1)
        )
        coords = (int(face_landmarks.landmark[0].x * img.shape[1]), int(face_landmarks.landmark[0].y * img.shape[0]))
        cv2.rectangle(image, (0, 0), (250, 60), (145, 217, 106), -1)
        cv2.putText(image, 'LABEL', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, face_class.split(' ')[0], (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, 'PROB', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        if face_prob is not None:
            cv2.putText(image, str(round(face_prob[np.argmax(face_prob)], 2)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Créer le folder 'data_pred' si il n'existe pas
    if not os.path.exists('image_pred'):
        os.makedirs('image_pred')
    
    annotated_image_path = os.path.join('image_pred', f'{face_class.split(" ")[0]}_{round(face_prob[np.argmax(face_prob)], 2)}.jpg')
    
    # Save de l'image annotée 
    cv2.imwrite(annotated_image_path, image)
    
    if not os.path.exists(annotated_image_path):
        return JSONResponse(content={"erreur": "Sauvegarde de l'image annotée a échoué"}, status_code=500)
    
    emotion = face_class.split(' ')[0]
    
    return {
        "emotion": emotion,
        "image": FileResponse(annotated_image_path, media_type='image/jpeg', filename=os.path.basename(annotated_image_path))
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
