from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Variables d'environnement BDD
dbname = os.getenv("DB_NAME")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT", 5432)


# Connexion à la BDD
SQLALCHEMY_DATABASE_URL = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

# Moteur de BDD
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Création de session pour la BDD 
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Création de la classe de base pour les modèles de la BDD
Base = declarative_base()

# onction de récupération de la session de la BDD
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
