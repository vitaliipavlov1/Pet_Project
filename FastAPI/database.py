from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from environs import Env
from fastapi import Depends


# Cargando variables de entorno
env = Env()
env.read_env() 

DATABASE_URL = env.str("DATABASE_URL")


engine = create_engine(DATABASE_URL)       # Creando engine SQLAlchemy

# Creando fábrica de sesiones
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)       # Creando fábrica de sesiones


Base = declarative_base()       # Clase base para modelos


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
