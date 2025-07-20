from dataclasses import dataclass
from environs import Env

@dataclass
class Settings:
    database_url: str
    secret_key: str
    debug: bool

def get_settings() -> Settings:
    env = Env()
    try:
        env.read_env()  # Intentando cargar desde .env localmente, si está disponible.
    except Exception:
        pass  # Ignorar los errores si no hay ningún archivo.
    return Settings(
        database_url=env("DATABASE_URL"),
        secret_key=env("SECRET_KEY"),
        debug=env.bool("DEBUG", False)
    )
