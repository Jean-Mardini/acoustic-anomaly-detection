from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "Acoustic Anomaly Detection API"
    app_version: str = "0.1.0"


settings = Settings()
