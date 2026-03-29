from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    whisper_model: str = "base"
    hf_token: str = ""
    log_level: str = "INFO"

    class Config:
        env_file = ".env"


settings = Settings()
