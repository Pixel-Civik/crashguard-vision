from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str
    gemini_model: str = "gemini-flash-latest"
    supabase_url: str
    supabase_service_key: str
    vision_api_key: str
    session_ttl_hours: int = 24
    port: int = 8080

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
