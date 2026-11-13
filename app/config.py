from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str | None = None
    vertex_ai_project: str | None = None
    vertex_ai_location: str = "global"
    gemini_api_version: str | None = None
    gemini_model: str = "gemini-flash-latest"
    supabase_url: str
    supabase_service_key: str
    vision_api_key: str
    session_ttl_hours: int = 24
    port: int = 8080

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @model_validator(mode="after")
    def require_gemini_credentials(self) -> "Settings":
        if not self.vertex_ai_project and not self.gemini_api_key:
            raise ValueError(
                "Set VERTEX_AI_PROJECT for ADC or GEMINI_API_KEY for the Developer API."
            )
        return self


settings = Settings()
