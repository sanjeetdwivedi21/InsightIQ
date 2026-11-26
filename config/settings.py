# config/settings.py
from functools import lru_cache
from pathlib import Path
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings(BaseModel):
    model_config = {"frozen": True}

    # --- LLM ---
    llm_provider: str = Field(default="anthropic", pattern="^(anthropic|openai|nvidia)$")
    llm_model: str = "claude-sonnet-4-5"
    llm_timeout_seconds: int = Field(default=60, ge=5, le=600)
    anthropic_api_key: SecretStr = SecretStr("")
    openai_api_key: SecretStr = SecretStr("")
    nvidia_api_key: SecretStr = SecretStr("")
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"

    # --- Databases ---
    business_db_path: Path = BASE_DIR / "data/processed/insightiq_business.db"
    app_db_path: Path = BASE_DIR / "data/processed/insightiq_app.db"

    # --- Safety limits ---
    sql_row_limit: int = Field(default=1000, ge=1, le=100_000)
    sql_timeout_seconds: int = Field(default=10, ge=1, le=120)
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    def path_env(key: str, default: Path) -> Path:
        value = os.getenv(key)
        return (BASE_DIR / value) if value else default

    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "anthropic"),
        llm_model=os.getenv("LLM_MODEL", "claude-sonnet-4-5"),
        llm_timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", 60)),
        anthropic_api_key=SecretStr(os.getenv("ANTHROPIC_API_KEY", "")),
        openai_api_key=SecretStr(os.getenv("OPENAI_API_KEY", "")),
        nvidia_api_key=SecretStr(os.getenv("NVIDIA_API_KEY", "")),
        nvidia_base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        business_db_path=path_env("BUSINESS_DB_PATH", BASE_DIR / "data/processed/insightiq_business.db"),
        app_db_path=path_env("APP_DB_PATH", BASE_DIR / "data/processed/insightiq_app.db"),
        sql_row_limit=int(os.getenv("SQL_ROW_LIMIT", 1000)),
        sql_timeout_seconds=int(os.getenv("SQL_TIMEOUT_SECONDS", 10)),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )