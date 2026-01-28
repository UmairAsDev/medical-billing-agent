import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    
    ENV: str = "dev" 
    PROJECT_NAME: str = "Medical Billing Agent"
    LOG_LEVEL: str = "INFO"
    

    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    BEDROCK_SERVICE_NAME: str = "bedrock-runtime"
    AWS_REGION: str = "us-east-1"
    

    MODEL_ID: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL_ID: str = "gpt-5.2"
    

    BEDROCK_GUARDRAIL_ID: Optional[str] = None
    BEDROCK_GUARDRAIL_VERSION: str = "DRAFT"


    CHROMA_DB_PATH: str = "./chroma_data"
    CHROMA_URL: Optional[str] = None

    MAX_REVISIONS: int = 3


    model_config = SettingsConfigDict(
        env_file= ".env",
        env_file_encoding='utf-8',
        extra='ignore' 
    )


settings = Settings() #type: ignore