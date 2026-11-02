from pydantic_settings import BaseSettings, SettingsConfigDict

from functools import lru_cache

from typing import Any
class Config(BaseSettings):

	model_config = SettingsConfigDict(env_file='.env')

	model_name: str = "Facenet512"
	vector_size: int = 512
	collection_name: str = "test_local_db"





@lru_cache
def get_settings() -> Any:
	return Config()