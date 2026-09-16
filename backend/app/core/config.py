from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration kept in environment variables or a local .env file."""

    app_name: str = "StudyMate AI"
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = ["http://localhost:5173"]
    data_dir: Path = Path(__file__).resolve().parents[2] / "data"
    max_upload_size_mb: int = 50
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size_words: int = 180
    chunk_overlap_words: int = 35

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def extracted_text_dir(self) -> Path:
        return self.data_dir / "extracted"

    @property
    def documents_index_path(self) -> Path:
        return self.data_dir / "documents.json"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
