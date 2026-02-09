"""
AI Pipeline Configuration
Supports multiple LLM providers including local Ollama
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class AISettings(BaseSettings):
    """AI Service Configuration"""

    # ===================
    # LLM Provider Selection
    # ===================
    LLM_PROVIDER: str = "ollama"  # ollama, openai, anthropic, local

    # ===================
    # Ollama (Local LLM - NO API KEY)
    # ===================
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1"  # or mistral, mixtral, phi-3, gemma2
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"

    # ===================
    # OpenAI (API-based)
    # ===================
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4-turbo-preview"

    # ===================
    # Anthropic Claude (API-based)
    # ===================
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-3-opus-20240229"

    # ===================
    # TTS Provider Selection
    # ===================
    TTS_PROVIDER: str = "local"  # local, elevenlabs, openai, azure

    # ElevenLabs (Needs API Key)
    ELEVENLABS_API_KEY: Optional[str] = None

    # Local TTS (No API Key - pyttsx3/gTTS)
    LOCAL_TTS_ENGINE: str = "gtts"  # gtts, pyttsx3

    # ===================
    # Vector Database
    # ===================
    VECTOR_DB: str = "chroma"  # chroma, pinecone, weaviate

    # ChromaDB (Local - NO API KEY)
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_PERSIST_DIR: str = "./data/chroma"

    # Pinecone (Cloud - Needs API Key)
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: str = "us-east1-gcp"

    # ===================
    # Supabase
    # ===================
    SUPABASE_URL: Optional[str] = None
    SUPABASE_KEY: Optional[str] = None

    # ===================
    # LangSmith (Monitoring)
    # ===================
    LANGSMITH_API_KEY: Optional[str] = None
    LANGSMITH_PROJECT: str = "vidcv-ai"
    LANGSMITH_TRACING: bool = True

    # ===================
    # Avatar APIs
    # ===================
    HEYGEN_API_KEY: Optional[str] = None
    DID_API_KEY: Optional[str] = None

    # ===================
    # Storage
    # ===================
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin123"
    MINIO_BUCKET: str = "vidcv"

    # ===================
    # Redis
    # ===================
    REDIS_URL: str = "redis://localhost:6379/2"

    # ===================
    # Kafka
    # ===================
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = AISettings()


# ===================
# LLM Factory
# ===================
def get_llm():
    """Get LLM based on configuration"""
    if settings.LLM_PROVIDER == "ollama":
        from langchain_community.llms import Ollama
        return Ollama(
            base_url=settings.OLLAMA_HOST,
            model=settings.OLLAMA_MODEL
        )
    elif settings.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL
        )
    elif settings.LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            api_key=settings.ANTHROPIC_API_KEY,
            model=settings.ANTHROPIC_MODEL
        )
    else:
        raise ValueError(f"Unknown LLM provider: {settings.LLM_PROVIDER}")


def get_embeddings():
    """Get embedding model based on configuration"""
    if settings.LLM_PROVIDER == "ollama":
        from langchain_community.embeddings import OllamaEmbeddings
        return OllamaEmbeddings(
            base_url=settings.OLLAMA_HOST,
            model=settings.OLLAMA_EMBED_MODEL
        )
    elif settings.LLM_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
    else:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def get_vector_store():
    """Get vector store based on configuration"""
    embeddings = get_embeddings()

    if settings.VECTOR_DB == "chroma":
        from langchain_community.vectorstores import Chroma
        return Chroma(
            persist_directory=settings.CHROMA_PERSIST_DIR,
            embedding_function=embeddings
        )
    elif settings.VECTOR_DB == "pinecone":
        from langchain_pinecone import PineconeVectorStore
        return PineconeVectorStore(
            index_name="vidcv",
            embedding=embeddings
        )
    else:
        raise ValueError(f"Unknown vector DB: {settings.VECTOR_DB}")
