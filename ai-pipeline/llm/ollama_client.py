"""
Ollama Client - Local LLM without API keys
"""
import os
from typing import Optional, List, Dict, Any
import ollama
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


class OllamaClient:
    """
    Ollama client for local LLM inference
    NO API KEYS REQUIRED - runs completely locally
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llama3.1",
        embed_model: str = "nomic-embed-text"
    ):
        self.host = host
        self.model = model
        self.embed_model = embed_model

        # Initialize LangChain components
        self.llm = Ollama(base_url=host, model=model)
        self.embeddings = OllamaEmbeddings(base_url=host, model=embed_model)

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7
    ) -> str:
        """Generate text completion"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={"temperature": temperature}
        )

        return response["message"]["content"]

    async def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate JSON output"""
        json_prompt = f"{prompt}\n\nRespond ONLY with valid JSON, no other text."

        response = await self.generate(
            prompt=json_prompt,
            system=system,
            temperature=0.1  # Lower temp for structured output
        )

        # Parse JSON from response
        import json
        # Find JSON in response
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])
        raise ValueError("Could not parse JSON from response")

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        embeddings = []
        for text in texts:
            response = ollama.embeddings(
                model=self.embed_model,
                prompt=text
            )
            embeddings.append(response["embedding"])
        return embeddings

    async def embed_query(self, text: str) -> List[float]:
        """Generate embedding for single text"""
        response = ollama.embeddings(
            model=self.embed_model,
            prompt=text
        )
        return response["embedding"]

    def get_langchain_llm(self) -> Ollama:
        """Get LangChain-compatible LLM"""
        return self.llm

    def get_langchain_embeddings(self) -> OllamaEmbeddings:
        """Get LangChain-compatible embeddings"""
        return self.embeddings

    @staticmethod
    async def list_models(host: str = "http://localhost:11434") -> List[str]:
        """List available Ollama models"""
        models = ollama.list()
        return [m["name"] for m in models.get("models", [])]

    @staticmethod
    async def pull_model(model: str, host: str = "http://localhost:11434"):
        """Pull a model from Ollama library"""
        ollama.pull(model)


# Recommended models for VidCV:
RECOMMENDED_MODELS = {
    "llama3.1": "Best overall performance, 8B parameters",
    "mistral": "Fast and efficient, good for CV parsing",
    "mixtral": "MoE model, excellent quality",
    "phi-3": "Microsoft's small but capable model",
    "gemma2": "Google's open model",
    "qwen2": "Alibaba's multilingual model (good for Turkish)",
    "nomic-embed-text": "Best open embedding model",
}
