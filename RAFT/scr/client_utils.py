from abc import ABC
from typing import Any, Optional
import logging
import time
import random
import os
from threading import Lock

# Import cloud APIs
from openai import AzureOpenAI, OpenAI
from groq import Groq
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azure.identity import get_bearer_token_provider

# Import local model dependencies
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Import config utilities
from env_config import read_env_config, set_env
from os import environ, getenv


logger = logging.getLogger("client_utils")

# =============================================
# OpenAI Model Implementation (Generation+Embeddings)
# =============================================
#OpenAI Client for generation
def build_openai_client(env_prefix : str = "COMPLETION", **kwargs: Any) -> OpenAI:
    """
    Build OpenAI client based on the environment variables.
    """

    kwargs = _remove_empty_values(kwargs)
    env = read_env_config(env_prefix)
    with set_env(**env):
        if is_azure():
            auth_args = _get_azure_auth_client_args()
            client = AzureOpenAI(**auth_args, **kwargs)
        else:
            client = OpenAI(**kwargs)
        return client

#OpenAI Embeddings for chunking 
def build_langchain_embeddings(**kwargs: Any) -> OpenAIEmbeddings:
    """
    Build OpenAI embeddings client based on the environment variables.
    """

    kwargs = _remove_empty_values(kwargs)
    env = read_env_config("EMBEDDING")
    with set_env(**env):
        if is_azure():
            auth_args = _get_azure_auth_client_args()
            client = AzureOpenAIEmbeddings(**auth_args, **kwargs)
        else:
            client = OpenAIEmbeddings(**kwargs)
        return client

# =============================================
# Groq Model Implementation (Generation)
# =============================================
def build_groq_client(env_prefix: str = "GROQ", api_key: Optional[str] = None, **kwargs: Any) -> Groq:
    """
    Build Groq client based on the environment variables or provided API key.
    """
    kwargs = _remove_empty_values(kwargs)
    env = read_env_config(env_prefix)
    with set_env(**env):
        if api_key is None:
            api_key = os.environ.get(f"{env_prefix}_API_KEY")
        if not api_key:
            raise ValueError(f"Groq API key not provided and {env_prefix}_API_KEY environment variable is not set")
        client = Groq(api_key=api_key, **kwargs)
        return client

# This function is for Groq, as free Groq API has a rate limit 
class TokenBucket:
    def __init__(self, tokens_per_minute, requests_per_minute):
        self.tokens_per_minute = tokens_per_minute
        self.requests_per_minute = requests_per_minute
        self.tokens = tokens_per_minute
        self.requests = requests_per_minute
        self.last_refill = time.time()
        self.lock = Lock()

    def _refill(self):
        now = time.time()
        time_passed = now - self.last_refill
        self.tokens = min(self.tokens_per_minute, self.tokens + time_passed * (self.tokens_per_minute / 60))
        self.requests = min(self.requests_per_minute, self.requests + time_passed * (self.requests_per_minute / 60))
        self.last_refill = now

    def consume(self, tokens):
        with self.lock:
            self._refill()
            if self.tokens >= tokens and self.requests >= 1:
                self.tokens -= tokens
                self.requests -= 1
                return True
            return False


# =============================================
# Local Model Implementation (Generation)
# =============================================
class LocalLanguageModel:
    def __init__(self, model_name_or_path: str):
        """Handles local model inference (for example Mistral) when GPU is available"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            device_map="auto", 
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        )  

        if self.device.type == "cuda":
            self.model.to(self.device)
            logger.info(f"Model moved to GPU: {self.model.device}")
        else:
            logger.warning("CUDA is not available. Running on CPU.")


    def generate(self, messages: list, **kwargs) -> str:
        try:
            prompt = self.tokenizer.apply_chat_template(messages,tokenize=False)
            #print(f"mistral Prompt: {prompt}") 
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            prompt_length = inputs.input_ids.shape[1]
            max_tokens = kwargs.get('max_tokens', 512)
            temperature = kwargs.get('temperature', 0.7)

            with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=max(temperature, 0.7),  # This change ensures temperature isn't too low
                        pad_token_id=self.tokenizer.eos_token_id,
                        do_sample=True,
                        top_k=50,     # Adding top_k sampling for stability
                        top_p=0.9  # Add this line
                    )
            new_tokens = outputs[0][prompt_length:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error in mistral generation: {str(e)}")
            return ""
            
# =============================================
# Helper Functions
# =============================================

def _remove_empty_values(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}

def _get_azure_auth_client_args() -> dict:
    """Handle Azure OpenAI Keyless, Managed Identity and Key based authentication
    https://techcommunity.microsoft.com/t5/microsoft-developer-community/using-keyless-authentication-with-azure-openai/ba-p/4111521
    """
    client_args = {}
    if getenv("AZURE_OPENAI_KEY"):
        logger.info("Using Azure OpenAI Key based authentication")
        client_args["api_key"] = getenv("AZURE_OPENAI_KEY")
    else:
        if client_id := getenv("AZURE_OPENAI_CLIENT_ID"):
            # Authenticate using a user-assigned managed identity on Azure
            logger.info("Using Azure OpenAI Managed Identity Keyless authentication")
            azure_credential = ManagedIdentityCredential(client_id=client_id)
        else:
            # Authenticate using the default Azure credential chain
            logger.info("Using Azure OpenAI Default Azure Credential Keyless authentication")
            azure_credential = DefaultAzureCredential()

        client_args["azure_ad_token_provider"] = get_bearer_token_provider(
            azure_credential, "https://cognitiveservices.azure.com/.default")
    client_args["api_version"] = getenv("AZURE_OPENAI_API_VERSION") or "2024-02-15-preview"
    client_args["azure_endpoint"] = getenv("AZURE_OPENAI_ENDPOINT")
    client_args["azure_deployment"] = getenv("AZURE_OPENAI_DEPLOYMENT")
    return client_args

def is_azure():
    azure = "AZURE_OPENAI_ENDPOINT" in environ or "AZURE_OPENAI_KEY" in environ or "AZURE_OPENAI_AD_TOKEN" in environ
    if azure:
        logger.debug("Using Azure OpenAI environment variables")
    else:
        logger.debug("Using OpenAI environment variables")
    return azure

def safe_min(a: Any, b: Any) -> Any:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)

def safe_max(a: Any, b: Any) -> Any:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)

class UsageStats:
    def __init__(self) -> None:
        self.start = time.time()
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.total_tokens = 0
        self.end = None
        self.duration = 0
        self.calls = 0

    def __add__(self, other: 'UsageStats') -> 'UsageStats':
        stats = UsageStats()
        stats.start = safe_min(self.start, other.start)
        stats.end = safe_max(self.end, other.end)
        stats.completion_tokens = self.completion_tokens + other.completion_tokens
        stats.prompt_tokens = self.prompt_tokens + other.prompt_tokens
        stats.total_tokens = self.total_tokens + other.total_tokens
        stats.duration = self.duration + other.duration
        stats.calls = self.calls + other.calls
        return stats

class StatsCompleter(ABC):
    def __init__(self, create_func):
        self.create_func = create_func
        self.stats = None
        self.lock = Lock()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        response = self.create_func(*args, **kwds)
        self.lock.acquire()
        try:
            if not self.stats:
                self.stats = UsageStats()
            self.stats.completion_tokens += response.usage.completion_tokens
            self.stats.prompt_tokens += response.usage.prompt_tokens
            self.stats.total_tokens += response.usage.total_tokens
            self.stats.calls += 1
            return response
        finally:
            self.lock.release()
    
    def get_stats_and_reset(self) -> UsageStats:
        self.lock.acquire()
        try:
            end = time.time()
            stats = self.stats
            if stats:
                stats.end = end
                stats.duration = end - self.stats.start
                self.stats = None
            return stats
        finally:
            self.lock.release()

# =============================================
# Unified Chat Interface
# =============================================

class ChatCompleter(StatsCompleter):
    def __init__(self, openai_client: OpenAI, groq_client: Groq, localmodel_client: LocalLanguageModel):
        self.openai_client = openai_client
        self.groq_client = groq_client
        self.localmodel_client = localmodel_client 
        self.token_bucket = TokenBucket(tokens_per_minute=20000, requests_per_minute=15)
        super().__init__(self._chat_completion)

    def _chat_completion(self, model: str, messages: list, **kwargs):
        #gpt models
        if model.startswith("gpt-"):
            response = self.openai_client.chat.completions.create(model=model, messages=messages, **kwargs)
            if response.choices[0].finish_reason != "stop":
                logger.warning(f"Discarding incomplete response. Finish reason: {response.choices[0].finish_reason}")
                return None
        #Mistral model installed on local machine 
        elif "mistral" in model.lower():
            if not self.localmodel_client:
                raise ValueError("Mistral client not initialized")
            response = self.localmodel_client.generate(messages, **kwargs)
        #Other models from Groq
        else:
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    # Estimate token count (this is a simplified estimation)
                    estimated_tokens = sum(len(m['content'].split()) for m in messages) + 100  # Add 100 as buffer

                    # Wait until we can make the request
                    while not self.token_bucket.consume(estimated_tokens):
                        time.sleep(1)

                    response = self.groq_client.chat.completions.create(model=model, messages=messages, **kwargs)
                except groq.RateLimitError as e:
                    if attempt == max_retries - 1:
                        raise
                    wait_time = (2 ** attempt) + random.random()
                    print(f"Rate limit reached. Waiting for {wait_time:.2f} seconds before retrying.")
                    time.sleep(wait_time)
             
        return response

    def __call__(self, model: str, messages: list, **kwargs):
        return self._chat_completion(model, messages, **kwargs)

class CompletionsCompleter(StatsCompleter):
    def __init__(self, client):
        super().__init__(client.completions.create)
