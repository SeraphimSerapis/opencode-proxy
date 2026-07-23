"""Typed request and response models for the Ollama-compatible API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class OllamaFunction(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class OllamaToolCall(BaseModel):
    function: OllamaFunction


class OllamaMessage(BaseModel):
    role: str
    content: str | None = None
    images: list[str] | None = None
    tool_calls: list[OllamaToolCall] | None = None
    tool_name: str | None = None
    thinking: str | None = None


class OllamaToolFunction(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class OllamaTool(BaseModel):
    type: str = "function"
    function: OllamaToolFunction


class OllamaChatRequest(BaseModel):
    model: str
    messages: list[OllamaMessage] = Field(default_factory=list)
    tools: list[OllamaTool] | None = None
    format: str | dict[str, Any] | None = None
    options: dict[str, Any] | None = None
    stream: bool = True
    keep_alive: str | int | None = None
    think: bool | None = None


class OllamaChatResponse(BaseModel):
    model: str = ""
    created_at: str = ""
    message: OllamaMessage = Field(default_factory=lambda: OllamaMessage(role="assistant"))
    done: bool = False
    done_reason: str | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None


class OllamaModelDetails(BaseModel):
    parent_model: str = ""
    format: str = "gguf"
    family: str = ""
    families: list[str] | None = None
    parameter_size: str = ""
    quantization_level: str = ""


class OllamaModelInfo(BaseModel):
    name: str
    model: str
    modified_at: str = ""
    size: int = 0
    digest: str = ""
    details: OllamaModelDetails = Field(default_factory=OllamaModelDetails)


class OllamaModelList(BaseModel):
    models: list[OllamaModelInfo] = Field(default_factory=list)


class OllamaShowRequest(BaseModel):
    name: str
    verbose: bool = False


class OllamaShowResponse(BaseModel):
    modelfile: str = ""
    parameters: str = ""
    template: str = ""
    details: OllamaModelDetails = Field(default_factory=OllamaModelDetails)
    model_info: dict[str, Any] = Field(default_factory=dict)


class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str = ""
    suffix: str | None = None
    images: list[str] | None = None
    system: str | None = None
    format: str | dict[str, Any] | None = None
    options: dict[str, Any] | None = None
    stream: bool = True
    raw: bool = False
    keep_alive: str | int | None = None
    think: bool | None = None


class OllamaGenerateResponse(BaseModel):
    model: str = ""
    created_at: str = ""
    response: str = ""
    thinking: str | None = None
    done: bool = False
    done_reason: str | None = None
    context: list[int] | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None


class OllamaEmbedRequest(BaseModel):
    model: str
    input: str | list[str]
    truncate: bool = True
    options: dict[str, Any] | None = None
    keep_alive: str | int | None = None
    dimensions: int | None = None


class OllamaEmbedResponse(BaseModel):
    model: str = ""
    embeddings: list[list[float]] = Field(default_factory=list)
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None


class OllamaEmbeddingsRequest(BaseModel):
    model: str
    prompt: str
    options: dict[str, Any] | None = None
    keep_alive: str | int | None = None


class OllamaEmbeddingsResponse(BaseModel):
    model: str = ""
    embedding: list[float] = Field(default_factory=list)


class OllamaRunningModelInfo(BaseModel):
    name: str
    model: str
    size: int = 0
    digest: str = ""
    details: OllamaModelDetails = Field(default_factory=OllamaModelDetails)
    expires_at: str = ""
    size_vram: int = 0


class OllamaRunningModels(BaseModel):
    models: list[OllamaRunningModelInfo] = Field(default_factory=list)


class OllamaPullRequest(BaseModel):
    name: str
    insecure: bool = False
    stream: bool = True


class OllamaPullResponse(BaseModel):
    status: str = "success"


class OllamaCopyRequest(BaseModel):
    source: str
    destination: str


class OllamaDeleteRequest(BaseModel):
    name: str


class OllamaCreateRequest(BaseModel):
    name: str
    modelfile: str | None = None
    stream: bool = True
    quantize: str | None = None
