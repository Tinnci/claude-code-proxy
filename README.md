# Anthropic API Proxy for Gemini, OpenAI, Vertex AI & xAI Models 🔄

**Use Anthropic clients (like Claude Code) with Gemini, OpenAI, Vertex AI, or xAI backends.** 🤝

A proxy server that lets you use Anthropic clients with multiple LLM providers via LiteLLM. 🌉


![Anthropic API Proxy](pic.png)

## Quick Start ⚡

### Prerequisites

- OpenAI API key 🔑 (if using OpenAI provider)
- Google AI Studio (Gemini) API key 🔑 (if using Google provider)
- Google Cloud Project with Vertex AI API enabled 🔑 (if using Vertex AI provider)
- xAI API key 🔑 (if using xAI provider)
- [uv](https://github.com/astral-sh/uv) installed.

### Setup 🛠️

1. **Clone this repository**:
   ```bash
   git clone https://github.com/CogAgent/claude-code-proxy.git
   cd claude-code-proxy
   ```

2. **Install uv** (if you haven't already):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   *(`uv` will handle dependencies based on `pyproject.toml` when you run the server)*

3. **Configure Environment Variables**:
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and fill in your API keys and model configurations:

   *   `ANTHROPIC_API_KEY`: (Optional) Needed only if proxying *to* Anthropic models.
   *   `OPENAI_API_KEY`: Your OpenAI API key (Required if using the default OpenAI preference or as fallback).
   *   `GEMINI_API_KEY`: Your Google AI Studio (Gemini) API key (Required if PREFERRED_PROVIDER=google).
   *   `XAI_API_KEY`: Your xAI API key (Required if PREFERRED_PROVIDER=xai).

   **For Vertex AI:**
   *   `VERTEX_PROJECT_ID`: Your Google Cloud Project ID (Required if PREFERRED_PROVIDER=vertex).
   *   `VERTEX_LOCATION`: Region where your Vertex AI resources are located (defaults to us-central1).
   *   Set up Application Default Credentials (ADC) with `gcloud auth application-default login` or set `GOOGLE_APPLICATION_CREDENTIALS` to point to your service account key file.

   **Provider and Model Configuration:**
   *   `PREFERRED_PROVIDER` (Optional): Set to `openai` (default), `google`, `vertex`, or `xai`. This determines the primary backend for mapping `haiku`/`sonnet`.
   *   `BIG_MODEL` (Optional): The model to map `sonnet` requests to. Defaults vary by provider.
   *   `SMALL_MODEL` (Optional): The model to map `haiku` requests to. Defaults vary by provider.

   **Mapping Logic:**
   - If `PREFERRED_PROVIDER=openai` (default), `haiku`/`sonnet` map to `SMALL_MODEL`/`BIG_MODEL` prefixed with `openai/`.
   - If `PREFERRED_PROVIDER=google`, `haiku`/`sonnet` map to `SMALL_MODEL`/`BIG_MODEL` prefixed with `gemini/`.
   - If `PREFERRED_PROVIDER=vertex`, `haiku`/`sonnet` map to `SMALL_MODEL`/`BIG_MODEL` prefixed with `vertex_ai/`.
   - If `PREFERRED_PROVIDER=xai`, `haiku`/`sonnet` map to `SMALL_MODEL`/`BIG_MODEL` prefixed with `xai/`.

4. **Run the server**:
   ```bash
   # Development (auto-reload)
   uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8082

   # Production example
   uvicorn app.main:app --host 0.0.0.0 --port 8082 --workers 4
   ```
   *(`--reload` is optional, for development)*

### Dependency Groups & Extras 📦

`pyproject.toml` exposes optional dependencies in two complementary ways:

| Type | Table Location | Typical Audience | Install Command |
|------|----------------|------------------|-----------------|
| **extras** | `[project.optional-dependencies]` (PEP-621) | End-users who install the wheel from PyPI | `pip install anthropic-proxy[vertex]` |
| **groups** | `[dependency-groups]` (uv specific) | Developers / CI / staged deployments | `uv sync --group tests` |

Common commands:

```bash
# Install main dependencies (default + dev)
uv sync

# Add test tools
uv sync --group tests

# Add Vertex-AI support
uv sync --group vertex
```

Using plain pip you can still consume the *extras* interface:

```bash
pip install .[vertex]
pip install .[tests]
```

### Using with Claude Code 🎮

1. **Install Claude Code** (if you haven't already):
   ```bash
   npm install -g @anthropic-ai/claude-code
   ```

2. **Connect to your proxy**:
   ```bash
   ANTHROPIC_BASE_URL=http://localhost:8082 claude
   ```

3. **That's it!** Your Claude Code client will now use the configured backend models (defaulting to Gemini) through the proxy. 🎯

## Model Mapping 🗺️

The proxy automatically maps Claude models to OpenAI, Gemini, Vertex AI, or xAI models based on the configured provider:

| Claude Model | OpenAI (default) | Gemini | Vertex AI | xAI |
|--------------|-----------------|--------|-----------|-----|
| haiku | openai/gpt-4.1-mini | gemini/gemini-2.0-flash | vertex_ai/gemini-2.0-flash | xai/grok-3-mini-beta |
| sonnet | openai/gpt-4.1 | gemini/gemini-2.5-pro-preview-03-25 | vertex_ai/gemini-2.5-pro-preview-03-25 | xai/grok-3 |

### Supported Models

#### OpenAI Models
The following OpenAI models are supported with automatic `openai/` prefix handling:
- o3-mini
- o1
- o1-mini
- o1-pro
- gpt-4.5-preview
- gpt-4o
- gpt-4o-audio-preview
- chatgpt-4o-latest
- gpt-4o-mini
- gpt-4o-mini-audio-preview
- gpt-4.1
- gpt-4.1-mini

#### Gemini Models
The following Gemini models are supported with automatic `gemini/` prefix handling:
- gemini-2.5-pro-preview-03-25
- gemini-2.0-flash

#### Vertex AI Models
The following Vertex AI models are supported with automatic `vertex_ai/` prefix handling:
- gemini-2.5-pro-preview-03-25
- gemini-2.5-flash-preview-04-17
- gemini-2.0-flash
- gemini-1.5-flash-preview-0514
- gemini-1.5-pro-preview-0514

#### xAI Models
The following xAI models are supported with automatic `xai/` prefix handling:
- grok-3-mini-beta
- grok-3-beta
- grok-2-vision-latest
- grok-2
- grok-1

### Model Prefix Handling
The proxy automatically adds the appropriate prefix to model names:
- OpenAI models get the `openai/` prefix
- Gemini models get the `gemini/` prefix
- Vertex AI models get the `vertex_ai/` prefix
- xAI models get the `xai/` prefix
- The BIG_MODEL and SMALL_MODEL will get the appropriate prefix based on the provider and model lists

For example:
- `gpt-4o` becomes `openai/gpt-4o`
- `gemini-2.5-pro-preview-03-25` becomes `gemini/gemini-2.5-pro-preview-03-25` or `vertex_ai/gemini-2.5-pro-preview-03-25` depending on the provider
- `grok-3` becomes `xai/grok-3`

### Customizing Model Mapping

Control the mapping using environment variables in your `.env` file or directly:

**Example 1: Default (Use OpenAI)**
```dotenv
OPENAI_API_KEY="your-openai-key"
# PREFERRED_PROVIDER="openai" # Optional, it's the default
# BIG_MODEL="gpt-4.1" # Optional, it's the default
# SMALL_MODEL="gpt-4.1-mini" # Optional, it's the default
```

**Example 2: Use Google AI Studio**
```dotenv
GEMINI_API_KEY="your-google-key"
OPENAI_API_KEY="your-openai-key" # Needed for fallback
PREFERRED_PROVIDER="google"
# BIG_MODEL="gemini-2.5-pro-preview-03-25" # Optional, it's the default for Google pref
# SMALL_MODEL="gemini-2.0-flash" # Optional, it's the default for Google pref
```

**Example 3: Use Vertex AI**
```dotenv
VERTEX_PROJECT_ID="your-gcp-project-id"
VERTEX_LOCATION="us-central1"
# Set GOOGLE_APPLICATION_CREDENTIALS or use gcloud auth application-default login
PREFERRED_PROVIDER="vertex"
BIG_MODEL="gemini-2.5-pro-preview-03-25"
SMALL_MODEL="gemini-2.0-flash"
```

**Example 4: Use xAI**
```dotenv
XAI_API_KEY="your-xai-api-key"
PREFERRED_PROVIDER="xai"
BIG_MODEL="grok-3"
SMALL_MODEL="grok-3-mini-beta"
```

**Example 5: Use Specific OpenAI Models**
```dotenv
OPENAI_API_KEY="your-openai-key"
PREFERRED_PROVIDER="openai"
BIG_MODEL="gpt-4o" # Example specific model
SMALL_MODEL="gpt-4o-mini" # Example specific model
```

## How It Works 🧩

This proxy works by:

1. **Receiving requests** in Anthropic's API format 📥
2. **Translating** the requests to the appropriate format via LiteLLM 🔄
3. **Sending** the translated request to the selected provider (OpenAI, Gemini, Vertex AI, or xAI) 📤
4. **Converting** the response back to Anthropic format 🔄
5. **Returning** the formatted response to the client ✅

The proxy handles both streaming and non-streaming responses, maintaining compatibility with all Claude clients. It also handles provider-specific authentication and configuration requirements. 🌊

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request. 🎁
