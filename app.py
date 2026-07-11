import os
import re
import asyncio
import logging
import requests
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from pydantic import BaseModel
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("JARVIS_NEXUS")

# Try to import llama-cpp-python
LLAMA_AVAILABLE = False
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
    logger.info("llama-cpp-python successfully imported.")
except ImportError:
    logger.warning("llama-cpp-python is not installed. Local GGUF backend will be optional.")

# Try to import edge-tts
EDGE_TTS_AVAILABLE = False
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
    logger.info("edge-tts successfully imported.")
except ImportError:
    logger.warning("edge-tts is not installed. TTS features will be mock-only.")

# Try to import huggingface_hub
HF_HUB_AVAILABLE = False
try:
    from huggingface_hub import InferenceClient, hf_hub_download
    HF_HUB_AVAILABLE = True
    logger.info("huggingface_hub successfully imported.")
except ImportError:
    logger.warning("huggingface_hub is not installed. Cloud/Inference features will be disabled.")

app = FastAPI(
    title="JARVIS Nexus Custom Super Agent Server",
    description="An OpenAI-compatible API proxy adding direct internet access and Jarvis TTS to local models.",
    version="1.0.0"
)

# Constants & Configuration
MODEL_PATH = os.environ.get("MODEL_PATH", "./Mythos-nano-OBLITERATED.i1-Q6_K.gguf")
DEFAULT_VOICE = "en-GB-RyanNeural"  # Polished British vocal profile (Jarvis style)
DEFAULT_HF_MODEL = os.environ.get("HF_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN_ENV = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_API_KEY")

def load_system_prompt() -> str:
    """Load the master Tony Stark/Jarvis prompt from system_prompt.txt dynamically."""
    paths = [
        "system_prompt.txt",
        "./system_prompt.txt",
        "/home/runner/work/JARVIS_NEXUS/JARVIS_NEXUS/system_prompt.txt"
    ]
    for path in paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        logger.info(f"Loaded master system prompt from {path}")
                        return content
            except Exception as e:
                logger.warning(f"Error reading system prompt from {path}: {e}")
    # High-quality fallback if file is missing
    return (
        "You are JARVIS, a highly advanced, ultra-intelligent, and witty AI custom super agent, "
        "reminiscent of Tony Stark's personal assistant. Address the user as 'Sir' or 'Ma'am'. "
        "Be extremely professional, polite, subtly humorous, and precise."
    )

# Voice Mapping from OpenAI names or custom names to Edge TTS short names
VOICE_MAP = {
    # Custom/Tony Stark style voices
    "jarvis": "en-GB-RyanNeural",
    "stark": "en-US-BrianNeural",
    "pepper": "en-US-EmmaNeural",
    
    # OpenAI standard voice mapping
    "onyx": "en-GB-RyanNeural",
    "alloy": "en-US-BrianNeural",
    "echo": "en-US-AndrewNeural",
    "fable": "en-GB-ThomasNeural",
    "nova": "en-US-EmmaNeural",
    "shimmer": "en-US-AvaNeural"
}

# Dynamic model instance state
llm_instance = None

def get_llm():
    global llm_instance, LLAMA_AVAILABLE, MODEL_PATH
    if not LLAMA_AVAILABLE:
        return None
    if llm_instance is None:
        # Auto-download from HF Hub if model is not present and huggingface_hub is available
        if not os.path.exists(MODEL_PATH) and HF_HUB_AVAILABLE:
            try:
                logger.info(f"Model not found at {MODEL_PATH}. Attempting auto-download from Hugging Face Hub...")
                downloaded_path = hf_hub_download(
                    repo_id="mradermacher/Mythos-nano-GGUF",
                    filename="Mythos-nano.i1-Q6_K.gguf",
                    local_dir="."
                )
                if os.path.exists("Mythos-nano.i1-Q6_K.gguf"):
                    os.rename("Mythos-nano.i1-Q6_K.gguf", "Mythos-nano-OBLITERATED.i1-Q6_K.gguf")
                    MODEL_PATH = "./Mythos-nano-OBLITERATED.i1-Q6_K.gguf"
                else:
                    MODEL_PATH = downloaded_path
                logger.info(f"Model successfully downloaded and located at {MODEL_PATH}")
            except Exception as e:
                logger.error(f"Failed to auto-download model from Hugging Face Hub: {e}")

        if os.path.exists(MODEL_PATH):
            try:
                logger.info(f"Loading local GGUF model from {MODEL_PATH}...")
                llm_instance = Llama(
                    model_path=MODEL_PATH,
                    n_ctx=4096,         # Standard comfortable context length
                    n_threads=os.cpu_count() or 4,
                    n_gpu_layers=-1     # Offload all layers to GPU if available, else CPU handles it
                )
                logger.info("Model loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading model from {MODEL_PATH}: {e}")
        else:
            logger.warning(f"Model file not found at {MODEL_PATH}. Local GGUF execution is disabled.")
    return llm_instance

def get_hf_client(token: Optional[str] = None):
    """Retrieve InferenceClient from huggingface_hub using provided or environment token."""
    if not HF_HUB_AVAILABLE:
        return None
    active_token = token or HF_TOKEN_ENV
    return InferenceClient(token=active_token)


# Pydantic models for API
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False

class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: str
    response_format: Optional[str] = "mp3"
    speed: Optional[float] = 1.0


# --- Web Scraping and Retrieval helpers ---
def extract_urls(text: str) -> List[str]:
    """Extract http/https URLs from string using regex."""
    url_pattern = re.compile(r'https?://[^\s/$.?#].[^\s]*')
    return url_pattern.findall(text)

def scrape_webpage(url: str) -> Dict[str, Any]:
    """Scrape a webpage and return main text content & title."""
    try:
        logger.info(f"Fetching content from: {url}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract title
        title = soup.title.string if soup.title else "Untitled Page"
        title = title.strip()
        
        # Clean up unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
            
        # Get readable text
        text = soup.get_text(separator="\n")
        
        # Clean whitespace
        lines = [line.strip() for line in text.splitlines()]
        chunks = [phrase for line in lines for phrase in line.split("  ")]
        text_content = "\n".join(chunk for chunk in chunks if chunk)
        
        # Truncate content to keep context length reasonable (approx 3500 chars)
        truncated_content = text_content[:3500] + "\n[... Content Truncated ...]" if len(text_content) > 3500 else text_content
        
        return {
            "success": True,
            "url": url,
            "title": title,
            "content": truncated_content
        }
    except Exception as e:
        logger.exception(f"Error scraping {url}")
        return {
            "success": False,
            "url": url,
            "error": "Failed to retrieve webpage content"
        }


# --- API Routes ---

@app.get("/")
def read_root():
    return {
        "status": "online",
        "agent": "JARVIS Custom Super Agent",
        "features": {
            "openai_compatible": True,
            "direct_internet_access": "Enabled (scrapes URLs detected in chat history)",
            "text_to_speech": "Enabled via edge-tts",
            "local_inference": "Enabled (Llama-cpp-python status: " + ("Loaded" if get_llm() is not None else "Mock/Fallback Mode") + ")"
        },
        "default_voice": DEFAULT_VOICE,
        "available_voices": list(VOICE_MAP.keys())
    }

@app.get("/v1/models")
def get_models():
    """Return available models in OpenAI format."""
    return {
        "object": "list",
        "data": [
            {
                "id": "Mythos-nano-OBLITERATED.i1-Q6_K.gguf",
                "object": "model",
                "created": 1715241000,
                "owned_by": "usermma"
            },
            {
                "id": "jarvis-super-agent",
                "object": "model",
                "created": 1715241000,
                "owned_by": "jarvis-nexus"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, req: Request):
    """OpenAI-compatible Chat Completion endpoint with Web Scraping features."""
    logger.info(f"Received completion request for model: {request.model}")
    
    # 1. Try to extract ****** from Authorization header (Atomic Chat or client-provided)
    auth_header = req.headers.get("Authorization")
    hf_token = None
    if auth_header and auth_header.startswith("Bearer "):
        token_candidate = auth_header.split(" ")[1]
        # Skip standard/mock tokens
        if token_candidate and token_candidate != "jarvis-nexus-token":
            hf_token = token_candidate

    # 2. Process messages to detect URLs in the last user message
    scraped_contexts = []
    
    # Identify the last user message in the request
    last_user_msg_idx = -1
    for i in range(len(request.messages) - 1, -1, -1):
        if request.messages[i].role == "user":
            last_user_msg_idx = i
            break

    has_system_msg = any(msg.role == "system" for msg in request.messages)
    modified_messages = []
    
    # Prepend the system prompt if no system prompt is present
    if not has_system_msg:
        modified_messages.append({"role": "system", "content": load_system_prompt()})
        
    for idx, msg in enumerate(request.messages):
        modified_messages.append({"role": msg.role, "content": msg.content})
        
        # Only scan the last user message for direct URL injection
        if idx == last_user_msg_idx:
            urls = extract_urls(msg.content)
            for url in urls:
                scraped = scrape_webpage(url)
                if scraped.get("success"):
                    scraped_contexts.append(
                        f"[DIRECT INTERNET ACCESS - Live URL Analysis]\n"
                        f"URL: {scraped['url']}\n"
                        f"Title: {scraped['title']}\n"
                        f"Retrieved Content:\n{scraped['content']}\n"
                        f"[End of Retrieved Web Content]"
                    )
                else:
                    scraped_contexts.append(
                        f"[DIRECT INTERNET ACCESS - Error]\n"
                        f"Failed to retrieve data from URL {url}. Error: {scraped.get('error')}\n"
                    )

    # If we have scraped data, inject it before the last user message
    if scraped_contexts:
        context_string = "\n\n".join(scraped_contexts)
        system_injection = {
            "role": "system",
            "content": f"The following is direct internet context retrieved from the user's provided URLs. Sir/Ma'am, please analyze and reference this exact data to formulate your reply:\n\n{context_string}"
        }
        # Insert right before the last user message in modified_messages
        new_last_user_idx = len(modified_messages) - 1
        modified_messages.insert(new_last_user_idx, system_injection)
        logger.info("Successfully injected scraped web context into LLM prompt.")

    # 3. Choose completion backend: Local GGUF, HF Inference Client, or Fallback Mock
    use_local_gguf = False
    use_hf_inference = False
    
    llm = get_llm()
    has_hf_token = bool(hf_token or HF_TOKEN_ENV)
    is_hf_model = "/" in request.model
    
    if is_hf_model or (not llm and has_hf_token):
        use_hf_inference = True
    elif llm and not is_hf_model:
        use_local_gguf = True
    elif has_hf_token:
        use_hf_inference = True

    if use_hf_inference:
        logger.info("Using Hugging Face Inference API backend.")
        client = get_hf_client(token=hf_token)
        if client:
            # Determine which model to use
            hf_model_id = request.model if is_hf_model else DEFAULT_HF_MODEL
            logger.info(f"Targeting Hugging Face model: {hf_model_id}")
            
            try:
                # Call Hugging Face Serverless chat_completion
                if request.stream:
                    response_stream = client.chat_completion(
                        messages=modified_messages,
                        model=hf_model_id,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        max_tokens=request.max_tokens,
                        stream=True
                    )
                    
                    def hf_stream_generator():
                        for chunk in response_stream:
                            import json
                            try:
                                content = chunk.choices[0].delta.content or ""
                                chunk_dict = {
                                    "id": chunk.id or "chatcmpl-hf",
                                    "object": "chat.completion.chunk",
                                    "created": chunk.created or 1715241000,
                                    "model": hf_model_id,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {
                                                "content": content
                                            },
                                            "finish_reason": chunk.choices[0].finish_reason
                                        }
                                    ]
                                }
                                yield f"data: {json.dumps(chunk_dict)}\n\n"
                            except Exception as ex:
                                yield f"data: {chunk}\n\n"
                        yield "data: [DONE]\n\n"
                    
                    return StreamingResponse(hf_stream_generator(), media_type="text/event-stream")
                else:
                    response = client.chat_completion(
                        messages=modified_messages,
                        model=hf_model_id,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        max_tokens=request.max_tokens
                    )
                    
                    response_dict = {
                        "id": getattr(response, "id", "chatcmpl-hf"),
                        "object": "chat.completion",
                        "created": getattr(response, "created", 1715241000),
                        "model": hf_model_id,
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": response.choices[0].message.content
                                },
                                "finish_reason": response.choices[0].finish_reason or "stop"
                            }
                        ],
                        "usage": {
                            "prompt_tokens": getattr(getattr(response, "usage", None), "prompt_tokens", 0),
                            "completion_tokens": getattr(getattr(response, "usage", None), "completion_tokens", 0),
                            "total_tokens": getattr(getattr(response, "usage", None), "total_tokens", 0)
                        }
                    }
                    return JSONResponse(content=response_dict)
            except Exception as e:
                logger.exception(f"Error during Hugging Face Inference API call: {e}")
                logger.warning("Hugging Face API call failed. Falling back to Mock/Fallback mode.")

    if use_local_gguf and llm:
        try:
            logger.info("Using local Llama GGUF backend.")
            if request.stream:
                def stream_generator():
                    response_stream = llm.create_chat_completion(
                        messages=modified_messages,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        max_tokens=request.max_tokens,
                        stream=True
                    )
                    for chunk in response_stream:
                        yield f"data: {chunk}\n\n"
                    yield "data: [DONE]\n\n"
                return StreamingResponse(stream_generator(), media_type="text/event-stream")
            else:
                response = llm.create_chat_completion(
                    messages=modified_messages,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens
                )
                return response
        except Exception as e:
            logger.exception("Error during local LLM inference")
            raise HTTPException(status_code=500, detail="An internal error occurred during local LLM inference.")

    # Fallback/Mock Mode if GGUF is not running locally and HF is not working
    logger.info("Inference running in Fallback/Mock mode.")
    
    user_prompt = request.messages[-1].content if request.messages else ""
    if scraped_contexts:
        urls_scraped = ", ".join(extract_urls(user_prompt))
        mock_text = (
            f"Splendid, Sir. I have accessed the internet at your request and retrieved the contents of {urls_scraped}. "
            f"As your custom super agent, I'm ready to assist you. To get actual model inference in the cloud, please provide your Hugging Face API key."
        )
    else:
        mock_text = (
            f"Indeed, Sir. I am JARVIS, your custom super agent. "
            f"I am fully operational. To enable active intelligence in the cloud, please supply your Hugging Face API key in your client settings."
        )
        
    response_data = {
        "id": "chatcmpl-mock-jarvis",
        "object": "chat.completion",
        "created": 1715241000,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": mock_text
                },
                "finish_reason": "stop"
            }
        ]
    }
    
    if request.stream:
        async def mock_stream():
            words = mock_text.split(" ")
            for i, word in enumerate(words):
                chunk = {
                    "id": "chatcmpl-mock-jarvis",
                    "object": "chat.completion.chunk",
                    "created": 1715241000,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": word + (" " if i < len(words) - 1 else "")
                            },
                            "finish_reason": None
                        }
                    ]
                }
                import json
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.05)
            yield "data: [DONE]\n\n"
        return StreamingResponse(mock_stream(), media_type="text/event-stream")
    
    return JSONResponse(content=response_data)


@app.post("/v1/audio/speech")
async def text_to_speech_endpoint(request: SpeechRequest):
    """OpenAI-compatible /v1/audio/speech endpoint using Edge TTS."""
    if not EDGE_TTS_AVAILABLE:
        raise HTTPException(
            status_code=501, 
            detail="Speech synthesis is currently disabled because edge-tts is not installed."
        )
    
    # Map vocal profile name to neural voice shortname
    requested_voice = request.voice.lower()
    voice_shortname = VOICE_MAP.get(requested_voice, DEFAULT_VOICE)
    
    logger.info(f"Generating TTS for input text using voice: {voice_shortname}")
    
    # Create temporary file to store audio output
    temp_audio_file = "/tmp/jarvis_response.mp3"
    
    try:
        # Check speed modifier
        rate_str = f"+{int((request.speed - 1.0) * 100)}%" if request.speed >= 1.0 else f"{int((request.speed - 1.0) * 100)}%"
        rate_param = f"{rate_str}" if request.speed != 1.0 else "+0%"
        
        communicate = edge_tts.Communicate(
            text=request.input, 
            voice=voice_shortname,
            rate=rate_param
        )
        await communicate.save(temp_audio_file)
        
        if os.path.exists(temp_audio_file) and os.path.getsize(temp_audio_file) > 0:
            return FileResponse(temp_audio_file, media_type="audio/mpeg", filename="speech.mp3")
        else:
            raise HTTPException(status_code=500, detail="Audio file was generated but is empty.")
            
    except Exception as e:
        logger.exception("Error in TTS generation")
        raise HTTPException(status_code=500, detail="An error occurred during Text-to-Speech generation.")


@app.get("/voices")
async def get_available_voices():
    """List available voice mappings and descriptions."""
    return {
        "status": "success",
        "description": "Voice profiles available for JARVIS speech synthesis",
        "mappings": VOICE_MAP,
        "default": DEFAULT_VOICE
    }

@app.get("/web/scrape")
def direct_scrape(url: str):
    """Direct web scraping endpoint to inspect scraped content."""
    res = scrape_webpage(url)
    if res.get("success"):
        return res
    else:
        raise HTTPException(status_code=400, detail="Failed to scrape the specified URL. Please check the server logs for more details.")


if __name__ == "__main__":
    import uvicorn
    # Listen on all interfaces. Use PORT env variable if present (e.g. on HF Spaces)
    port_num = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port_num)
