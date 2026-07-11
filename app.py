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
    logger.warning("llama-cpp-python is not installed. Running in hybrid/mock/proxy mode.")

# Try to import edge-tts
EDGE_TTS_AVAILABLE = False
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
    logger.info("edge-tts successfully imported.")
except ImportError:
    logger.warning("edge-tts is not installed. TTS features will be mock-only.")

app = FastAPI(
    title="JARVIS Nexus Custom Super Agent Server",
    description="An OpenAI-compatible API proxy adding direct internet access and Jarvis TTS to local models.",
    version="1.0.0"
)

# Constants & Configuration
MODEL_PATH = os.environ.get("MODEL_PATH", "./Mythos-nano-OBLITERATED.i1-Q6_K.gguf")
DEFAULT_VOICE = "en-GB-RyanNeural"  # Polished British vocal profile (Jarvis style)

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
    global llm_instance, LLAMA_AVAILABLE
    if not LLAMA_AVAILABLE:
        return None
    if llm_instance is None:
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
            logger.warning(f"Model file not found at {MODEL_PATH}. Running in mock/proxy mode.")
    return llm_instance


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
        logger.error(f"Error scraping {url}: {e}")
        return {
            "success": False,
            "url": url,
            "error": str(e)
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
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible Chat Completion endpoint with Web Scraping features."""
    logger.info(f"Received completion request for model: {request.model}")
    
    # Process messages to detect URLs in the last user message
    modified_messages = []
    scraped_contexts = []
    
    for idx, msg in enumerate(request.messages):
        modified_messages.append({"role": msg.role, "content": msg.content})
        
        # Only scan user messages for direct URL injection
        if msg.role == "user" and idx == len(request.messages) - 1:
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
        # We place the context injection right before the user message or as a system prompt update
        system_injection = {
            "role": "system",
            "content": f"The following is direct internet context retrieved from the user's provided URLs. Sir/Ma'am, please analyze and reference this exact data to formulate your reply:\n\n{context_string}"
        }
        # Insert at the beginning or right before the user message
        modified_messages.insert(-1, system_injection)
        logger.info("Successfully injected scraped web context into LLM prompt.")

    # Convert messages list to llama-cpp format or run fallback
    llm = get_llm()
    if llm:
        try:
            # We use llama-cpp-python's create_chat_completion
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
            logger.error(f"Error during LLM inference: {e}")
            raise HTTPException(status_code=500, detail=f"LLM Inference Error: {e}")
    else:
        # Fallback/Mock Mode if GGUF is not running locally (e.g. while testing)
        logger.info("Inference running in Fallback/Mock mode (GGUF model not loaded).")
        
        # Let's generate a clever Jarvis-like response indicating we scraped the URL
        user_prompt = request.messages[-1].content
        if scraped_contexts:
            urls_scraped = ", ".join(extract_urls(user_prompt))
            mock_text = (
                f"Splendid, Sir. I have accessed the internet at your request and retrieved the contents of {urls_scraped}. "
                f"The webpage appears to be titled '{scraped['title'] if 'scraped' in locals() else 'Webpage'}' and contains detailed data regarding your query. "
                f"As your custom super agent, I will analyze this information immediately. Let me know how you'd like to proceed with the parsed details, Sir."
            )
        else:
            mock_text = (
                f"Indeed, Sir. I am JARVIS, your custom super agent running on the Mythos-nano-OBLITERATED GGUF model. "
                f"I am fully operational, although the local GGUF weights are currently offline. How can I assist you with your day today?"
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
            # Simple streaming simulation
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
        logger.error(f"Error in TTS generation: {e}")
        raise HTTPException(status_code=500, detail=f"TTS Generation failed: {str(e)}")


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
        raise HTTPException(status_code=400, detail=res.get("error"))


if __name__ == "__main__":
    import uvicorn
    # Listen on all interfaces so the phone can connect over the local network (LAN)
    uvicorn.run(app, host="0.0.0.0", port=8000)
