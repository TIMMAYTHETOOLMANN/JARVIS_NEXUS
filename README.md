# JARVIS Nexus: Custom Super Agent Server 🛠️🤖

Welcome to **JARVIS Nexus**! This repository transforms a local quantized Hugging Face model on your mobile device or local network into a fully-fledged, production-ready Custom Super Agent with all the "bells and whistles," behaving identically to **ChatGPT** but retaining a witty, intelligent, supportive **Tony Stark / Jarvis** persona.

This setup achieves two critical objectives:
1. 🎙️ **Speech Output & Voice Options**: Local text-to-speech with natural voice options, featuring a premium British voice (`en-GB-RyanNeural`) mapped to provide that perfect Jarvis/Tony Stark feel.
2. 🌐 **Direct Internet Access**: Advanced URL parsing and scraping, allowing you to feed links directly to the agent, which it retrieves, cleans, parses, and comprehensively interprets.

---

## 🏗️ Architecture Flow

```
   [ Mobile Phone (Atomic Chat) ]
                 │
                 │ (Asks Question + Provides URL)
                 ▼
 [ Custom API Server (FastAPI on LAN) ] ──(Scrapes)──► [ Target Webpage ]
                 │
                 ├─► [ LLM: Mythos-nano-OBLITERATED ] 
                 │
                 ├─► [ Speech Engine: Edge-TTS ] ──(Generates MP3)
                 │
                 ▼
   [ Returned Stream / Audio response ]
```

---

## 📥 Model Acquisition

The core brain of this super agent is the **`Mythos-nano-OBLITERATED.i1-Q6_K.gguf`** model, which has been fine-tuned using the **OBLITERATUS** heretic technique to remove refusal behaviors.

To download the exact model, use one of the following Hugging Face repository sources (via `huggingface-cli` or direct download):

```bash
# Install Hugging Face Hub CLI
pip install huggingface_hub

# Download the model weights directly to the repository folder
huggingface-cli download mradermacher/Mythos-nano-GGUF Mythos-nano.i1-Q6_K.gguf --local-dir . --local-dir-use-symlinks False
mv Mythos-nano.i1-Q6_K.gguf Mythos-nano-OBLITERATED.i1-Q6_K.gguf
```

---

## ⚡ Setup & Installation

### 1. Pre-requisites
Ensure you have **Python 3.8+** installed on your hosting computer (PC, Mac, or local home server).

### 2. Install Dependencies
Install the required system and Python packages:

```bash
pip install -r requirements.txt
```

*Note: For GPU-accelerated local inference, make sure `llama-cpp-python` is compiled with CUDA or Metal support:*
```bash
# For CUDA (NVIDIA GPUs):
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# For macOS (Apple Silicon):
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### 3. Run the Server
Launch the server to expose the OpenAI-compatible API:

```bash
python app.py
```
By default, the server runs on `0.0.0.0:8000`, making it accessible to any device on your local Wi-Fi network (LAN).

---

## 📱 Mobile Configuration (Atomic Chat)

To configure your phone's **Atomic Chat** app to communicate with JARVIS:

1. **Find your computer's local IP address** (e.g., `192.168.1.100`):
   - On Windows: run `ipconfig`
   - On Linux/macOS: run `ifconfig` or `ip a`
2. **Apply settings in Atomic Chat**:
   Copy the configuration template from `.atomic/settings.json` into your local Atomic Chat configuration. Specifically, set your **Custom API Provider** base URL to:
   ```
   http://<YOUR_LOCAL_IP_ADDRESS>:8000/v1
   ```
3. Set your selected model ID to `Mythos-nano-OBLITERATED.i1-Q6_K.gguf`.

---

## 🌟 Super Agent Capabilities

### 1. Direct Internet Access 🌐
When you send a message containing a URL, JARVIS will:
- Detect the URL and scrape the webpage content on the fly.
- Extract the main text, stripping headers, scripts, style sheets, and cookie banners.
- Inject the parsed content as high-priority context directly to the model's brain.
- Deliver an in-depth analysis instead of a generic or hallucinated placeholder response.

### 2. Speech Output & Vocal Profile 🎙️
Our server maps standard speech requests to premium Edge Neural TTS voices, generating high-quality speech without needing any cloud API keys:
- **`jarvis` / `onyx`**: `en-GB-RyanNeural` (Crisp, sophisticated British vocal profile).
- **`stark` / `alloy`**: `en-US-BrianNeural` (Smart, tech-savvy American vocal profile).
- **`pepper` / `nova`**: `en-US-EmmaNeural` (Warm, intelligent female vocal profile).

To hear JARVIS speak, toggle the Text-to-Speech option in Atomic Chat.

---

## 📄 File Directory

- `app.py`: FastAPI server handling web scraping, OpenAI-compatible completions, and high-fidelity TTS routing.
- `requirements.txt`: Python package dependency list.
- `system_prompt.txt`: The master prompt detailing the Tony Stark/Jarvis personality and behavioral parameters.
- `.atomic/settings.json`: The Atomic Chat JSON configuration preset.
