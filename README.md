---
title: JARVIS Nexus Custom Super Agent
emoji: 🤖
colorFrom: red
colorTo: blue
sdk: fastapi
app_file: app.py
pinned: false
---

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

## ☁️ Deployment to Hugging Face Spaces (Cloud / Serverless)

You can host **JARVIS Nexus** completely in the cloud on **Hugging Face Spaces** for free! 
In this mode, the server will leverage the free, high-performance **Hugging Face Serverless Inference API** as its brain instead of running a heavy local GGUF model. This eliminates memory limits, does not require compiling C++ packages, and is highly responsive with zero cold-starts!

### Step 1: Create a Hugging Face Space
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Name your Space (e.g., `jarvis-nexus`).
3. Select **FastAPI** as the Space SDK (this repository has built-in support for Hugging Face's native FastAPI SDK).
4. Set the space visibility to **Public** or **Private** (Private is recommended to protect your API usage and tokens).

### Step 2: Upload the Files
Choose one of the following methods to push this repository to your Hugging Face Space:
- **Direct Upload**: Go to the **Files** tab of your Space, click **Add file** -> **Upload files**, and upload the following files from your local repository: `app.py`, `requirements.txt`, `system_prompt.txt`, `.huggingface.yml`, and `README.md`.
- **Git Push**: Clone your Hugging Face Space repository locally and push this codebase to your Space's git remote.

### Step 3: Configure Environment Variables (Optional but Recommended)
In your Hugging Face Space's **Settings** tab, scroll down to **Variables and secrets**:
- **`HF_TOKEN`** (Secret): Your Hugging Face API Token (get one from `huggingface.co/settings/tokens`). This enables calling larger/gated models.
- **`HF_MODEL_ID`** (Variable): The model to use for completions. Defaults to `Qwen/Qwen2.5-7B-Instruct`, but you can use any chat-supported model on Hugging Face (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`, etc.).

### Step 4: Configure Atomic Chat
Once your Space is built and running:
1. Copy the Direct URL of your Space (which looks like `https://<your-username>-<your-space-name>.hf.space`).
2. Open **Atomic Chat** on your mobile device.
3. Apply the settings from `.atomic/settings.json`, modifying the **Custom API Provider** base URL to:
   ```
   https://<your-username>-<your-space-name>.hf.space/v1
   ```
4. Set the **API Key** in Atomic Chat to your **Hugging Face API Token** (`hf_...`).
5. Select any model ID or set it to your Hugging Face Model ID!

Now, you have a private, fully-featured Custom Super Agent with direct internet access and high-fidelity text-to-speech accessible securely from anywhere in the world!

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
