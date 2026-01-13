# Personal AI Voice Clone & Chatbot 🤖

A full-stack AI application that replicates a specific user's personality and voice. The system utilizes a fine-tuned **Llama-3 8B** model for text generation and the **ElevenLabs API** for low-latency voice synthesis, delivered via a Streamlit interface.

This project demonstrates the end-to-end pipeline of creating a personal LLM: from raw data extraction (iMessage), to cleaning/structuring, fine-tuning (QLoRA), and deployment via a RAG-enhanced app.

## 🚀 Key Features

* **Fine-Tuned LLM:** Utilized **Unsloth** (QLoRA) to fine-tune Llama-3-8B on 50k+ private text messages, capturing specific slang, sentence structure, and personality quirks.
* **Voice Synthesis:** Integrated **ElevenLabs** for realistic text-to-speech generation with custom voice cloning (Optional).
* **RAG Pipeline:** Implemented a context-aware system prompt utilizing "Emotional Mirroring" to adapt tone based on the partner's input.
* **Local Inference:** Optimized for local GPU execution using `llama.cpp` (GGUF format) for privacy and speed.
* **Remote Deployment:** Includes a custom Colab notebook with **TCP Tunneling** support.

## 🛠️ Architecture

1.  **Data Engineering:**
    * **Extraction:** Custom Python scripts (`sqlite3`) to extract raw messages from local iMessage `chat.db`.
    * **Cleaning:** Regex pipelines to strip system messages ("Loved an image") and group conversations.
2.  **Model Training:**
    * **Framework:** Unsloth (PyTorch) + Hugging Face TRL.
    * **Technique:** QLoRA (4-bit quantization) on a Tesla T4 GPU.
    * **Notebook:** See `notebooks/fine_tuning_pipeline.ipynb`.
3.  **Deployment:**
    * **Backend:** `llama-cpp-python` for GGUF inference.
    * **Frontend:** Streamlit for chat interface and audio playback.

---

## 🧠 Part 1: How to Train Your Own Model

Follow this pipeline if you want to create a model trained on your own text messages.

### 1. Data Extraction (macOS Only)
*To preface, I trained my persona on my own texts and messages from my own Apple phone, therefore this method would only work on **macOS**. I'm positive it would be *much easier* to use other platforms, but this is just how I did it.*

Use the included script to pull messages from your local `chat.db`.
1.  Open `scripts/export_messages.py`.
2.  Update `TARGET_HANDLE_ID` with the phone number/email of the person you text the most.
3.  Run the script:
    ```bash
    python scripts/export_messages.py
    ```
    *Output: `data/grouped_training_data.jsonl`*

### 2. Data Cleaning & Formatting
Raw texts are messy. This script removes "Tapback" reactions and weird formatting, preparing it for the LLM.
1.  Run the cleaning script:
    ```bash
    python scripts/clean_chats.py
    ```
    *Output: `data/clean_training_data.jsonl`*

**Format Example:**
The script produces a JSONL file compatible with Alpaca/Unsloth:
```json
{"instruction": "Hey, how was your day?", "input": "", "output": "It was pretty good. Just grounded out some code. Hbu?"}
```

### 3. Fine-Tuning (Google Colab)
1. Open `notebooks/fine_tuning_pipeline.ipynb` in Google Colab.

2. Upload your `clean_training_data.jsonl` to the session.

3. Run the notebook. It uses Unsloth to fine-tune Llama-3 8B (2x faster, 60% less memory).

4. **Export:** At the end of the notebook, ensure you run the cell to save as **GGUF (q4_k_m)**.

### 🚨 Troubleshooting: If Export Fails
If the automatic export crashes Colab (common with memory limits), use this manual script in a new Colab cell to build llama.cpp and convert the model yourself:

```python
import os
# 1. Clone & Build llama.cpp
if not os.path.exists("/content/llama.cpp"):
    !git clone [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) /content/llama.cpp

%cd /content/llama.cpp

!pip install -r requirements.txt
!cmake -B build
!cmake --build build --config Release -j 4

# 2. Convert & Quantize
print("Converting...")
!python convert_hf_to_gguf.py /content/model --outfile /content/temp.gguf --outtype f16

print("Quantizing...")
!./build/bin/llama-quantize /content/temp.gguf /content/final_model.gguf q4_k_m

print("✅ Done! Download 'final_model.gguf'")
```


## ⚙️ Part 2: How to Run the App
Once you have a model (or if you want to skip training and use a generic one), follow these steps.

### Option A: Local Installation (Recommended)
**Prerequisites:** A computer with a GPU (NVIDIA or Mac M-series) is highly recommended.

### 1. Clone the Repository

```bash
git clone [https://ghttps://github.com/micccon/Personal-AI-Clone.git](https://github.com/micccon/Personal-AI-Clone.git)
cd Personal-Voice-Clone
```
### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup the Model (You need a .gguf model file in the models/ folder.)

* **If you trained your own:** Move your `final_model.gguf` (from Part 1) into `models/`.

* **If skipping training:** Download a generic Llama-3 model:

    * **Link:** [Meta-Llama-3-8B-Instruct-Q4_K_M.gguf](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf)

    * **Action:** Save it as models/`default_model.gguf`.

### 4. Configure Environment

Create a `.env` file in the root directory (copy from `.env.example`) and fill in your keys.

* `ELEVENLABS_API_KEY`: (Optional) Required for voice mode.

* `VOICE_ID`: (Optional) The ID of the voice you want to clone.

* `MODEL_PATH`: Path to your model (e.g., `models/default_model.gguf`).

### 5. Run the App

```bash
streamlit run app.py
```

### Option B: Google Colab (Cloud Demo)
If you don't have a strong GPU, use the provided deployment notebook. This method allows you to run the app in the cloud and access it via a secure tunnel.

1.  Open `notebooks/colab_deployment.ipynb` in Google Colab.
2.  **Run the Cells:** The notebook will guide you through the setup.
    * **Cloud Setup:** It automates cloning the repo and installing dependencies.
    * **Model Selection:** You will be prompted to either:
        * Use a **generic Llama-3 demo model** (default).
        * OR provide a **direct download link** to your own fine-tuned `.gguf` model if you have trained one (e.g., from Hugging Face).
    * **Launch:** It establishes a **TCP Tunnel** via Ngrok so you can interact with the app from your browser.

*(Note: You will need a free Ngrok Authtoken for the tunnel).*

**Finally!** Click the `tcp://...` link provided in the output to use the app!

⚠️ *Privacy Note:*
The dataset (private text messages) and the fine-tuned model weights are ***not included*** in this repository to preserve privacy. This repository serves as a technical showcase of the architecture and training code.