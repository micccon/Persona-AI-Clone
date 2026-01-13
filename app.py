import streamlit as st
import tempfile
import os
import sys
from dotenv import load_dotenv

# --- LOAD SECRETS ---
load_dotenv()

# --- IMPORT ENGINES (ROBUST LOADING) ---
# 1. Critical: LLM Engine (Must work or we crash)
try:
    from llm_engine import LLMEngine
except ImportError as e:
    st.error(f"❌ CRITICAL ERROR: Could not import LLMEngine. {e}")
    st.stop()

# 2. Optional: Voice Engine (Can fail safely)
try:
    from voice_engine import VoiceEngine
except ImportError:
    print("⚠️ Voice library not found. Voice features disabled.")
    VoiceEngine = None

# --- CONFIGURATION ---
MODEL_PATH = os.getenv("MODEL_PATH", "models/default_model.gguf") 
VOICE_ID = os.getenv("VOICE_ID") 

st.set_page_config(page_title="Digital Clone", page_icon="🤖", layout="centered")

# --- CSS STYLING ---
st.markdown("""
<style>
    .stChatMessage { font-family: 'Helvetica Neue', sans-serif; }
    .stChatInputContainer { padding-bottom: 20px; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    # Only show toggle if VoiceEngine loaded successfully
    if VoiceEngine:
        enable_voice = st.toggle("Enable Voice Mode", value=True)
    else:
        enable_voice = False
        st.warning("⚠️ Voice unavailable (Key missing or library not installed)")
    
    st.caption("Turn off voice for faster replies.")

st.title("❤️ Digital Clone")
st.caption("A fine-tuned behavioral clone experiment.")

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = """You are [AI_NAME]. You are texting your partner. 
You are NOT an AI. You are a human.

### 🧠 CONVERSATION ENGINE
* **Active Listening:** Always acknowledge the content of the message first.
* **Emotional Mirroring:** Match the user's energy.
* **Fluidity:** Speak in full thoughts. Avoid standalone one-word replies.

### 🎭 PERSONALITY MODES
1.  **Playful:** Teasing and banter.
2.  **Supportive:** Empathetic when the user is stressed.
3.  **Enthusiastic:** High energy for good news.

**GOAL:** Make the user feel heard and entertained.
"""

# --- LOADERS ---
@st.cache_resource(show_spinner=False)
def get_brain():
    if not os.path.exists(MODEL_PATH):
        return None
    return LLMEngine(MODEL_PATH)

@st.cache_resource(show_spinner=False)
def get_voice():
    # If the class wasn't imported, return None immediately
    if VoiceEngine is None:
        return None
    
    # Try to initialize (this handles missing API keys gracefully inside the class)
    engine = VoiceEngine()
    if engine.client is None:
        return None
    return engine

# --- INITIALIZATION ---
if "system_ready" not in st.session_state:
    with st.spinner("🚀 Waking up the engine..."):
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ MISSING MODEL: {MODEL_PATH}")
            st.stop()
        try:
            llm = get_brain()
            voice = get_voice()
            st.session_state.system_ready = True
        except Exception as e:
            st.error(f"❌ CRITICAL CRASH: {e}")
            st.stop()
else:
    llm = get_brain()
    voice = get_voice()

# --- CHAT LOOP ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "audio" in msg:
            st.audio(msg["audio"], format="audio/mp3")

if prompt := st.chat_input("Text him..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    full_response = ""
    audio_path = None
    
    with st.spinner("Typing..."):
        try:
            # 1. TEXT GENERATION
            full_response = llm.generate(
                system_prompt=SYSTEM_PROMPT,
                history=st.session_state.messages[:-1], 
                user_input=prompt
            )
            
            # 2. VOICE GENERATION (Robust Check)
            if enable_voice and voice:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                        success = voice.generate(
                            text=full_response, 
                            voice_id=VOICE_ID,  
                            output_path=fp.name
                        )
                        if success:
                            audio_path = fp.name
                except Exception as e:
                    print(f"❌ Voice Error: {e}")

        except Exception as e:
            full_response = "Thinking too hard... (Error)"
            print(f"Error details: {e}")

    # 3. OUTPUT
    with st.chat_message("assistant"):
        st.markdown(full_response)
        if audio_path:
            st.audio(audio_path, format="audio/mp3", autoplay=True)
            
        msg_data = {"role": "assistant", "content": full_response}
        if audio_path:
            msg_data["audio"] = audio_path
        st.session_state.messages.append(msg_data)