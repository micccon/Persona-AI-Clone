import os
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

class VoiceEngine:
    def __init__(self):
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            print("❌ ERROR: ELEVENLABS_API_KEY not found in .env")
            self.client = None
        else:
            try:
                self.client = ElevenLabs(api_key=api_key)
                print("✅ ElevenLabs Connected")
            except Exception as e:
                print(f"❌ Connection Error: {e}")
                self.client = None

    def generate(self, text, voice_id, output_path):
        if self.client is None: return False
        try:
            audio_stream = self.client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id="eleven_turbo_v2_5",
                output_format="mp3_44100_128",
                voice_settings=VoiceSettings(
                    stability=0.4,       
                    similarity_boost=0.8, 
                    style=0.5,           
                    use_speaker_boost=True 
                )
            )
            with open(output_path, "wb") as f:
                for chunk in audio_stream:
                    if chunk: f.write(chunk)    
            return True
        except Exception as e:
            print(f"❌ ElevenLabs Generation Error: {e}")
            return False