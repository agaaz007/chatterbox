import base64
import io
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from sagemaker.serve.builder.inference_spec import InferenceSpec

class ChatterboxTTSInferenceSpec(InferenceSpec):
    def load(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ChatterboxTTS.from_pretrained(device=self.device)
        self.sr = getattr(self.model, 'sr', 24000)  # Default to 24kHz if not set

    def predict(self, data):
        text = data.get("text")
        if not text:
            return {"error": "Missing 'text' in request body"}
        # Optionally support other params here
        wav = self.model.generate(text)
        buf = io.BytesIO()
        ta.save(buf, wav, self.sr, format="wav")
        audio_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"audio_base64": audio_base64} 