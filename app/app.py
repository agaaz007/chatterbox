import os
from flask import Flask, request, Response, jsonify
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the Chatterbox TTS model
try:
    logger.info("Loading ChatterboxTTS model...")
    model = ChatterboxTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
    logger.info("ChatterboxTTS model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

@app.route('/ping', methods=['GET'])
def ping():
    """Health check endpoint for SageMaker."""
    return jsonify(status='ok'), 200

@app.route('/invocations', methods=['POST'])
def invocations():
    """Main endpoint for TTS synthesis."""
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415

    data = request.get_json()
    text = data.get('text')
    audio_prompt_path = data.get('audio_prompt_path', 'app/Sector 20 10.wav')
    exaggeration = float(data.get('exaggeration', 0.7))
    cfg_weight = float(data.get('cfg_weight', 0.3))
    chunk_size = int(data.get('chunk_size', 25))

    if not text:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    if not os.path.exists(audio_prompt_path):
        return jsonify({"error": f"Audio prompt file not found at {audio_prompt_path}"}), 400

    def generate():
        try:
            for audio_chunk, metrics in model.generate_stream(
                text,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                chunk_size=chunk_size
            ):
                # Convert the tensor to bytes
                yield audio_chunk.cpu().numpy().tobytes()
        except Exception as e:
            logger.error(f"Error during audio generation: {e}")

    return Response(generate(), mimetype='audio/wav')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)