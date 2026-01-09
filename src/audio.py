import onnxruntime as ort
import numpy as np

class AudioEncoder:
    def __init__(self, model_path):
        # Load INT8 Quantized Model
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
    def preprocess(self, audio_bytes):
        # Convert bytes to float32 array (simplified)
        # Real impl would use librosa/soundfile to decode WAV
        audio_input = np.frombuffer(audio_bytes, dtype=np.float32)
        
        # Pad/Trim to window size
        # Run Inference
        ort_inputs = {self.session.get_inputs()[0].name: audio_input[None, :]} # Add batch dim
        features = self.session.run(None, ort_inputs)[0]
        
        return torch.tensor(features, device='cuda')