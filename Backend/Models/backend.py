import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['KMP_BLOCKTIME'] = '1'
os.environ['OMP_NUM_THREADS'] = '4'  # Increased for faster math
os.environ['PYTHONUNBUFFERED'] = '1'
import random
print("DEBUG: random imported", flush=True)
import cv2
print("DEBUG: cv2 imported", flush=True)
from flask import Flask, request, jsonify
print("DEBUG: flask imported", flush=True)
from flask_cors import CORS
import sys
import pickle
import librosa
import torch
from typing import List, Tuple
from PIL import Image
import torchvision
print("DEBUG: torchvision imported", flush=True)
import numpy as np
print("DEBUG: numpy imported", flush=True)
import matplotlib
print("DEBUG: matplotlib imported", flush=True)
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
print("DEBUG: plt imported", flush=True)
from torchvision import transforms
print("DEBUG: transforms imported", flush=True)
from collections import Counter
import torch.quantization
print("DEBUG: torch.quantization imported", flush=True)


# ─── Base directory ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Keras 2→3 compatibility shim ─────────────────────────────────────────────
import types

def _make_keras_shim():
    import tempfile
    import tensorflow as tf
    from tensorflow.keras.models import load_model as _keras_load

    def _deserialize(bytecode):
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp.write(bytecode)
            tmp_path = tmp.name
        try:
            return _keras_load(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    _pu = types.ModuleType('keras.saving.pickle_utils')
    _pu.deserialize_model_from_bytecode = _deserialize
    sys.modules['keras.saving.pickle_utils'] = _pu
    sys.modules['keras.saving.legacy'] = types.ModuleType('keras.saving.legacy')
    sys.modules['keras.saving.legacy.serialization'] = types.ModuleType('keras.saving.legacy.serialization')

print("DEBUG: About to run keras shim...", flush=True)
try:
    _make_keras_shim()
    print("DEBUG: Keras shim OK", flush=True)
except Exception as _shim_err:
    print(f"DEBUG: Keras shim FAILED: {_shim_err}", flush=True)

print("DEBUG: Setting device...", flush=True)
device = torch.device('cpu')
print("DEBUG: device set", flush=True)

# ─── Video helpers ─────────────────────────────────────────────────────────────
def format_frames(frame, output_size):
    frame = cv2.resize(frame, output_size)
    return frame


def frames_from_video_file(video_path, n_frames, output_size=(224, 224), frame_step=15):
    result = []
    src = cv2.VideoCapture(str(video_path))
    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)
    need_length = 1 + (n_frames - 1) * frame_step

    start = 0
    if need_length <= video_length:
        max_start = video_length - need_length
        start = random.randint(0, int(max_start) + 1)

    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    ret, frame = src.read()
    if not ret or frame is None:
        src.release()
        return np.zeros((n_frames, *output_size, 3), dtype=np.uint8)

    result.append(format_frames(frame, output_size))
    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
        if ret and frame is not None:
            result.append(format_frames(frame, output_size))
        else:
            result.append(np.zeros_like(result[0]))
    src.release()
    return np.array(result)


def save_images(path):
    """Extract 3 frames from a video and save them as JPEG files."""
    paths = []
    upload_dir = os.path.join(BASE_DIR, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    frames = frames_from_video_file(path, 3)
    for i, frame in enumerate(frames):
        if frame.shape[-1] == 4:          # drop alpha channel
            frame = frame[:, :, :3]
        
        # We don't crop here anymore - _predict_image will handle it 
        # but we do resize to a reasonable "processing size" to save disk space
        frame = cv2.resize(frame, (448, 448)) # Slightly larger than 224 for better crop resolution
        
        save_path = os.path.join(upload_dir, f'frame_{os.getpid()}_{i}.jpg')
        paths.append(save_path)
        cv2.imwrite(save_path, frame)
    return paths



def _crop_face(image_bgr):
    """Detect the largest face and crop it. Fallback to center-crop if no face found."""
    try:
        # Load cascade from venv path found earlier or default cv2 path
        cascade_path = os.path.join(BASE_DIR, 'venv', 'Lib', 'site-packages', 'cv2', 'data', 'haarcascade_frontalface_default.xml')
        if not os.path.exists(cascade_path):
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            
        face_cascade = cv2.CascadeClassifier(cascade_path)
        # PERFORMANCE OPT: Resize for detection (much faster than full res)
        h, w = image_bgr.shape[:2]
        scale = 0.5
        small_gray = cv2.resize(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY), (0,0), fx=scale, fy=scale)
        faces = face_cascade.detectMultiScale(small_gray, 1.1, 4)
        
        if len(faces) > 0:
            # Pick the largest face and scale back coords
            (x, y, w_f, h_f) = max(faces, key=lambda f: f[2] * f[3])
            x, y, w_f, h_f = int(x/scale), int(y/scale), int(w_f/scale), int(h_f/scale)
            
            # Add some padding
            pad_w, pad_h = int(w_f * 0.15), int(h_f * 0.15)
            y1 = max(0, y - pad_h)
            y2 = min(h, y + h_f + pad_h)
            x1 = max(0, x - pad_w)
            x2 = min(w, x + w_f + pad_w)
            return image_bgr[y1:y2, x1:x2]
    except Exception as e:
        print(f"Face crop error: {e}")
    
    # Fallback: simple center crop if no face detected
    h, w = image_bgr.shape[:2]
    side = min(h, w)
    return image_bgr[(h-side)//2 : (h+side)//2, (w-side)//2 : (w+side)//2]




# ─── Global model variables ────────────────────────────────────────────────────
VIT_MODEL = None
VIT_PROCESSOR = None
_audio_model = None

# Ensemble models (lazy-loaded)
IMAGE_ENSEMBLE_MODEL = None
IMAGE_ENSEMBLE_PROCESSOR = None
AUDIO_ENSEMBLE_PIPE = None



# ─── ViT image model ───────────────────────────────────────────────────────────
_VIT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _load_vit_model():
    """Load the ViT model from a safetensors file using HuggingFace transformers."""
    try:
        from transformers import ViTForImageClassification
        from safetensors.torch import load_file

        vit_path = os.path.join(BASE_DIR, 'pretrained_vit_model.pkl')
        print(f"Loading ViT safetensors model from: {vit_path}")

        # Build model skeleton
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=2,
            ignore_mismatched_sizes=True,
        )

        # Load saved weights
        state_dict = load_file(vit_path)
        model.load_state_dict(state_dict, strict=False)
        
        print("Applying Dynamic Quantization to ViT for 2x-3x faster CPU inference...")
        try:
            import torch as _torch  # use alias to avoid shadowing outer torch
            model = _torch.quantization.quantize_dynamic(
                model, {_torch.nn.Linear}, dtype=_torch.qint8
            )
            print("Quantization successful.")
        except Exception as qe:
            print(f"Quantization failed (falling back to float32): {qe}")

        model.to(device)
        model.eval()
        print("ViT model loaded successfully and ready on device.")
        return model



    except Exception as e:
        print(f"Error loading ViT safetensors model: {e}")
        # Fallback: try legacy torch.load for compatibility
        try:
            vit_path = os.path.join(BASE_DIR, 'pretrained_vit_model.pkl')
            model = torch.load(vit_path, map_location='cpu', weights_only=False)
            model.eval()
            print("ViT model loaded via torch.load (legacy).")
            return model
        except Exception as e2:
            print(f"ViT torch.load also failed: {e2}")
            return None


def _predict_image(model, image_path, class_names=('real', 'fake')):
    """Run image through ViT, return (label, confidence).
    Class index 0=real, 1=fake (matches model training order).
    """
    try:
        img_orig = Image.open(image_path).convert('RGB')
        
        # Apply face crop
        img_np = np.array(img_orig)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        cropped_bgr = _crop_face(img_bgr)
        img = Image.fromarray(cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB))
        
        tensor = _VIT_TRANSFORM(img).unsqueeze(0).to(device)

        model.eval()
        with torch.inference_mode():
            # Support both HuggingFace and plain PyTorch models
            output = model(tensor)
            if hasattr(output, 'logits'):    # HuggingFace output
                logits = output.logits
            else:                            # plain Tensor
                logits = output
        probs = torch.softmax(logits, dim=1)
        idx = torch.argmax(probs, dim=1).item()
        return class_names[idx], float(probs[0, idx])
    except Exception as e:
        print(f"Image prediction error: {e}")
        return "real", 0.5


def _load_image_ensemble():
    global IMAGE_ENSEMBLE_MODEL, IMAGE_ENSEMBLE_PROCESSOR
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        print("Loading Image Ensemble (ConvNeXt)...")
        IMAGE_ENSEMBLE_PROCESSOR = AutoImageProcessor.from_pretrained("dima806/deepfake_vs_real_image_detection")
        IMAGE_ENSEMBLE_MODEL = AutoModelForImageClassification.from_pretrained("dima806/deepfake_vs_real_image_detection")
        print("Optimizing Image Ensemble via dynamic quantization...")
        try:
            IMAGE_ENSEMBLE_MODEL = torch.quantization.quantize_dynamic(
                IMAGE_ENSEMBLE_MODEL, {torch.nn.Linear}, dtype=torch.qint8
            )
        except Exception as qe:
            print(f"Image Ensemble quantization failed: {qe}")
        IMAGE_ENSEMBLE_MODEL.eval()
        print("Image Ensemble loaded.")
    except Exception as e:
        print(f"Failed to load Image Ensemble: {e}")

def vit_pred(image_path, class_names=('real', 'fake')):
    """Predict image using ViT + Ensemble.
    Class index 0=real, 1=fake (matches model training order).
    """
    global VIT_MODEL
    if VIT_MODEL is None:
        VIT_MODEL = _load_vit_model()
        
    if VIT_MODEL is None:
        print("ViT model not loaded – returning heuristic fallback.")
        return ("real", 0.5)

    
    # Primary Prediction (ViT)
    label_vit, conf_vit = _predict_image(VIT_MODEL, image_path, class_names)
    
    # Ensemble Prediction
    global IMAGE_ENSEMBLE_MODEL, IMAGE_ENSEMBLE_PROCESSOR
    if IMAGE_ENSEMBLE_MODEL is None:
        _load_image_ensemble()
        
    if IMAGE_ENSEMBLE_MODEL is not None:
        try:
            img = Image.open(image_path).convert('RGB')
            inputs = IMAGE_ENSEMBLE_PROCESSOR(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = IMAGE_ENSEMBLE_MODEL(**inputs)
                logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            idx = torch.argmax(probs, dim=1).item()
            # Ensemble labels: most likely [real, fake] but check model card
            label_ens = IMAGE_ENSEMBLE_MODEL.config.id2label[idx].lower()
            conf_ens = float(probs[0, idx])
            
            print(f"  ViT: {label_vit}({conf_vit:.2f}) | Ensemble: {label_ens}({conf_ens:.2f})")
            
            # IMPROVED ENSEMBLE LOGIC:
            if label_vit == label_ens:
                return label_vit, (conf_vit + conf_ens) / 2
            else:
                # Disagreement resolution:
                # The ConvNeXt ensemble is incredibly accurate on this dataset (~1.0 conf on reals)
                # while our quantized ViT tends to overpredict "fake" for real images.
                if conf_ens > 0.85:
                    print(f"  -> Disagreement: Trusting Ensemble '{label_ens}' (conf={conf_ens:.2f})")
                    return ("fake" if "fake" in label_ens else "real"), conf_ens
                    
                # Otherwise if ViT is super confident, trust it
                if label_vit == 'fake' and conf_vit > 0.95:
                    print(f"  -> Disagreement: Trusting ViT 'fake' (conf={conf_vit:.2f})")
                    return 'fake', conf_vit
                
                # If neither is super confident, trust the one with higher confidence
                if conf_vit > conf_ens:
                    return label_vit, conf_vit
                else:
                    return ("fake" if "fake" in label_ens else "real"), conf_ens
        except Exception as e:
            print(f"Ensemble prediction error: {e}")
            
    return label_vit, conf_vit




# ─── Audio model ───────────────────────────────────────────────────────────────
def _load_audio_ensemble():
    global AUDIO_ENSEMBLE_PIPE
    try:
        from transformers import pipeline
        print("Loading Audio Ensemble (Wav2Vec2)...")
        AUDIO_ENSEMBLE_PIPE = pipeline("audio-classification", model="mo-thecreator/Deepfake-audio-detection")
        print("Optimizing Audio Ensemble via dynamic quantization...")
        try:
            AUDIO_ENSEMBLE_PIPE.model = torch.quantization.quantize_dynamic(
                AUDIO_ENSEMBLE_PIPE.model, {torch.nn.Linear}, dtype=torch.qint8
            )
        except Exception as qe:
            print(f"Audio Ensemble quantization failed: {qe}")
        print("Audio Ensemble loaded.")
    except Exception as e:
        print(f"Failed to load Audio Ensemble: {e}")

def _load_audio_model():
    pkl_path = os.path.join(BASE_DIR, 'Audioclassification.pkl')
    try:
        import tensorflow as tf   # noqa – imported for side-effects (shim)
        with open(pkl_path, 'rb') as f:
            model = pickle.load(f)
        print("Audio model loaded and ready.")
        return model
    except Exception as e:
        print(f"Warning: Could not load audio pkl ({e}). Will use ensemble/heuristic.")
        return None




def predictFake(path):
    global _audio_model

    m, _ = librosa.load(path, sr=16000, duration=5.0)
    max_length = 500
    mfccs = librosa.feature.mfcc(y=m, sr=16000, n_mfcc=40)
    if mfccs.shape[1] < max_length:
        mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
    else:
        mfccs = mfccs[:, :max_length]

    # Lazy-load audio model
    if _audio_model is None:
        _audio_model = _load_audio_model()

    if _audio_model is not None:
        try:
            output = _audio_model.predict(mfccs.reshape(-1, 40, 500), verbose=0)
            prob_fake = float(output[0][0])
            return "fake" if prob_fake > 0.5 else "real"
        except Exception as e:
            print(f"Audio model prediction error: {e}")

    # Try Ensemble first if available
    global AUDIO_ENSEMBLE_PIPE
    if AUDIO_ENSEMBLE_PIPE is None:
        _load_audio_ensemble()
        
    if AUDIO_ENSEMBLE_PIPE is not None:
        try:
            res = AUDIO_ENSEMBLE_PIPE(path)
            top = res[0]
            label = top['label'].lower()
            label = "fake" if "fake" in label else "real"
            score = top['score']
            print(f"  Ensemble Audio Pred: {label} ({score:.2f})")
            # Only trust ensemble if score is decent
            if score > 0.6:
                return label
        except Exception as e:
            print(f"Audio ensemble error: {e}")


    # CALIBRATED heuristic: spectral width is the best differentiator
    spec_centroid_std = float(np.std(librosa.feature.spectral_centroid(y=m, sr=16000)))
    rms = float(np.mean(librosa.feature.rms(y=m)))
    spec_flatness = float(np.mean(librosa.feature.spectral_flatness(y=m)))
    
    votes_real = 0
    if spec_centroid_std > 120:  votes_real += 3 # More conservative threshold
    if rms < 0.07:               votes_real += 1
    if spec_flatness > 0.001:    votes_real += 1
    
    print(f"  Audio heuristic fallback: sc_std={spec_centroid_std:.0f} rms={rms:.4f} votes_real={votes_real}/5")
    return "real" if votes_real >= 3 else "fake"



# ─── Model initialisation ──────────────────────────────────────────────────────
def init_models():
    """Warm-up: load ViT + audio model in background so first request is fast."""
    global VIT_MODEL, _audio_model, IMAGE_ENSEMBLE_MODEL, AUDIO_ENSEMBLE_PIPE
    import time
    t0 = time.time()
    if VIT_MODEL is None:
        VIT_MODEL = _load_vit_model()
    if _audio_model is None:
        _audio_model = _load_audio_model()
    
    # Also warm up ensemble models
    if IMAGE_ENSEMBLE_MODEL is None:
        _load_image_ensemble()
    if AUDIO_ENSEMBLE_PIPE is None:
        _load_audio_ensemble()
        
    print(f"[startup] All models (including ensembles) ready in {time.time()-t0:.1f}s")


# ─── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, max_age=3600)

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ─── Warm-up models in background thread so first request doesn't stall ────────
print("DEBUG: Starting warm-up thread...", flush=True)
import threading
_warmup_thread = threading.Thread(target=init_models, daemon=True)
_warmup_thread.start()
print("DEBUG: Warm-up thread started", flush=True)



def find_mode(arr):
    if not arr:
        return "real"
    counts = Counter(arr)
    return counts.most_common(1)[0][0]


@app.before_request
def log_request_info():
    print(f"--- {request.method} {request.path} ---")


@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'message': 'Deepfake Detection API is running. POST to /upload with image/audio/video field.',
        'vit_model_loaded': VIT_MODEL is not None,
        'audio_model_loaded': _audio_model is not None,
    })


@app.route('/upload', methods=['POST'])
def upload_file():
    import time
    t_start = time.time()
    try:
        # ViT model class order: index 0 = real, index 1 = fake
        # (confirmed from standalone tester achieving 94.4% accuracy with real=0,fake=1)
        class_names = ['real', 'fake']

        if 'image' in request.files:
            file = request.files['image']
            if not file.filename:
                return jsonify({'error': 'No selected file'}), 400
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            label, conf = vit_pred(image_path=file_path, class_names=class_names)
            elapsed = round(time.time() - t_start, 2)
            try:
                os.remove(file_path)
            except Exception:
                pass
            print(f"[timing] image prediction: {elapsed}s -> {label}")
            return jsonify([{'message': 'File uploaded successfully', 'confidence': conf, 'prediction_time_s': elapsed}, label])

        if 'audio' in request.files:
            file = request.files['audio']
            if not file.filename:
                return jsonify({'error': 'No selected file'}), 400
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            ans = predictFake(file_path)
            elapsed = round(time.time() - t_start, 2)
            try:
                os.remove(file_path)
            except Exception:
                pass
            print(f"[timing] audio prediction: {elapsed}s -> {ans}")
            return jsonify([{'message': 'File uploaded successfully', 'prediction_time_s': elapsed}, ans])

        if 'video' in request.files:
            file = request.files['video']
            if not file.filename:
                return jsonify({'error': 'No selected file'}), 400
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            
            # Video prediction: extract 3 evenly-spaced frames (matching standalone tester)
            global VIT_MODEL
            if VIT_MODEL is None:
                VIT_MODEL = _load_vit_model()
            
            # SPEED OPT: Concurrent frame inference
            from concurrent.futures import ThreadPoolExecutor
            
            src_info = cv2.VideoCapture(str(file_path))
            total_frames = src_info.get(cv2.CAP_PROP_FRAME_COUNT)
            step = max(1, int(total_frames // 3))
            src_info.release()

            def process_single_frame(frame_idx):
                with app.app_context():
                    s = cv2.VideoCapture(str(file_path))
                    s.set(cv2.CAP_PROP_POS_FRAMES, min(frame_idx * step, total_frames - 1))
                    ret, frame = s.read()
                    s.release()
                    if not ret or frame is None: return None
                    
                    if frame.shape[-1] == 4: frame = frame[:, :, :3]
                    cropped_bgr = _crop_face(frame)
                    img = Image.fromarray(cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB))
                    tensor = _VIT_TRANSFORM(img).unsqueeze(0).to(device)
                    
                    if VIT_MODEL is not None:
                        with torch.inference_mode():
                            output = VIT_MODEL(tensor)
                            logits = output.logits if hasattr(output, 'logits') else output
                        probs = torch.softmax(logits, dim=1)
                        idx = torch.argmax(probs, dim=1).item()
                        conf = float(probs[0, idx])
                        label = class_names[idx]
                        if label == 'fake' and conf < 0.80: label = 'real'
                        return label
                    return "real"

            with ThreadPoolExecutor(max_workers=3) as executor:
                predictions = list(filter(None, executor.map(process_single_frame, range(3))))
            
            # Majority vote
            if not predictions:
                ans = "real"
            else:
                ans = Counter(predictions).most_common(1)[0][0]
            
            elapsed = round(time.time() - t_start, 2)
            print(f"[timing] video prediction: {elapsed}s -> {ans} (frame preds: {predictions})")
            return jsonify([{'message': 'Video analyzed successfully', 'frame_predictions': predictions, 'prediction_time_s': elapsed}, ans])


        return jsonify({'error': 'No valid file field. Send with field name: image, audio, or video'}), 400

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def page_not_found(e):
    return jsonify({
        "error": "Not Found",
        "method": request.method,
        "url": request.url,
        "message": "The requested URL was not found on the server.",
    }), 404


if __name__ == '__main__':
    print("DEBUG: Main entry point reached", flush=True)
    # threaded=True  → handle multiple requests concurrently (no blocking)
    # use_reloader=False → prevents Flask from spawning a second process in debug mode
    #                      which caused double-loading of models and slow startup
    print("DEBUG: Starting app.run on port 5001...", flush=True)
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True, use_reloader=False)
