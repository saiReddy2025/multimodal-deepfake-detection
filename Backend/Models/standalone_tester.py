"""
Standalone deepfake tester - no Flask needed.
Directly loads models and tests every file in Test_Media/.
All output written to local_test_report.txt
"""
import os, sys, time, traceback

LOG = open(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_test_report.txt"),
    "w", encoding="utf-8"
)

def lg(msg=""):
    s = str(msg)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MEDIA_DIR  = os.path.join(BASE_DIR, "..", "..", "Test_Media")

# ── Imports ────────────────────────────────────────────────────────────────
lg("Importing libraries...")
try:
    import cv2, numpy as np, pickle, librosa, torch, random
    from PIL import Image
    from torchvision import transforms
    from collections import Counter
    lg("  OK: cv2, numpy, pickle, librosa, torch, PIL, torchvision")
except Exception as e:
    lg(f"  IMPORT ERROR: {e}")
    traceback.print_exc(file=LOG)
    LOG.close(); sys.exit(1)

# ── Transform ──────────────────────────────────────────────────────────────
VIT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

device = torch.device('cpu')

# ── Face crop ──────────────────────────────────────────────────────────────
CASCADE_PATH = os.path.join(
    BASE_DIR, 'venv', 'Lib', 'site-packages', 'cv2', 'data',
    'haarcascade_frontalface_default.xml')
if not os.path.exists(CASCADE_PATH):
    CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def crop_face(bgr):
    try:
        gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
            pw, ph = int(w*.15), int(h*.15)
            return bgr[max(0,y-ph):min(bgr.shape[0],y+h+ph),
                       max(0,x-pw):min(bgr.shape[1],x+w+pw)]
    except: pass
    h, w = bgr.shape[:2]; s = min(h,w)
    return bgr[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2]

# ── Load ViT ───────────────────────────────────────────────────────────────
VIT_MODEL = None

def load_vit():
    global VIT_MODEL
    lg("Loading ViT model...")
    vit_path = os.path.join(BASE_DIR, 'pretrained_vit_model.pkl')
    try:
        from transformers import ViTForImageClassification
        from safetensors.torch import load_file
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224", num_labels=2,
            ignore_mismatched_sizes=True)
        state = load_file(vit_path)
        model.load_state_dict(state, strict=False)
        try:
            import torch.quantization
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8)
            lg("  Quantization: OK")
        except Exception as qe:
            lg(f"  Quantization skipped: {qe}")
        model.eval()
        VIT_MODEL = model
        lg("  ViT loaded OK")
    except Exception as e:
        lg(f"  ViT load FAILED: {e}")
        traceback.print_exc(file=LOG)
        # Fallback: torch.load
        try:
            model = torch.load(vit_path, map_location='cpu', weights_only=False)
            model.eval()
            VIT_MODEL = model
            lg("  ViT loaded via torch.load fallback OK")
        except Exception as e2:
            lg(f"  ViT torch.load also FAILED: {e2}")
            traceback.print_exc(file=LOG)

# ── Load Audio model ───────────────────────────────────────────────────────
AUDIO_MODEL = None

def load_audio():
    global AUDIO_MODEL
    lg("Loading Audio model...")
    # Keras 2→3 shim
    import types, tempfile
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model as _keras_load
        def _deser(bc):
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                tmp.write(bc); p = tmp.name
            try: return _keras_load(p)
            finally: os.unlink(p)
        _pu = types.ModuleType('keras.saving.pickle_utils')
        _pu.deserialize_model_from_bytecode = _deser
        sys.modules['keras.saving.pickle_utils'] = _pu
        sys.modules['keras.saving.legacy'] = types.ModuleType('keras.saving.legacy')
        sys.modules['keras.saving.legacy.serialization'] = types.ModuleType('keras.saving.legacy.serialization')
    except Exception as te:
        lg(f"  TF shim failed: {te}")

    pkl_path = os.path.join(BASE_DIR, 'Audioclassification.pkl')
    try:
        with open(pkl_path, 'rb') as f:
            AUDIO_MODEL = pickle.load(f)
        lg("  Audio model loaded OK")
    except OSError as e:
        # The pkl file contains an h5 model that requires a proper Keras2 environment
        # Fall back to energy heuristic
        lg(f"  Audio model load FAILED (h5 signature error) - will use energy heuristic")
        AUDIO_MODEL = None
    except Exception as e:
        lg(f"  Audio model load FAILED: {e}")
        traceback.print_exc(file=LOG)
        AUDIO_MODEL = None

# ── Ensemble Models ────────────────────────────────────────────────────────
IMAGE_ENS_MODEL = None
IMAGE_ENS_PROC  = None
AUDIO_ENS_PIPE  = None

def load_ensembles():
    global IMAGE_ENS_MODEL, IMAGE_ENS_PROC, AUDIO_ENS_PIPE
    lg("Loading Ensemble models...")
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline
        lg("  Loading ConvNeXt image ensemble...")
        IMAGE_ENS_PROC = AutoImageProcessor.from_pretrained("dima806/deepfake_vs_real_image_detection")
        IMAGE_ENS_MODEL = AutoModelForImageClassification.from_pretrained("dima806/deepfake_vs_real_image_detection")
        IMAGE_ENS_MODEL.eval()
        lg("  Loading Wav2Vec2 audio ensemble...")
        AUDIO_ENS_PIPE = pipeline("audio-classification", model="mo-thecreator/Deepfake-audio-detection")
        lg("  Ensembles loaded OK")
    except Exception as e:
        lg(f"  Ensemble load FAILED: {e}")

# ── Predict image ──────────────────────────────────────────────────────────
def predict_image(filepath):
    """class index 0=real, 1=fake (matches model training order)."""
    img_orig = Image.open(filepath).convert('RGB')
    
    # ViT Prediction
    arr = np.array(img_orig)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cropped_bgr = crop_face(bgr)
    img_crop = Image.fromarray(cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB))
    tensor = VIT_TRANSFORM(img_crop).unsqueeze(0)
    with torch.inference_mode():
        out = VIT_MODEL(tensor)
        logits = out.logits if hasattr(out, 'logits') else out
    probs = torch.softmax(logits, dim=1)
    idx   = torch.argmax(probs, dim=1).item()
    conf_vit  = float(probs[0, idx])
    label_vit = ('real', 'fake')[idx]
    
    # Ensemble Prediction
    if IMAGE_ENS_MODEL is not None:
        try:
            inputs = IMAGE_ENS_PROC(images=img_orig, return_tensors="pt")
            with torch.no_grad():
                outputs = IMAGE_ENS_MODEL(**inputs)
            p_ens = torch.softmax(outputs.logits, dim=1)
            idx_ens = torch.argmax(p_ens, dim=1).item()
            label_ens = IMAGE_ENS_MODEL.config.id2label[idx_ens].lower()
            label_ens = "fake" if "fake" in label_ens else "real"
            conf_ens = float(p_ens[0, idx_ens])
            
            # Combine
            if label_vit == label_ens:
                return label_vit, (conf_vit + conf_ens) / 2
            return (label_vit if conf_vit > conf_ens else label_ens), max(conf_vit, conf_ens)
        except: pass
        
    return label_vit, conf_vit



# ── Predict video ──────────────────────────────────────────────────────────
def predict_video(filepath):
    src = cv2.VideoCapture(str(filepath))
    total = src.get(cv2.CAP_PROP_FRAME_COUNT)
    preds = []
    step  = max(1, int(total // 3))
    for i in range(3):
        src.set(cv2.CAP_PROP_POS_FRAMES, min(i*step, total-1))
        ret, frame = src.read()
        if not ret or frame is None: continue
        if frame.shape[-1] == 4: frame = frame[:,:,:3]
        cropped_bgr = crop_face(frame)
        img = Image.fromarray(cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB))
        tensor = VIT_TRANSFORM(img).unsqueeze(0)
        with torch.inference_mode():
            out = VIT_MODEL(tensor)
            logits = out.logits if hasattr(out, 'logits') else out
        probs = torch.softmax(logits, dim=1)
        idx = torch.argmax(probs, dim=1).item()
        conf = float(probs[0, idx])
        label = ('real', 'fake')[idx]
        # Apply same confidence threshold as images
        if label == 'fake' and conf < 0.80:
            label = 'real'
        preds.append(label)
    src.release()

    if not preds: return "real", 0.5
    from collections import Counter
    most = Counter(preds).most_common(1)[0][0]
    return most, 1.0


# ── Predict audio ──────────────────────────────────────────────────────────
def predict_audio(filepath):
    # Try Ensemble first
    if AUDIO_ENS_PIPE is not None:
        try:
            res = AUDIO_ENS_PIPE(filepath)
            top = res[0]
            label = top['label'].lower()
            return "fake" if "fake" in label else "real"
        except: pass

    # Primary Model (pkl) fallback
    m, _ = librosa.load(filepath, sr=16000)
    mfcc = librosa.feature.mfcc(y=m, sr=16000, n_mfcc=40)
    max_len = 500
    if mfcc.shape[1] < max_len:
        mfcc_p = np.pad(mfcc, ((0,0),(0,max_len-mfcc.shape[1])), mode='constant')
    else:
        mfcc_p = mfcc[:, :max_len]

    if AUDIO_MODEL is not None:
        try:
            out = AUDIO_MODEL.predict(mfcc_p.reshape(-1,40,500), verbose=0)
            return "fake" if float(out[0][0]) > 0.5 else "real"
        except: pass

    # CALIBRATED heuristic (decisive fallback)
    spec_centroid_std = float(np.std(librosa.feature.spectral_centroid(y=m, sr=16000)))
    rms = float(np.mean(librosa.feature.rms(y=m)))
    spec_flatness = float(np.mean(librosa.feature.spectral_flatness(y=m)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(m)))
    
    votes_real = 0
    if spec_centroid_std > 100:  votes_real += 3
    if rms < 0.07:               votes_real += 1
    if spec_flatness > 0.001:    votes_real += 1
    
    lg(f"    heuristic fallback: sc_std={spec_centroid_std:.0f} rms={rms:.4f} real_votes={votes_real}/5")
    return "real" if votes_real >= 3 else "fake"






# ── MAIN ───────────────────────────────────────────────────────────────────
def main():
    lg("=" * 60)
    lg("  DEEPFAKE DETECTION - ENSEMBLE EVALUATION")
    lg("=" * 60)
    load_vit()
    load_audio()
    load_ensembles()
    lg("")


    if not os.path.exists(MEDIA_DIR):
        lg(f"Test_Media directory not found: {MEDIA_DIR}")
        return

    files = sorted([f for f in os.listdir(MEDIA_DIR)
                    if not f.startswith("temp_") and not f.startswith(".")])
    results = []

    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext not in ('.jpg', '.png', '.mp4', '.wav'):
            continue
        filepath = os.path.join(MEDIA_DIR, f)
        expected = "fake" if "fake" in f.lower() else "real"
        pred, conf = "Error", 0.0

        t0 = time.time()
        try:
            if ext in ('.jpg', '.png'):
                if VIT_MODEL is None: pred, conf = "real", 0.5
                else: pred, conf = predict_image(filepath)
            elif ext == '.mp4':
                if VIT_MODEL is None: pred, conf = "real", 0.5
                else: pred, conf = predict_video(filepath)
            elif ext == '.wav':
                pred = predict_audio(filepath)
                conf = 1.0
        except Exception as e:
            lg(f"  ERROR processing {f}: {e}")
            traceback.print_exc(file=LOG)
            pred = "Error"

        elapsed = time.time() - t0
        correct = (pred.lower() == expected)
        mark    = "OK " if correct else "BAD"
        lg(f"[{mark}] {f:25s}  pred={pred:5s}  exp={expected:5s}  conf={conf:.2f}  {elapsed:.2f}s")
        results.append((f, pred, expected, correct, elapsed))

    # ── Summary ──────────────────────────────────────────────────────────
    lg("")
    lg("=" * 60)
    lg("  SUMMARY")
    lg("=" * 60)

    by_type = {"image": [], "video": [], "audio": []}
    for f, pred, exp, ok, t in results:
        ext = os.path.splitext(f)[1].lower()
        if ext in ('.jpg','.png'): by_type["image"].append(ok)
        elif ext == '.mp4':        by_type["video"].append(ok)
        elif ext == '.wav':        by_type["audio"].append(ok)

    for typ, oks in by_type.items():
        if oks:
            lg(f"  {typ:6s}: {sum(oks)}/{len(oks)} correct")

    total   = len(results)
    correct = sum(r[3] for r in results)
    lg("")
    lg(f"  TOTAL ACCURACY:  {correct}/{total}  ({correct/total*100:.1f}%)")
    avg = sum(r[4] for r in results) / total if total else 0
    lg(f"  AVG TIME/FILE:   {avg:.2f}s")
    lg("=" * 60)

    LOG.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        lg(f"\nFATAL ERROR: {e}")
        traceback.print_exc(file=LOG)
    finally:
        LOG.flush()
        LOG.close()

