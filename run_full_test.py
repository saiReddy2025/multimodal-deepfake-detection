import os
import sys
import time
import requests

API_URL = "http://127.0.0.1:5001/upload"
TEST_MEDIA_DIR = "Test_Media"

RESULTS = []

def log(msg, outf=None):
    print(msg, flush=True)
    sys.stdout.flush()
    if outf:
        outf.write(msg + "\n")
        outf.flush()

def test_file(filepath, outf):
    filename = os.path.basename(filepath)
    if "temp_" in filename:
        return None  # skip temp files
    if filename.endswith((".jpg", ".png")):
        ftype = "image"
    elif filename.endswith(".mp4"):
        ftype = "video"
    elif filename.endswith(".wav"):
        ftype = "audio"
    else:
        return None

    t0 = time.time()
    try:
        with open(filepath, "rb") as fh:
            r = requests.post(API_URL, files={ftype: (filename, fh)}, timeout=120)
        elapsed = time.time() - t0

        if r.status_code == 200:
            data = r.json()
            pred = data[1] if isinstance(data, list) and len(data) > 1 else "Unknown"
            backend_time = data[0].get("prediction_time_s", "?") if isinstance(data, list) else "?"
            log(f"  [OK] {filename:25s} => {pred.upper():5s}  total={elapsed:.2f}s  backend={backend_time}s", outf)
            return (filename, pred, elapsed)
        else:
            log(f"  [ERR {r.status_code}] {filename}", outf)
            return (filename, "error", time.time() - t0)

    except requests.exceptions.ConnectionError:
        log(f"  [CONNECTION ERROR] Could not reach backend for {filename}", outf)
        return None
    except Exception as e:
        log(f"  [EXCEPTION] {filename}: {e}", outf)
        return (filename, "error", time.time() - t0)


def main():
    output_file = "full_test_results.txt"
    with open(output_file, "w", encoding="utf-8") as outf:
        log(f"=== Deepfake Detection Full Test ===", outf)
        log(f"Backend: {API_URL}", outf)
        log(f"Test Media: {os.path.abspath(TEST_MEDIA_DIR)}", outf)
        log("", outf)

        if not os.path.exists(TEST_MEDIA_DIR):
            log(f"ERROR: {TEST_MEDIA_DIR} not found!", outf)
            return

        all_files = sorted(f for f in os.listdir(TEST_MEDIA_DIR)
                           if not f.startswith("temp_") and not f.startswith("."))

        images = [f for f in all_files if f.endswith((".jpg", ".png"))]
        audios = [f for f in all_files if f.endswith(".wav")]
        videos = [f for f in all_files if f.endswith(".mp4")]

        results = []

        for category, files in [("IMAGES", images), ("AUDIO", audios), ("VIDEOS", videos)]:
            log(f"--- {category} ({len(files)} files) ---", outf)
            for fname in files:
                fpath = os.path.join(TEST_MEDIA_DIR, fname)
                res = test_file(fpath, outf)
                if res is None:  # connection error
                    log("  Stopping due to connection error.", outf)
                    break
                if isinstance(res, tuple):
                    results.append(res)
            log("", outf)

        log("=== SUMMARY ===", outf)
        correct = 0
        total = len(results)
        for filename, pred, elapsed in results:
            expected = "fake" if "fake" in filename.lower() else "real"
            ok = pred.lower() == expected
            if ok:
                correct += 1
            mark = "✓" if ok else "✗"
            log(f"  {mark} {filename:28s} | Expected: {expected:5s} | Got: {pred:5s} | {elapsed:.2f}s", outf)

        if total > 0:
            accuracy = (correct / total) * 100
            avg_time = sum(r[2] for r in results) / total
            log("", outf)
            log(f"  Accuracy : {correct}/{total} = {accuracy:.1f}%", outf)
            log(f"  Avg Time : {avg_time:.2f}s per file", outf)
        else:
            log("  No results recorded.", outf)

    print(f"\nResults saved to: {os.path.abspath(output_file)}", flush=True)


if __name__ == "__main__":
    main()
