import subprocess
import os
import time
import sys

# Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "Backend", "Models")
FRONTEND_DIR = os.path.join(ROOT_DIR, "Frontend")

PYTHON_EXE = os.path.join(BACKEND_DIR, "venv", "Scripts", "python.exe")
NODE_EXE = r"C:\Program Files\nodejs\node.exe"
VITE_BIN = os.path.join(FRONTEND_DIR, "node_modules", "vite", "bin", "vite.js")

# Environment for backend: suppress Intel MKL forrtl crashes
BACKEND_ENV = os.environ.copy()
BACKEND_ENV.update({
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "KMP_BLOCKTIME": "1",
    "OMP_NUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
    "TF_ENABLE_ONEDNN_OPTS": "0",  # quieter TensorFlow logs
    "TF_CPP_MIN_LOG_LEVEL": "2",
})


def start_backend():
    print("Starting Backend...")
    log_path = os.path.join(ROOT_DIR, "backend_run_log.txt")
    log_file = open(log_path, "w", buffering=1)
    log_file.write(f"Starting backend at {time.ctime()}\n")
    log_file.flush()

    # Use CREATE_NEW_PROCESS_GROUP + DETACHED_PROCESS so the backend survives
    # if the parent terminal/console window is closed (prevents forrtl error 200).
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    DETACHED_PROCESS = 0x00000008

    proc = subprocess.Popen(
        [PYTHON_EXE, "backend.py"],
        cwd=BACKEND_DIR,
        stdout=log_file,
        stderr=log_file,
        env=BACKEND_ENV,
        creationflags=CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS,
    )
    return proc, log_file


def start_frontend():
    print("Starting Frontend...")
    log_path = os.path.join(ROOT_DIR, "frontend_run_log.txt")
    log_file = open(log_path, "w", buffering=1)
    return subprocess.Popen(
        [NODE_EXE, VITE_BIN],
        cwd=FRONTEND_DIR,
        stdout=log_file,
        stderr=log_file,
    ), log_file


if __name__ == "__main__":
    if not os.path.exists(PYTHON_EXE):
        print(f"Error: Backend venv not found at {PYTHON_EXE}")
        sys.exit(1)

    backend_proc, backend_log = start_backend()
    frontend_proc, frontend_log = start_frontend()

    print("\n--- Project is running ---")
    print(f"Backend API: http://127.0.0.1:5001")
    print(f"Frontend:    http://localhost:5173")
    print(f"Backend log: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend_run_log.txt')}")
    print("\nPress Ctrl+C to stop both servers.")

    try:
        while True:
            if backend_proc.poll() is not None:
                rc = backend_proc.poll()
                print(f"Backend process exited (code={rc})! Check backend_run_log.txt")
                break
            if frontend_proc.poll() is not None:
                rc = frontend_proc.poll()
                print(f"Frontend process exited (code={rc})!")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping servers...")
        backend_proc.terminate()
        frontend_proc.terminate()
        backend_log.close()
        frontend_log.close()
        print("Done.")
