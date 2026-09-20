"""CUDA runtime loading shared by the app and isolated comparison workers."""
import os
import sys
import ctypes
from pathlib import Path

IS_WIN = sys.platform == "win32"

def _prepare_cuda_libs():
    """Expõe cublas/cudnn do wheel NVIDIA ao carregador nativo (DLL no Windows, .so no Linux)."""
    dirs = []
    for name in ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_nvrtc"):
        try:
            mod = __import__(name, fromlist=["*"])
        except ImportError:
            continue
        if getattr(mod, "__file__", None):
            root = Path(mod.__file__).resolve().parent
        elif getattr(mod, "__path__", None):
            root = Path(next(iter(mod.__path__))).resolve()
        else:
            continue
        for sub in ("bin", "lib", "lib64"):
            d = root / sub
            if d.is_dir():
                dirs.append(d)
    if not dirs:
        nvidia = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
        if not nvidia.is_dir():
            nvidia = (
                Path(sys.prefix) / "lib"
                / f"python{sys.version_info.major}.{sys.version_info.minor}"
                / "site-packages" / "nvidia"
            )
        for pkg in ("cublas", "cudnn", "cuda_nvrtc"):
            for sub in ("bin", "lib", "lib64"):
                d = nvidia / pkg / sub
                if d.is_dir():
                    dirs.append(d)
    for d in dirs:
        if IS_WIN:
            os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
            continue
        os.environ["LD_LIBRARY_PATH"] = str(d) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        pending = [p for p in d.iterdir() if p.is_file() and ".so" in p.name]
        for _ in range(4):
            still = []
            for so in pending:
                try:
                    ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    still.append(so)
            if len(still) == len(pending):
                break
            pending = still
