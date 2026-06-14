#!/usr/bin/env python3
"""
🚨 NOEMA AUTÓNOMO — SIN TRAMPAS — Para Google Colab

USO:
  !cd /content && python noema/colab_autonomous.py [N_STEPS]

Ejemplos:
  !cd /content && python noema/colab_autonomous.py        # 100 pasos
  !cd /content && python noema/colab_autonomous.py 500     # 500 pasos
"""

import subprocess
import sys
import os


def main():
    print("📦 Verificando dependencias...")
    try:
        import torch
        import numpy
        print("  ✓ OK")
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install",
            "torch", "--index-url", "https://download.pytorch.org/whl/cpu",
            "--no-cache-dir", "-q"])
        subprocess.check_call([sys.executable, "-m", "pip", "install",
            "numpy", "scipy", "-q"])
        print("  ✓ Instalado")

    # Setup path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for p in [os.path.dirname(script_dir), "/content", "/content/noema", "."]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # Import and run
    from noema.autonomous import run_autonomous

    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    wdir = "/content" if os.path.exists("/content") else os.getcwd()

    run_autonomous(n_steps=n_steps, working_dir=wdir)


if __name__ == "__main__":
    main()
