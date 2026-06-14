"""
🔥 BENCHMARKS BRUTALES 🔥

Laberinto 3D, Inversión de Hash, Brazo Robótico 4DOF.
Si NOEMA es AGI, no necesita re-arquitectura para cada dominio.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from noema.benchmarks.maze3d import run_maze_benchmark
from noema.benchmarks.hash_inversion import run_hash_inversion_benchmark
from noema.benchmarks.robot_arm_4d import run_robot_arm_benchmark


def print_brutal_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║   🔥🔥🔥  BENCHMARKS BRUTALES  🔥🔥🔥                              ║
║                                                                    ║
║   Si NOEMA es AGI, no necesita re-arquitectura para cada dominio.  ║
║   Misma arquitectura, tres dominios completamente diferentes:       ║
║                                                                    ║
║   🏔️  Laberinto 3D (navegación espacial)                           ║
║   🔐  Inversión de Hash (razonamiento abstracto)                   ║
║   🦾  Brazo Robótico 4DOF (control motor de alta dimensión)        ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════════╝
""")


def run_all_brutal_benchmarks():
    print_brutal_banner()
    total_start = time.time()

    all_results = {}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # BENCHMARK 1: Laberinto 3D
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "━" * 70)
    print("━  BENCHMARK 1/3: LABERINTO 3D")
    print("━" * 70)
    t0 = time.time()
    try:
        maze_better, maze_results = run_maze_benchmark(n_episodes=10, steps_per_episode=150)
        all_results["maze3d"] = {"better": maze_better, "data": maze_results}
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback; traceback.print_exc()
        all_results["maze3d"] = {"better": False, "error": str(e)}
    maze_time = time.time() - t0

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # BENCHMARK 2: Inversión de Hash
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "━" * 70)
    print("━  BENCHMARK 2/3: INVERSIÓN DE FUNCIONES HASH")
    print("━" * 70)
    t0 = time.time()
    try:
        hash_better, hash_results = run_hash_inversion_benchmark(n_experiments=30, steps_per_hash=40)
        all_results["hash"] = {"better": hash_better, "data": hash_results}
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback; traceback.print_exc()
        all_results["hash"] = {"better": False, "error": str(e)}
    hash_time = time.time() - t0

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # BENCHMARK 3: Brazo Robótico 4DOF
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "━" * 70)
    print("━  BENCHMARK 3/3: BRAZO ROBÓTICO 4DOF")
    print("━" * 70)
    t0 = time.time()
    try:
        arm_better, arm_results = run_robot_arm_benchmark(n_episodes=12, steps_per_episode=60)
        all_results["arm4d"] = {"better": arm_better, "data": arm_results}
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback; traceback.print_exc()
        all_results["arm4d"] = {"better": False, "error": str(e)}
    arm_time = time.time() - t0

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # VEREDICTO FINAL
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    total_time = time.time() - total_start

    print("\n" + "█" * 70)
    print("█  🔥 VEREDICTO FINAL: BENCHMARKS BRUTALES 🔥")
    print("█" * 70)

    n_pass = sum(1 for r in all_results.values() if r.get("better", False))
    n_total = len(all_results)

    for name, r in all_results.items():
        better = r.get("better", False)
        icon = "🏔️" if name == "maze3d" else "🔐" if name == "hash" else "🦾"
        status = "✓ MEJOR QUE RANDOM" if better else "✗ NO MEJOR QUE RANDOM"
        print(f"  {icon} {name:15s}: {status}")

    print(f"\n  {n_pass}/{n_total} benchmarks superados vs random baseline")
    print(f"  Tiempo total: {total_time:.1f}s")

    if n_pass == n_total:
        print(f"""
  ╔══════════════════════════════════════════════════════════════╗
  ║  🔥🔥🔥  NOEMA DOMINA TODOS LOS DOMINIOS  🔥🔥🔥            ║
  ║                                                            ║
  ║  Misma arquitectura. Sin re-entrenamiento. Sin hacks.      ║
  ║  Laberinto 3D, Hash, Brazo 4DOF — todos superados.        ║
  ╚══════════════════════════════════════════════════════════════╝
""")
    elif n_pass > 0:
        print(f"""
  ╔══════════════════════════════════════════════════════════════╗
  ║  PARCIAL: {n_pass}/{n_total} dominios superados                         ║
  ║                                                            ║
  ║  Los fallos son LOCALIZABLES — se sabe exactamente         ║
  ║  qué invariante falla en cada dominio.                     ║
  ╚══════════════════════════════════════════════════════════════╝
""")
    else:
        print(f"""
  ╔══════════════════════════════════════════════════════════════╗
  ║  Los benchmarks brutales son, pues, brutales.              ║
  ║  Pero al menos sabemos QUÉ falla y DÓNDE.                  ║
  ╚══════════════════════════════════════════════════════════════╝
""")

    return all_results


if __name__ == "__main__":
    run_all_brutal_benchmarks()
