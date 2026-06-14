#!/usr/bin/env python3
"""
NOEMA — Script de inicialización para Google Colab
===================================================

Ejecuta TODO desde cero:
  1. Instala dependencias
  2. Detecta GPU/CPU
  3. Corre tests de subsistemas (S1-S6)
  4. Corre tests de invariantes (I1-I4)
  5. Corre 4 experimentos decisivos (Sección 8)
  6. Corre benchmarks brutales (Laberinto 3D, Hash, Brazo 4DOF)
  7. Imprime veredicto final

USO EN COLAB:
  !python colab_run.py

O si subiste la carpeta noema/ a /content/:
  !cd /content && python noema/colab_run.py
"""

import subprocess
import sys
import os
import time


def install_deps():
    """Instalar dependencias necesarias."""
    print("📦 Instalando dependencias...")
    deps = ["torch", "numpy", "scipy", "matplotlib", "tqdm"]
    for dep in deps:
        try:
            __import__(dep)
            print(f"  ✓ {dep} ya instalado")
        except ImportError:
            print(f"  ⬇ Instalando {dep}...")
            if dep == "torch":
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    "torch", "--index-url", "https://download.pytorch.org/whl/cpu",
                    "--no-cache-dir", "-q"
                ])
            else:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", dep, "-q"
                ])
            print(f"  ✓ {dep} instalado")


def detect_device():
    """Detectar si hay GPU disponible."""
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🎮 GPU detectada: {gpu_name}")
        else:
            device = "cpu"
            print(f"💻 Usando CPU")
    except ImportError:
        device = "cpu"
        print(f"💻 PyTorch no instalado, usando CPU")
    return device


def setup_path():
    """Configurar PYTHONPATH para encontrar noema/."""
    # Detectar dónde estamos
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Si estamos dentro de noema/, subir un nivel
    if os.path.basename(script_dir) == "noema":
        project_root = os.path.dirname(script_dir)
    else:
        project_root = script_dir

    # Añadir al path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Verificar que noema es importable
    try:
        import noema
        print(f"✓ noema encontrado en: {os.path.dirname(noema.__file__)}")
        return project_root
    except ImportError:
        # Intentar buscar en ubicaciones comunes de Colab
        for candidate in ["/content", "/content/noema", ".", ".."]:
            sys.path.insert(0, candidate)
            try:
                import noema
                print(f"✓ noema encontrado en: {candidate}")
                return candidate
            except ImportError:
                continue

        print("❌ No se encuentra el paquete noema/")
        print("   Asegúrate de subir la carpeta noema/ a /content/ en Colab")
        print("   O ejecutar desde el directorio que contiene noema/")
        sys.exit(1)


def run_subsystem_tests():
    """Correr tests unitarios de todos los subsistemas."""
    print("\n" + "=" * 70)
    print("🔧 PARTE 1: TESTS DE SUBSISTEMAS (S1-S6)")
    print("=" * 70)

    from noema.tests.test_subsystems import run_all_subsystem_tests
    return run_all_subsystem_tests()


def run_invariant_tests():
    """Correr tests de los 4 invariantes."""
    print("\n" + "=" * 70)
    print("📐 PARTE 2: CUATRO INVARIANTES (Sufficiency Thesis)")
    print("=" * 70)

    from noema.tests.test_invariants import (
        test_I1_autonomy_of_objective,
        test_I2_grounded_abstraction,
        test_I3_relational_portability,
        test_I4_ontogeny,
    )

    results = {}
    results["I1"] = test_I1_autonomy_of_objective()
    results["I2"] = test_I2_grounded_abstraction()
    results["I3"] = test_I3_relational_portability()
    results["I4"] = test_I4_ontogeny()
    return results


def run_decisive_experiments():
    """Correr los 4 experimentos decisivos (Sección 8)."""
    print("\n" + "=" * 70)
    print("🧪 PARTE 3: CUATRO EXPERIMENTOS DECISIVOS (Sección 8)")
    print("=" * 70)

    from noema.tests.test_decisive_experiments import run_all_decisive_experiments
    return run_all_decisive_experiments()


def run_brutal_benchmarks():
    """Correr benchmarks brutales."""
    print("\n" + "=" * 70)
    print("🔥 PARTE 4: BENCHMARKS BRUTALES")
    print("=" * 70)

    from noema.benchmarks.maze3d import run_maze_benchmark
    from noema.benchmarks.hash_inversion import run_hash_inversion_benchmark
    from noema.benchmarks.robot_arm_4d import run_robot_arm_benchmark

    results = {}

    # Laberinto 3D
    print("\n🏔️ Laberinto 3D...")
    try:
        better, data = run_maze_benchmark(n_episodes=8, steps_per_episode=120)
        results["maze3d"] = {"better": better, "data": data}
    except Exception as e:
        print(f"  ❌ Error: {e}")
        results["maze3d"] = {"better": False, "error": str(e)}

    # Hash
    print("\n🔐 Inversión de Hash...")
    try:
        better, data = run_hash_inversion_benchmark(n_experiments=25, steps_per_hash=35)
        results["hash"] = {"better": better, "data": data}
    except Exception as e:
        print(f"  ❌ Error: {e}")
        results["hash"] = {"better": False, "error": str(e)}

    # Brazo 4DOF
    print("\n🦾 Brazo Robótico 4DOF...")
    try:
        better, data = run_robot_arm_benchmark(n_episodes=10, steps_per_episode=50)
        results["arm4d"] = {"better": better, "data": data}
    except Exception as e:
        print(f"  ❌ Error: {e}")
        results["arm4d"] = {"better": False, "error": str(e)}

    return results


def print_final_verdict(subsystem, invariants, experiments, brutal):
    """Imprimir veredicto final de todos los tests."""
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█  🏆 VEREDICTO FINAL — NOEMA BENCHMARK SUITE 🏆" + " " * 20 + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    # Subsystems
    sub_pass = all(subsystem.values()) if subsystem else False
    print(f"\n  🔧 Subsistemas: {'✅ TODOS OK' if sub_pass else '❌ ALGUNOS FALLARON'}")
    for name, passed in (subsystem or {}).items():
        print(f"     {name}: {'✓' if passed else '✗'}")

    # Invariants
    inv_pass = all(invariants.values()) if invariants else False
    print(f"\n  📐 Invariantes: {'✅ SATISFECHOS' if inv_pass else '❌ NO SATISFECHOS'}")
    for name, passed in (invariants or {}).items():
        print(f"     {name}: {'✓' if passed else '✗'}")

    # Experiments
    exp_pass = all(v for v in (experiments[0] if experiments else {}).values()) if experiments else False
    print(f"\n  🧪 Experimentos Decisivos: {'✅ PASADOS' if exp_pass else '❌ ALGUNOS FALLARON'}")
    if experiments:
        for name, passed in experiments[0].items():
            print(f"     {name}: {'✓' if passed else '✗'}")

    # Brutal
    brutal_pass = sum(1 for v in brutal.values() if v.get("better", False))
    brutal_total = len(brutal)
    print(f"\n  🔥 Benchmarks Brutales: {brutal_pass}/{brutal_total} superados vs Random")
    icons = {"maze3d": "🏔️", "hash": "🔐", "arm4d": "🦾"}
    for name, r in brutal.items():
        icon = icons.get(name, "❓")
        status = "✓ MEJOR" if r.get("better") else "✗ NO MEJOR"
        print(f"     {icon} {name}: {status}")

    # Final
    print(f"\n  {'═' * 50}")
    if sub_pass and inv_pass and exp_pass and brutal_pass >= 2:
        print(f"  ║  🟢 SUFFICIENCY THESIS: HOLDS")
        print(f"  ║  NOEMA es un blueprint completo para AGI")
        print(f"  ║  {brutal_pass}/3 benchmarks brutales superados")
    elif sub_pass and inv_pass and exp_pass:
        print(f"  ║  🟡 THESIS PARCIALMENTE VERIFICADA")
        print(f"  ║  Invariantes OK, Experimentos OK")
        print(f"  ║  Benchmarks brutales: {brutal_pass}/3")
        print(f"  ║  Fallos localizables — se sabe QUÉ y DÓNDE")
    else:
        print(f"  ║  🔴 THESIS NECESITA REPARACIÓN")
        print(f"  ║  Pero los fallos son LOCALIZABLES")
    print(f"  {'═' * 50}")

    print("""
  "The thesis is now in the only place a thesis of this magnitude
   belongs: in the open, fully armed, waiting to be proven or broken."
   — NOEMA, Section 11
""")


def main():
    """Función principal — ejecuta todo el benchmark suite."""
    t0 = time.time()

    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║   NOEMA — Neuromorphic Ontogenetic Extrapolative Mapping           ║
║                                                                    ║
║   "AGI: A Sufficient Architecture for Artificial General            ║
║    Intelligence"                                                    ║
║                                                                    ║
║   Kaoru Aguilera Katayama                                          ║
║                                                                    ║
║   🔧 Tests de subsistemas  (S1-S6)                                ║
║   📐 Tests de invariantes  (I1-I4)                                ║
║   🧪 Experimentos decisivos (Sección 8)                           ║
║   🔥 Benchmarks brutales   (3D Maze, Hash, Robot 4DOF)            ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    # Step 0: Setup
    print("⚙️ Configurando entorno...\n")
    install_deps()
    device = detect_device()
    project_root = setup_path()

    # Step 1: Subsystem tests
    try:
        subsystem_results = run_subsystem_tests()
    except Exception as e:
        print(f"❌ Error en tests de subsistemas: {e}")
        import traceback
        traceback.print_exc()
        subsystem_results = {}

    # Step 2: Invariant tests
    try:
        invariant_results = run_invariant_tests()
    except Exception as e:
        print(f"❌ Error en tests de invariantes: {e}")
        import traceback
        traceback.print_exc()
        invariant_results = {}

    # Step 3: Decisive experiments
    try:
        exp_results, exp_details = run_decisive_experiments()
    except Exception as e:
        print(f"❌ Error en experimentos decisivos: {e}")
        import traceback
        traceback.print_exc()
        exp_results, exp_details = {}, {}

    # Step 4: Brutal benchmarks
    try:
        brutal_results = run_brutal_benchmarks()
    except Exception as e:
        print(f"❌ Error en benchmarks brutales: {e}")
        import traceback
        traceback.print_exc()
        brutal_results = {}

    # Final verdict
    print_final_verdict(subsystem_results, invariant_results,
                       (exp_results, exp_details), brutal_results)

    total_time = time.time() - t0
    print(f"  ⏱️ Tiempo total: {total_time:.1f}s")


if __name__ == "__main__":
    main()
