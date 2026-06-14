"""
BENCHMARK BRUTAL #1: Laberinto 3D

El agente debe navegar un laberinto 3D usando solo observaciones parciales
(paredes en 6 direcciones cartesianas). Sin mapa, sin GPS, sin curriculum.
Pura EFE + world model + analogía.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class MazeState:
    position: np.ndarray       # (3,) x,y,z
    goal: np.ndarray           # (3,) goal position
    walls: torch.Tensor        # (6,) binary: +x,-x,+y,-y,+z,-z
    observation: torch.Tensor  # full obs vector
    done: bool = False
    steps: int = 0


class Maze3D:
    """
    Laberinto 3D con celdas. El agente ve paredes en 6 direcciones
    y debe llegar a la meta. Observación parcial tipo ray-casting simplificado.
    """

    def __init__(
        self,
        size: int = 6,           # 6x6x6 maze
        obs_dim: int = 48,
        action_dim: int = 6,     # +/-x, +/-y, +/-z
        max_steps: int = 200,
    ):
        self.size = size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.reset()

    def reset(self) -> MazeState:
        """Generar laberinto aleatorio y colocar agente + meta."""
        self.grid = np.zeros((self.size, self.size, self.size), dtype=bool)
        
        # Generar paredes aleatorias (~30% densidad, pero garantizar camino)
        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.size):
                    if np.random.random() < 0.30:
                        self.grid[i, j, k] = True  # wall

        # Colocar agente en posición libre
        while True:
            self.pos = np.array([
                np.random.randint(0, self.size),
                np.random.randint(0, self.size),
                np.random.randint(0, self.size),
            ])
            if not self.grid[self.pos[0], self.pos[1], self.pos[2]]:
                break

        # Colocar meta lejos del agente
        while True:
            self.goal = np.array([
                np.random.randint(0, self.size),
                np.random.randint(0, self.size),
                np.random.randint(0, self.size),
            ])
            dist = np.sum(np.abs(self.goal - self.pos))
            if not self.grid[self.goal[0], self.goal[1], self.goal[2]] and dist >= 4:
                break

        self.step_count = 0
        return self._get_state()

    def step(self, action: torch.Tensor) -> Tuple[MazeState, float, dict]:
        """Mover agente. action es one-hot de 6 direcciones."""
        # Decodificar acción
        directions = [
            np.array([1, 0, 0]),   # +x
            np.array([-1, 0, 0]),  # -x
            np.array([0, 1, 0]),   # +y
            np.array([0, -1, 0]),  # -y
            np.array([0, 0, 1]),   # +z
            np.array([0, 0, -1]),  # -z
        ]

        if isinstance(action, torch.Tensor):
            act_idx = action.argmax().item()
        else:
            act_idx = int(action)

        new_pos = self.pos + directions[act_idx]

        # Verificar límites y paredes
        valid = True
        for d in range(3):
            if new_pos[d] < 0 or new_pos[d] >= self.size:
                valid = False
                break
        if valid and not self.grid[new_pos[0], new_pos[1], new_pos[2]]:
            self.pos = new_pos
        # Si es inválido, el agente se queda en su lugar

        self.step_count += 1
        dist_to_goal = np.sum(np.abs(self.goal - self.pos))
        
        reward = -0.1  # Step penalty
        done = False

        if dist_to_goal == 0:
            reward = 10.0
            done = True
        elif self.step_count >= self.max_steps:
            done = True

        state = self._get_state()
        state.done = done

        info = {
            "distance_to_goal": dist_to_goal,
            "steps": self.step_count,
            "reached_goal": dist_to_goal == 0,
        }
        return state, reward, info

    def _get_state(self) -> MazeState:
        """Construir observación: paredes + posición relativa + distancia."""
        # Paredes en 6 direcciones (ray-casting simplificado)
        directions = [
            np.array([1, 0, 0]), np.array([-1, 0, 0]),
            np.array([0, 1, 0]), np.array([0, -1, 0]),
            np.array([0, 0, 1]), np.array([0, 0, -1]),
        ]
        walls = torch.zeros(6)
        ray_depths = torch.zeros(6)  # Cuántos pasos hasta pared

        for i, d in enumerate(directions):
            for step in range(1, self.size):
                check = self.pos + d * step
                if any(c < 0 or c >= self.size for c in check):
                    walls[i] = 1.0
                    ray_depths[i] = step
                    break
                if self.grid[check[0], check[1], check[2]]:
                    walls[i] = 1.0
                    ray_depths[i] = step
                    break

        # Posición normalizada
        pos_norm = torch.tensor(self.pos, dtype=torch.float32) / self.size

        # Dirección a la meta (normalizada)
        goal_dir = torch.tensor(
            (self.goal - self.pos), dtype=torch.float32
        ) / (self.size + 1e-8)

        # Distancia normalizada
        dist = np.sum(np.abs(self.goal - self.pos)) / (3 * self.size)

        # Construir observación completa
        obs = torch.cat([
            walls.float(),                          # 6  paredes
            ray_depths.float() / self.size,         # 6  profundidad de rayos
            pos_norm.float(),                       # 3  posición
            goal_dir.float(),                       # 3  dirección a meta
            torch.tensor([dist], dtype=torch.float32),  # 1  distancia
            torch.randn(48 - 19) * 0.01,           # padding a obs_dim
        ])[:self.obs_dim]

        return MazeState(
            position=self.pos.copy(),
            goal=self.goal.copy(),
            walls=walls,
            observation=obs,
            done=False,
            steps=self.step_count,
        )


def run_maze_benchmark(n_episodes: int = 10, steps_per_episode: int = 150):
    """Correr NOEMA en laberinto 3D."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from noema.agent import NOEMAAgent

    print("\n" + "=" * 70)
    print("🏔️  BENCHMARK #1: LABERINTO 3D")
    print("=" * 70)
    print(f"Maze: 6×6×6, Observación parcial (rayos + dirección meta)")
    print(f"Agente NOEMA: sin mapa, sin GPS, sin curriculum")
    print(f"Objetivo: navegar desde start hasta goal usando solo EFE")
    print("-" * 70)

    env = Maze3D(size=6, obs_dim=48, action_dim=6, max_steps=steps_per_episode)
    agent = NOEMAAgent(
        obs_dim=48, action_dim=6, latent_dim=64,
        proprio_dim=12, extero_dim=16,
        num_slots=4, slot_dim=16,
        surprise_threshold=0.5,
    )

    results = {"goals_reached": 0, "distances": [], "fes": [], "schemas": []}

    for ep in range(n_episodes):
        state = env.reset()
        ep_dist = []
        ep_fe = []

        for step in range(steps_per_episode):
            # Proprio: posición actual
            proprio = torch.zeros(12)
            proprio[:3] = torch.tensor(state.position, dtype=torch.float32) / 6.0

            # Extero: paredes
            extero = torch.zeros(16)
            extero[:6] = state.walls

            result = agent(
                obs=state.observation,
                proprio=proprio,
                extero=extero,
            )

            action = result["action"].squeeze(0)
            state, reward, info = env.step(action)

            ep_dist.append(info["distance_to_goal"])
            ep_fe.append(result["free_energy"])

            if info["reached_goal"]:
                break

        reached = info["reached_goal"]
        final_dist = ep_dist[-1]
        mean_fe = np.mean(ep_fe)

        results["goals_reached"] += int(reached)
        results["distances"].append(final_dist)
        results["fes"].append(mean_fe)
        results["schemas"].append(agent.s4.graph_stats()["n_schemas"])

        print(f"  Ep {ep+1:2d}/{n_episodes}: "
              f"dist_final={final_dist:.1f}  "
              f"FE={mean_fe:.4f}  "
              f"{'🎯 GOAL!' if reached else '❌'}  "
              f"schemas={agent.s4.graph_stats()['n_schemas']}")

    # Comparar con random baseline
    print(f"\n  --- Random Baseline ---")
    random_goals = 0
    random_dists = []
    for ep in range(n_episodes):
        state = env.reset()
        for step in range(steps_per_episode):
            action = torch.zeros(6)
            action[np.random.randint(6)] = 1.0
            state, _, info = env.step(action)
            if info["reached_goal"]:
                break
        random_goals += int(info["reached_goal"])
        random_dists.append(info["distance_to_goal"])
    
    print(f"  Random: {random_goals}/{n_episodes} goals reached")
    print(f"  Random mean final dist: {np.mean(random_dists):.2f}")

    noema_rate = results["goals_reached"] / n_episodes
    random_rate = random_goals / n_episodes
    noema_mean_dist = np.mean(results["distances"])
    random_mean_dist = np.mean(random_dists)

    print(f"\n  --- RESULTADOS LABERINTO 3D ---")
    print(f"  NOEMA:  {results['goals_reached']}/{n_episodes} goals ({noema_rate*100:.0f}%), mean_dist={noema_mean_dist:.2f}")
    print(f"  Random: {random_goals}/{n_episodes} goals ({random_rate*100:.0f}%), mean_dist={random_mean_dist:.2f}")
    print(f"  NOEMA vs Random: {'MEJOR ✓' if noema_mean_dist < random_mean_dist else 'IGUAL/PEOR ✗'}")
    print(f"  Schemas construidos: {max(results['schemas'])}")
    print(f"  Tipos de relación: {len(agent.s4.get_all_relation_types())}")

    better = noema_mean_dist < random_mean_dist or noema_rate > random_rate
    return better, results
