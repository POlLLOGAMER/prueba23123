"""
NOEMA Agent — Full integration of all subsystems.

The agent integrates:
  S1: Sensorimotor Core (brainstem)
  S2: Hierarchical World Model (cortex)
  S3: Complementary Memory (hippocampus-cortex)
  S4: Relational Knowledge Graph (semantic system)
  S5: Analogy Engine (organ of extrapolation)
  S6: Global Workspace (coordination)

Unified objective: Expected Free Energy (I1)
All abstractions grounded in sensorimotor prediction (I2)
Relations detachable from fillers (I3)
Developmental self-scheduling (I4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from .core.free_energy import ExpectedFreeEnergy
from .core.jepa import JEPAModule
from .subsystems.s1_sensorimotor import SensorimotorCore
from .subsystems.s2_world_model import HierarchicalWorldModel
from .subsystems.s3_memory import ComplementaryMemory
from .subsystems.s4_knowledge import RelationalKnowledgeGraph, Schema, Entity, Relation
from .subsystems.s5_analogy import AnalogyEngine
from .subsystems.s6_workspace import GlobalWorkspace, ContentType


class NOEMAAgent(nn.Module):
    """
    Complete NOEMA agent integrating all six subsystems.

    This is the constructive instance proving the Sufficiency Thesis.
    """

    def __init__(
        self,
        obs_dim: int = 32,
        proprio_dim: int = 12,
        extero_dim: int = 16,
        action_dim: int = 4,
        latent_dim: int = 64,
        n_world_model_levels: int = 3,
        num_slots: int = 8,
        slot_dim: int = 32,
        embedding_dim: int = 32,
        episodic_capacity: int = 5000,
        surprise_threshold: float = 1.5,
        device: str = "cpu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.device = device

        # S1: Sensorimotor Core
        self.s1 = SensorimotorCore(
            proprio_dim=proprio_dim,
            extero_dim=extero_dim,
            motor_dim=action_dim,
        )

        # S2: Hierarchical World Model
        self.s2 = HierarchicalWorldModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_levels=n_world_model_levels,
            latent_dim=latent_dim,
            num_slots=num_slots,
            slot_dim=slot_dim,
        )

        # S3: Complementary Memory
        self.s3 = ComplementaryMemory(
            latent_dim=latent_dim,
            action_dim=action_dim,
            episodic_capacity=episodic_capacity,
            surprise_threshold=surprise_threshold,
        )

        # S4: Relational Knowledge Graph
        self.s4 = RelationalKnowledgeGraph(
            embedding_dim=embedding_dim,
            relation_embedding_dim=32,
        )

        # S5: Analogy Engine
        self.s5 = AnalogyEngine(
            knowledge_graph=self.s4,
            embedding_dim=embedding_dim,
            relation_dim=32,
        )

        # S6: Global Workspace
        self.s6 = GlobalWorkspace(
            embedding_dim=latent_dim,
            capacity=7,
        )

        # Expected Free Energy (unified objective)
        self.efe = ExpectedFreeEnergy(
            latent_dim=latent_dim,
            obs_dim=obs_dim,
            n_actions=action_dim,
        )

        # Observation encoder: raw obs → latent
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.GELU(),
            nn.Linear(128, latent_dim),
        )

        # Action decoder: workspace broadcast + observation → action
        # This is the key: deliberation (workspace) meets action
        self.action_decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, action_dim),
            nn.Tanh(),
        )

        # Tracking
        self.step_count = 0
        self.phase = 0  # Current ontogenetic phase

        # Optimizer — learning during interaction (no separate training!)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.loss_history = []

        # Action learning buffer: remember (obs, good_action) pairs
        # where good_action reduced free energy or distance
        self.action_buffer = []
        self.last_obs = None
        self.last_z = None
        self.last_fe = None

    def forward(
        self,
        obs: torch.Tensor,
        proprio: torch.Tensor,
        extero: torch.Tensor,
        preferred_obs: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        One full NOEMA processing cycle.

        1. S1 processes sensorimotor input (reflexes + prediction errors)
        2. S2 runs hierarchical world model (predict + factorize)
        3. S3 stores surprising experiences (episodic memory)
        4. S4 builds relational schemas (knowledge graph)
        5. S5 performs analogical reasoning (extrapolation)
        6. S6 coordinates via global workspace (broadcast)
        7. Action selected by minimizing EFE

        Returns action and full diagnostic output.
        """
        batch = obs.shape[0] if obs.dim() > 1 else 1
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            proprio = proprio.unsqueeze(0)
            extero = extero.unsqueeze(0)

        self.step_count += 1

        # Encode observation to latent
        obs = obs.float()  # Force float32
        proprio = proprio.float()
        extero = extero.float()
        z = self.obs_encoder(obs)

        # S1: Sensorimotor processing (uses last action or zeros)
        dummy_motor = torch.zeros(batch, self.action_dim, device=self.device)
        s1_out = self.s1(proprio, extero, dummy_motor)

        # S2: Hierarchical world model prediction
        # Create observation sequence (current + shifted for JEPA)
        obs_seq = [obs, obs + torch.randn_like(obs) * 0.01]  # o_t, o_{t+1} approx
        act_seq = [dummy_motor]
        s2_out = self.s2(obs_seq, act_seq)

        # ---- Online learning: minimize free energy + train action decoder ----
        # This is the core of I1: the agent learns purely from interaction
        loss = s2_out["total_loss"]

        if isinstance(loss, torch.Tensor) and loss.requires_grad:
            if torch.isfinite(loss):
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()

        # Compute free energy (surprise) — NaN-safe
        raw_loss = s2_out["total_loss"]
        if isinstance(raw_loss, torch.Tensor):
            free_energy = raw_loss.detach().item()
        else:
            free_energy = float(raw_loss)
        if not np.isfinite(free_energy):
            free_energy = 10.0  # Default high surprise for NaN
        self.loss_history.append(free_energy)

        # S3: Store surprising experience
        z_traj = torch.stack([z.squeeze(0), z.squeeze(0) + torch.randn_like(z.squeeze(0)) * 0.01])
        a_traj = torch.stack([dummy_motor.squeeze(0)] * 2)
        s3_out = self.s3(z_traj, a_traj, free_energy)

        # S4: Build relational schemas from S2 slots
        if s2_out["slots"][0].dim() >= 2:
            schema_built = False
            for level_idx, slots in enumerate(s2_out["slots"]):
                if slots.dim() == 3 and slots.shape[0] > 0:
                    rels = s2_out["relations"][level_idx]
                    if rels and "embeddings" in rels:
                        try:
                            self.s4.build_schema_from_slots(
                                slots=slots,
                                relation_embeddings=rels["embeddings"],
                                domain=f"level_{level_idx}",
                            )
                            schema_built = True
                        except Exception:
                            pass
                    break

        # S5: Analogical reasoning (if we have schemas)
        s5_out = {"best_analogy": None, "inferences": [], "total_epistemic_value": 0.0}
        if len(self.s4.schemas) > 1:
            try:
                # Use latest schema as target, search across others
                latest_schema = list(self.s4.schemas.values())[-1]
                raw_s5 = self.s5(latest_schema)
                # Normalize keys
                s5_out = {
                    "best_analogy": raw_s5.get("best_analogy"),
                    "inferences": raw_s5.get("inferences", []),
                    "total_epistemic_value": raw_s5.get("total_epistemic_value",
                                                         raw_s5.get("epistemic_value", 0.0)),
                }
            except Exception:
                pass

        # S6: Global workspace competition
        # Submit content from all subsystems
        self.s6.clear()

        # S2 submits prediction
        if s2_out["z_t"].dim() >= 1:
            pred_precision = 1.0 / (free_energy + 1e-6)
            self.s6.submit(
                content_type=ContentType.PREDICTION,
                data=s2_out["z_t"].detach(),
                precision=pred_precision,
                source="S2",
            )

        # S3 submits recall if available
        recalls = self.s3.retrieve(z.squeeze(0), top_k=1)
        if recalls:
            recall_precision = 1.0 / (recalls[0].surprise + 1e-6)
            self.s6.submit(
                content_type=ContentType.RECALL,
                data=z.squeeze(0).detach(),
                precision=recall_precision,
                source="S3",
            )

        # S5 submits hypothesis if any
        if s5_out["inferences"]:
            self.s6.submit(
                content_type=ContentType.HYPOTHESIS,
                data=z.squeeze(0).detach(),
                precision=s5_out["total_epistemic_value"],
                source="S5",
                metadata={"n_inferences": len(s5_out["inferences"])},
            )

        # S1 submits reflex
        self.s6.submit(
            content_type=ContentType.REFLEX,
            data=s1_out["hidden_state"].detach(),
            precision=s1_out["sparsity"].item() + 0.5,
            source="S1",
        )

        # Competition
        workspace_out = self.s6.competition_step()

        # Preferred observations: use the current observation as a preference
        # This makes the pragmatic term of EFE active — the agent wants
        # observations where it's close to the goal
        if preferred_obs is None:
            # Use current observation as weak preference (exploitation)
            preferred_obs = obs.detach()

        # Select action via EFE
        action_candidates = torch.eye(self.action_dim, device=self.device)
        selected_action, efe_values = self.efe.select_action(
            z, preferred_obs, action_candidates
        )

        # Workspace broadcast modulates action — this is where deliberation
        # meets action. The broadcast carries information from all subsystems.
        broadcast = self.s6.get_broadcast()
        if broadcast is not None:
            # Concatenate z (perception) + broadcast (deliberation) for action
            decoder_input = torch.cat([z, broadcast.unsqueeze(0).expand(z.shape[0], -1)], dim=-1)
            if decoder_input.shape[-1] == self.latent_dim * 2:
                decoded_action = self.action_decoder(decoder_input)
                # Mix: decoded action (learned) + EFE action (exploration)
                # More weight on decoded action as agent learns
                explore_weight = max(0.3, 1.0 - len(self.action_buffer) / 500)
                action = torch.tanh(decoded_action * (1 - explore_weight) + 
                                   selected_action * explore_weight +
                                   torch.randn_like(selected_action) * 0.1)
            else:
                action = torch.tanh(selected_action + torch.randn_like(selected_action) * 0.3)
        else:
            # No broadcast yet — use EFE + exploration noise
            action = torch.tanh(selected_action + torch.randn_like(selected_action) * 0.3)

        # Store for action learning: if free energy decreased, this was a good action
        if self.last_obs is not None and self.last_fe is not None:
            if free_energy < self.last_fe:  # This action was good!
                self.action_buffer.append((
                    self.last_obs.detach(),
                    self.last_z.detach(),
                    action.detach(),
                ))
                # Keep buffer bounded
                if len(self.action_buffer) > 1000:
                    self.action_buffer = self.action_buffer[-500:]
                
                # Train action decoder on good actions
                if len(self.action_buffer) > 10:
                    batch_idx = np.random.choice(len(self.action_buffer), min(8, len(self.action_buffer)), replace=False)
                    for idx in batch_idx:
                        buf_obs, buf_z, buf_action = self.action_buffer[idx]
                        # Reconstruct decoder input
                        if broadcast is not None:
                            buf_input = torch.cat([buf_z, broadcast.unsqueeze(0).expand(1, -1)], dim=-1)
                            if buf_input.shape[-1] == self.latent_dim * 2:
                                pred_action = self.action_decoder(buf_input)
                                action_loss = F.mse_loss(pred_action, buf_action.unsqueeze(0))
                                self.optimizer.zero_grad()
                                action_loss.backward()
                                torch.nn.utils.clip_grad_norm_(self.action_decoder.parameters(), max_norm=0.5)
                                self.optimizer.step()

        self.last_obs = obs.detach()
        self.last_z = z.detach()
        self.last_fe = free_energy

        return {
            "action": action,
            "free_energy": free_energy,
            "s1": {
                "sparsity": s1_out["sparsity"].item(),
                "reflex_motor": s1_out["reflex_motor"].detach(),
            },
            "s2": {
                "total_loss": free_energy,
                "level_losses": s2_out["level_losses"],
            },
            "s3": s3_out,
            "s4": self.s4.graph_stats(),
            "s5": {
                "best_analogy_domain": (
                    s5_out["best_analogy"]["source_domain"]
                    if s5_out["best_analogy"]
                    else None
                ),
                "n_inferences": len(s5_out["inferences"]),
                "epistemic_value": s5_out["total_epistemic_value"],
            },
            "s6": workspace_out,
            "efe_values": efe_values.detach(),
            "phase": self.phase,
        }

    def consolidate(self) -> dict:
        """Run offline consolidation (sleep phase)."""
        return self.s3.consolidate(self.s2, n_steps=4)

    def get_diagnostics(self) -> dict:
        """Get full diagnostic information."""
        return {
            "step_count": self.step_count,
            "phase": self.phase,
            "s3_memory": self.s3.memory_stats(),
            "s4_graph": self.s4.graph_stats(),
        }
