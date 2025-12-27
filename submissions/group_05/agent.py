# submissions/group_05/agent.py
# -*- coding: utf-8 -*-
"""
Pacman chase and ghost evade agents with belief tracking.

Pacman:
- Momentum-based BFS to intercept ghost.
- Movement rules:
  + Turn (different direction): 1 step only
  + Straight (same direction): can do 1 or 2 steps
  + Never go over pacman_speed limit

Ghost (Hybrid RL + fallback heuristic):
- If ghost_policy.pth exists and torch is available:
  + Use small MLP policy for fast inference (legal-move masking + SOFT guardrails)
- Otherwise:
  + Use heuristic Safe Evade (distance + LOS + dead-end avoidance + exploration)

IMPORTANT FIXES vs your version:
- Add last_action + last2_action one-hot into features (reduces ABAB).
- Soft guardrails (penalty) instead of hard banning with -1e6 everywhere.
- Heuristic also tracks last_move and heavily penalizes reverse/backtrack.
"""

import os
import sys
from pathlib import Path
from collections import deque
import random
import numpy as np

# ---------------------------------------------------------
# Imports (robust)
# ---------------------------------------------------------
try:
    from agent_interface import PacmanAgent as BasePacmanAgent
    from agent_interface import GhostAgent as BaseGhostAgent
    from environment import Move
except Exception:
    src_path = Path(__file__).parent.parent.parent / "src"
    sys.path.insert(0, str(src_path))
    from agent_interface import PacmanAgent as BasePacmanAgent
    from agent_interface import GhostAgent as BaseGhostAgent
    from environment import Move

# Optional torch (for RL inference only)
try:
    import torch
    import torch.nn as nn
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
except Exception:
    torch = None
    nn = None


ALL_MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
DIRS = [
    (Move.UP, (-1, 0)),
    (Move.DOWN, (1, 0)),
    (Move.LEFT, (0, -1)),
    (Move.RIGHT, (0, 1)),
]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

OPPOSITE = {
    Move.UP: Move.DOWN,
    Move.DOWN: Move.UP,
    Move.LEFT: Move.RIGHT,
    Move.RIGHT: Move.LEFT,
    Move.STAY: Move.STAY,
}

# RL action convention
# 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: STAY
IDX_TO_MOVE = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]
MOVE_TO_IDX = {m: i for i, m in enumerate(IDX_TO_MOVE)}


def manhattan(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# =========================================================
# Belief tracker for enemy position estimation
# =========================================================
class BeliefTracker:
    def __init__(self, decay: float = 0.95, visible_decay: float = 0.05, seed=None):
        self.decay = float(decay)
        self.visible_decay = float(visible_decay)
        self.belief = None
        self.rng = random.Random(seed)

    def ensure(self, shape):
        if self.belief is None or self.belief.shape != shape:
            self.belief = np.zeros(shape, dtype=np.float32)

    def reset_to(self, pos):
        self.belief.fill(0.0)
        self.belief[pos[0], pos[1]] = 1.0

    def init_uniform(self, possible_mask: np.ndarray):
        self.belief.fill(0.0)
        cnt = int(np.sum(possible_mask))
        if cnt <= 0:
            self.belief[:] = 1.0 / self.belief.size
        else:
            self.belief[possible_mask] = 1.0 / cnt

    def update(self, obs_map: np.ndarray, memory_map: np.ndarray, enemy_pos):
        self.ensure(memory_map.shape)
        possible = (memory_map != 1)

        if enemy_pos is not None:
            self.reset_to(enemy_pos)
            return

        if float(self.belief.sum()) <= 1e-9:
            self.init_uniform(possible)
        else:
            self._propagate(possible)

        visible = (obs_map != -1)
        self.belief[visible] *= self.visible_decay
        self.belief[memory_map == 1] = 0.0

        s = float(self.belief.sum())
        if s <= 1e-9:
            self.init_uniform(possible)
        else:
            self.belief /= s

    def _propagate(self, possible_mask: np.ndarray):
        H, W = possible_mask.shape
        newb = np.zeros_like(self.belief, dtype=np.float32)

        rows, cols = np.where(self.belief > 1e-7)
        for r, c in zip(rows, cols):
            p = float(self.belief[r, c])
            if p <= 0 or not possible_mask[r, c]:
                continue

            opts = [(r, c)]  # stay allowed
            for _, (dr, dc) in DIRS:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and possible_mask[rr, cc]:
                    opts.append((rr, cc))

            share = (p * self.decay) / len(opts)
            for rr, cc in opts:
                newb[rr, cc] += share

        s = float(newb.sum())
        if s > 1e-9:
            newb /= s
        self.belief = newb

    def best_guess(self, last_known=None):
        if self.belief is None:
            return None
        mx = float(self.belief.max())
        if mx <= 0:
            return None
        coords = np.argwhere(self.belief >= 0.90 * mx)
        if coords.size == 0:
            return None
        cands = [tuple(x) for x in coords]
        if last_known is not None:
            cands.sort(key=lambda p: manhattan(p, last_known))
            cands = cands[:8]
        return self.rng.choice(cands)

    def get_map(self):
        return None if self.belief is None else self.belief.copy()


# =========================================================
# PACMAN (Seeker) - keep your existing logic (unchanged)
# =========================================================
class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self.capture_threshold = int(kwargs.get("capture_distance", 2))
        self.capture_ring = max(0, self.capture_threshold - 1)

        self.rng = random.Random(kwargs.get("seed", None))

        self.memory_map = None
        self.visit = None

        self.bt = BeliefTracker(decay=0.95, visible_decay=0.05, seed=kwargs.get("seed", None))

        self.last_move = Move.STAY
        self.prev_positions = deque(maxlen=6)

        self.last_seen_enemy = None
        self.last_seen_step = None

        self.name = "Pacman (Momentum BFS + Capture-ring Intercept)"

    def _ensure(self, obs: np.ndarray):
        if self.memory_map is None or self.memory_map.shape != obs.shape:
            self.memory_map = np.full(obs.shape, -1, dtype=np.int8)
            self.visit = np.zeros(obs.shape, dtype=np.int16)

    def _update_memory(self, obs: np.ndarray):
        self._ensure(obs)
        visible = (obs != -1)
        self.memory_map[visible] = obs[visible]

    def _in_bounds(self, r, c):
        H, W = self.memory_map.shape
        return 0 <= r < H and 0 <= c < W

    def _walkable_known(self, r, c) -> bool:
        return self._in_bounds(r, c) and int(self.memory_map[r, c]) == 0

    def _walkable_possible(self, r, c) -> bool:
        return self._in_bounds(r, c) and int(self.memory_map[r, c]) != 1

    def _frontiers(self):
        if self.memory_map is None:
            return []
        H, W = self.memory_map.shape
        fr = []
        zeros = np.where(self.memory_map == 0)
        for r, c in zip(zeros[0], zeros[1]):
            for _, (dr, dc) in DIRS:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and int(self.memory_map[rr, cc]) == -1:
                    fr.append((int(r), int(c)))
                    break
        return fr

    def _unknown_gain(self, pos, radius=2):
        r, c = pos
        H, W = self.memory_map.shape
        r0, r1 = max(0, r - radius), min(H, r + radius + 1)
        c0, c1 = max(0, c - radius), min(W, c + radius + 1)
        return int(np.sum(self.memory_map[r0:r1, c0:c1] == -1))

    def _allowed_steps(self, mv: Move, last_mv: Move):
        if mv == Move.STAY:
            return [1]
        if mv != last_mv:
            return [1]
        if self.pacman_speed >= 2:
            return [2, 1]
        return [1]

    def _apply_action(self, pos, mv: Move, steps: int, strict_known: bool):
        if mv == Move.STAY:
            return pos, 0
        dr, dc = mv.value
        r, c = pos
        moved = 0
        for _ in range(steps):
            rr, cc = r + dr, c + dc
            ok = self._walkable_known(rr, cc) if strict_known else self._walkable_possible(rr, cc)
            if not ok:
                break
            r, c = rr, cc
            moved += 1
        return (int(r), int(c)), moved

    def _capture_cells(self, ghost_pos):
        if ghost_pos is None:
            return []
        r, c = ghost_pos
        cells = [(r, c)]
        if self.capture_ring >= 1:
            for _, (dr, dc) in DIRS:
                cells.append((r + dr, c + dc))
        out = []
        for rr, cc in cells:
            if self._in_bounds(rr, cc):
                out.append((int(rr), int(cc)))
        return out

    def _estimate_turns_single(self, start, last_mv, goal, max_turns=18):
        if start == goal:
            return 0
        q = deque([(start, last_mv)])
        dist = {(start, last_mv): 0}
        while q:
            pos, lm = q.popleft()
            d = dist[(pos, lm)]
            if d >= max_turns:
                continue
            for mv in ALL_MOVES:
                for st in self._allowed_steps(mv, lm):
                    nxt, moved = self._apply_action(pos, mv, st, strict_known=True)
                    if moved <= 0:
                        continue
                    ns = (nxt, mv)
                    if ns in dist:
                        continue
                    nd = d + 1
                    if nxt == goal:
                        return nd
                    dist[ns] = nd
                    q.append(ns)
        return None

    def _best_first_action_to_targets(
        self,
        start,
        start_last_move,
        target_set: set,
        strict_known: bool,
        max_turns=14,
        allow_one_unknown_finish=False
    ):
        if start in target_set:
            return (Move.STAY, 1)

        q = deque([(start, start_last_move)])
        dist = {(start, start_last_move): 0}
        first_action = {(start, start_last_move): None}

        best_act = None
        best_turns = None
        best_tie = -1e18

        prev_cell = self.prev_positions[-2] if len(self.prev_positions) >= 2 else None

        def tie_value(curr_pos, curr_last_mv, mv, st_requested, moved_pos, moved_steps):
            val = 0.0
            if prev_cell is not None and moved_pos == prev_cell:
                val -= 120.0
            if mv == OPPOSITE.get(curr_last_mv, Move.STAY):
                val -= 60.0
            val -= 2.0 * float(self.visit[moved_pos[0], moved_pos[1]])

            if moved_steps == 2:
                near_target = any(manhattan(moved_pos, t) <= 2 for t in list(target_set)[:6])
                val += 70.0 if not near_target else 10.0
            return val

        while q:
            pos, last_mv = q.popleft()
            turns = dist[(pos, last_mv)]
            if best_turns is not None and turns > best_turns:
                continue
            if turns >= max_turns:
                continue

            for mv in ALL_MOVES:
                for st in self._allowed_steps(mv, last_mv):
                    nxt_pos, moved = self._apply_action(pos, mv, st, strict_known=strict_known)
                    if moved <= 0:
                        continue

                    ns = (nxt_pos, mv)
                    nd = turns + 1

                    fa = first_action[(pos, last_mv)]
                    if fa is None:
                        fa = (mv, moved)

                    if ns not in dist:
                        dist[ns] = nd
                        first_action[ns] = fa
                        q.append(ns)
                    elif nd < dist[ns]:
                        dist[ns] = nd
                        first_action[ns] = fa

                    if nxt_pos in target_set:
                        tv = tie_value(pos, last_mv, mv, st, nxt_pos, moved)
                        if best_turns is None or nd < best_turns or (nd == best_turns and tv > best_tie):
                            best_turns = nd
                            best_tie = tv
                            best_act = fa

            if allow_one_unknown_finish and strict_known:
                for mv in ALL_MOVES:
                    nxt_pos, moved = self._apply_action(pos, mv, 1, strict_known=False)
                    if moved <= 0:
                        continue
                    ns = (nxt_pos, mv)
                    nd = turns + 1
                    fa = first_action[(pos, last_mv)]
                    if fa is None:
                        fa = (mv, moved)
                    if ns not in dist:
                        dist[ns] = nd
                        first_action[ns] = fa
                        q.append(ns)
                    elif nd < dist[ns]:
                        dist[ns] = nd
                        first_action[ns] = fa

        return best_act

    def _axis_escape_predictions(self, ghost_pos, my_pos, steps_ahead=(2, 3, 4)):
        if ghost_pos is None:
            return []
        gr, gc = ghost_pos
        pr, pc = my_pos
        dr = gr - pr
        dc = gc - pc

        preds = []
        if abs(dr) >= abs(dc):
            step_r = int(np.sign(dr)) if dr != 0 else 0
            step_c = 0
        else:
            step_r = 0
            step_c = int(np.sign(dc)) if dc != 0 else 0

        if step_r == 0 and step_c == 0:
            return preds

        for k in steps_ahead:
            rr, cc = gr + step_r * k, gc + step_c * k
            if self._in_bounds(rr, cc):
                preds.append((int(rr), int(cc)))
        return preds

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int):
        self._update_memory(map_state)
        self.visit[my_position[0], my_position[1]] += 1
        self.prev_positions.append(my_position)

        if enemy_position is not None:
            self.last_seen_enemy = enemy_position
            self.last_seen_step = step_number

        self.bt.update(map_state, self.memory_map, enemy_position)

        if enemy_position is not None and manhattan(my_position, enemy_position) <= self.capture_ring:
            return (Move.STAY, 1)

        ghost_est = enemy_position if enemy_position is not None else self.bt.best_guess(self.last_seen_enemy)

        if ghost_est is not None:
            raw_targets = self._capture_cells(ghost_est)
            target_cells = {t for t in raw_targets if self._walkable_possible(t[0], t[1])}

            dist_to_est = manhattan(my_position, ghost_est)
            if enemy_position is not None and dist_to_est <= 8:
                for p in self._axis_escape_predictions(ghost_est, my_position):
                    if self._walkable_possible(p[0], p[1]):
                        target_cells.add(p)

            allow_unknown = (dist_to_est <= 4)

            act = self._best_first_action_to_targets(
                my_position,
                self.last_move,
                target_cells,
                strict_known=True,
                max_turns=16,
                allow_one_unknown_finish=allow_unknown
            )
            if act:
                mv, steps = act
                if mv != self.last_move:
                    steps = 1
                else:
                    steps = min(steps, 2, self.pacman_speed)
                steps = min(steps, self.pacman_speed)

                _, moved = self._apply_action(my_position, mv, steps, strict_known=True)
                if moved > 0:
                    self.last_move = mv
                    return (mv, moved)

        frontiers = self._frontiers()
        if frontiers:
            self.rng.shuffle(frontiers)
            sample = frontiers[:30]

            best_f = None
            best_val = -1e18
            bm = self.bt.get_map()

            for f in sample:
                turns = self._estimate_turns_single(my_position, self.last_move, f, max_turns=18)
                if turns is None:
                    continue

                gain = self._unknown_gain(f, radius=2)
                val = 10.0 * gain - 4.0 * turns - 3.0 * float(self.visit[f[0], f[1]])

                if bm is not None:
                    val += 40.0 * float(bm[f[0], f[1]])

                if len(self.prev_positions) >= 2 and f == self.prev_positions[-2]:
                    val -= 25.0

                if val > best_val:
                    best_val = val
                    best_f = f

            if best_f is not None:
                act = self._best_first_action_to_targets(
                    my_position, self.last_move, {best_f},
                    strict_known=True, max_turns=20
                )
                if act:
                    mv, steps = act
                    if mv != self.last_move:
                        steps = 1
                    else:
                        steps = min(steps, 2, self.pacman_speed)
                    steps = min(steps, self.pacman_speed)

                    _, moved = self._apply_action(my_position, mv, steps, strict_known=True)
                    if moved > 0:
                        self.last_move = mv
                        return (mv, moved)

        prev_cell = self.prev_positions[-2] if len(self.prev_positions) >= 2 else None
        candidates = []
        for mv in ALL_MOVES:
            for st in self._allowed_steps(mv, self.last_move):
                nxt, moved = self._apply_action(my_position, mv, st, strict_known=True)
                if moved <= 0:
                    continue
                score = 0.0
                if prev_cell is not None and nxt == prev_cell:
                    score -= 120.0
                if mv == OPPOSITE.get(self.last_move, Move.STAY):
                    score -= 40.0
                score -= 2.0 * float(self.visit[nxt[0], nxt[1]])
                if ghost_est is not None:
                    score -= 1.0 * manhattan(nxt, ghost_est)
                if moved == 2 and ghost_est is not None and manhattan(my_position, ghost_est) <= 3:
                    score -= 30.0
                candidates.append((score, mv, moved))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            _, mv, moved = candidates[0]
            self.last_move = mv
            return (mv, moved)

        return (Move.STAY, 1)


# =========================================================
# Ghost RL (inference) + fallback heuristic
# =========================================================
def _los_flag(memory_map: np.ndarray, a: tuple, b: tuple, radius=5) -> float:
    if b is None:
        return 0.0
    ar, ac = a
    br, bc = b
    if ar != br and ac != bc:
        return 0.0
    dist = abs(ar - br) + abs(ac - bc)
    if dist > radius:
        return 0.0
    dr = 0 if ar == br else (1 if br > ar else -1)
    dc = 0 if ac == bc else (1 if bc > ac else -1)
    r, c = ar, ac
    for _ in range(dist):
        r += dr
        c += dc
        if not (0 <= r < memory_map.shape[0] and 0 <= c < memory_map.shape[1]):
            return 0.0
        if int(memory_map[r, c]) == 1:
            return 0.0
    return 1.0


def _one_hot(idx: int, n: int) -> list:
    v = [0.0] * n
    if 0 <= idx < n:
        v[idx] = 1.0
    return v


def extract_ghost_features_for_infer(
    obs_ghost: np.ndarray,
    memory_map: np.ndarray,
    visit: np.ndarray,
    prev_positions: deque,
    my_pos: tuple,
    pac_est,
    enemy_visible: bool,
    last_action_idx: int,
    last2_action_idx: int,
) -> np.ndarray:
    """
    MUST match train_ghost_dqn.py extract_ghost_features().

    Vector:
      - center cell (obs)
      - 4 rays * 5 (obs)
      - 4 neighbor walkable (memory)
      - enemy_visible
      - dx, dy, dist (norm)
      - los_flag (from memory)
      - degree_norm (from memory)
      - visit_norm (cell)
      - backtrack flag (pos == prev2)
      - unknown flag (current cell unknown in memory)
      - last_action one-hot (5)
      - last2_action one-hot (5)
    """
    r, c = my_pos
    H, W = obs_ghost.shape

    feats = [float(obs_ghost[r, c])]

    # rays
    for dr, dc in DIR4:
        rr, cc = r, c
        for _ in range(5):
            rr += dr
            cc += dc
            if 0 <= rr < H and 0 <= cc < W:
                feats.append(float(obs_ghost[rr, cc]))
            else:
                feats.append(1.0)

    # neighbor walkable flags
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        ok = (0 <= rr < H and 0 <= cc < W and int(memory_map[rr, cc]) != 1)
        feats.append(1.0 if ok else 0.0)

    feats.append(1.0 if enemy_visible else 0.0)

    if pac_est is None:
        feats.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    else:
        dx = float(pac_est[0] - r)
        dy = float(pac_est[1] - c)
        dist = float(abs(dx) + abs(dy))
        feats.extend([dx / 21.0, dy / 21.0, dist / 42.0])
        feats.append(_los_flag(memory_map, my_pos, pac_est, radius=5))
        # degree
        deg = 0
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and int(memory_map[rr, cc]) != 1:
                deg += 1
        feats.append(float(deg) / 4.0)

    feats.append(float(visit[r, c]) / 50.0)
    prev2 = prev_positions[-2] if len(prev_positions) >= 2 else None
    feats.append(1.0 if (prev2 is not None and prev2 == my_pos) else 0.0)
    feats.append(1.0 if int(memory_map[r, c]) == -1 else 0.0)

    feats.extend(_one_hot(int(last_action_idx), 5))
    feats.extend(_one_hot(int(last2_action_idx), 5))

    return np.array(feats, dtype=np.float32)


class MLPDQN(nn.Module):
    """Must match train_ghost_dqn.py architecture."""
    def __init__(self, input_dim: int, hidden1: int = 128, hidden2: int = 128, num_actions: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_actions),
        )

    def forward(self, x):
        return self.net(x)


class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rng = random.Random(kwargs.get("seed", None))
        self.memory_map = None
        self.visit = None
        self.prev_positions = deque(maxlen=10)

        self.bt = BeliefTracker(decay=0.95, visible_decay=0.10, seed=kwargs.get("seed", None))
        self.last_seen_enemy = None
        self.last_seen_step = None

        self.name = "Ghost (Hybrid RL + Safe Evade Fallback v2)"

        # Track last actions (fix ABAB)
        self.last_action_idx = MOVE_TO_IDX[Move.STAY]
        self.last2_action_idx = MOVE_TO_IDX[Move.STAY]
        self.last_move = Move.STAY

        # RL
        self.rl_enabled = bool(kwargs.get("use_rl", True))
        self.policy = None
        self._rl_input_dim = None
        self.weight_path = str(Path(__file__).parent / "ghost_policy.pth")
        if torch is None or nn is None:
            self.rl_enabled = False

    def _ensure(self, obs: np.ndarray):
        if self.memory_map is None or self.memory_map.shape != obs.shape:
            self.memory_map = np.full(obs.shape, -1, dtype=np.int8)
            self.visit = np.zeros(obs.shape, dtype=np.int16)

    def _update_memory(self, obs: np.ndarray):
        self._ensure(obs)
        visible = (obs != -1)
        self.memory_map[visible] = obs[visible]

    def _in_bounds(self, r, c):
        H, W = self.memory_map.shape
        return 0 <= r < H and 0 <= c < W

    def _walkable(self, r, c):
        return self._in_bounds(r, c) and int(self.memory_map[r, c]) != 1

    def _degree(self, pos):
        deg = 0
        r, c = pos
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if self._walkable(rr, cc):
                deg += 1
        return deg

    def _has_los(self, a, b, radius=5):
        if b is None:
            return False
        if a[0] != b[0] and a[1] != b[1]:
            return False
        dist = abs(a[0] - b[0]) + abs(a[1] - b[1])
        if dist > radius:
            return False
        dr = int(np.sign(b[0] - a[0]))
        dc = int(np.sign(b[1] - a[1]))
        r, c = a
        for _ in range(dist):
            r += dr
            c += dc
            if not self._in_bounds(r, c):
                return False
            if int(self.memory_map[r, c]) == 1:
                return False
        return True

    def _legal_mask(self, pos):
        legal = [True, True, True, True, True]
        for mv in ALL_MOVES:
            dr, dc = mv.value
            nxt = (pos[0] + dr, pos[1] + dc)
            if not self._walkable(nxt[0], nxt[1]):
                legal[MOVE_TO_IDX[mv]] = False
        legal[MOVE_TO_IDX[Move.STAY]] = True
        return legal

    # ---------------- RL load/infer ----------------
    def _maybe_load_policy(self, feat_dim: int):
        if not self.rl_enabled or self.policy is not None:
            return
        if not os.path.exists(self.weight_path):
            return
        try:
            self._rl_input_dim = int(feat_dim)
            model = MLPDQN(self._rl_input_dim, hidden1=128, hidden2=128, num_actions=5)
            state = torch.load(self.weight_path, map_location="cpu")
            model.load_state_dict(state)
            model.eval()
            self.policy = model
        except Exception:
            self.policy = None

    def _policy_q(self, feat: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(feat.astype(np.float32)).unsqueeze(0)
            q = self.policy(x).squeeze(0).cpu().numpy()
        return q

    def _update_action_memory(self, chosen_idx: int):
        self.last2_action_idx = int(self.last_action_idx)
        self.last_action_idx = int(chosen_idx)

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        self._update_memory(map_state)
        self.visit[my_position[0], my_position[1]] += 1
        self.prev_positions.append(my_position)

        if enemy_position is not None:
            self.last_seen_enemy = enemy_position
            self.last_seen_step = step_number

        self.bt.update(map_state, self.memory_map, enemy_position)
        pac_est = enemy_position if enemy_position is not None else self.bt.best_guess(self.last_seen_enemy)
        enemy_visible = (enemy_position is not None)

        # -------------------------------------------------
        # RL branch
        # -------------------------------------------------
        if self.rl_enabled and torch is not None and nn is not None:
            feat = extract_ghost_features_for_infer(
                map_state, self.memory_map, self.visit, self.prev_positions,
                my_position, pac_est, enemy_visible,
                self.last_action_idx, self.last2_action_idx
            )
            self._maybe_load_policy(feat.shape[0])

            if self.policy is not None:
                q = self._policy_q(feat)

                legal = self._legal_mask(my_position)
                # Hard mask only for illegal actions
                for i in range(5):
                    if not legal[i]:
                        q[i] -= 1e9

                # ---- SOFT guardrails (penalties) ----
                # Rationale: do not force only 1 path => reduces ping-pong.
                if pac_est is not None:
                    d0 = manhattan(my_position, pac_est)

                    # penalize reducing distance when too close (soft)
                    if d0 <= 2:
                        for mv in ALL_MOVES:
                            idx = MOVE_TO_IDX[mv]
                            if not legal[idx]:
                                continue
                            dr, dc = mv.value
                            nxt = (my_position[0] + dr, my_position[1] + dc)
                            if manhattan(nxt, pac_est) < d0:
                                q[idx] -= 80.0

                    # penalize LOS (soft, stronger when close)
                    for mv in ALL_MOVES:
                        idx = MOVE_TO_IDX[mv]
                        if not legal[idx]:
                            continue
                        dr, dc = mv.value
                        nxt = (my_position[0] + dr, my_position[1] + dc)
                        if self._has_los(nxt, pac_est, radius=5):
                            q[idx] -= (60.0 if d0 <= 6 else 25.0)

                        # penalize dead-end (soft)
                        if self._degree(nxt) <= 1:
                            q[idx] -= (50.0 if d0 <= 6 else 20.0)

                # strong anti-ABAB in inference too
                prev2 = self.prev_positions[-2] if len(self.prev_positions) >= 2 else None
                if prev2 is not None:
                    for mv in ALL_MOVES:
                        idx = MOVE_TO_IDX[mv]
                        if not legal[idx]:
                            continue
                        dr, dc = mv.value
                        nxt = (my_position[0] + dr, my_position[1] + dc)
                        if nxt == prev2:
                            q[idx] -= 120.0

                # discourage immediate reverse direction (soft)
                if self.last_move in OPPOSITE:
                    rev = OPPOSITE[self.last_move]
                    if rev in MOVE_TO_IDX:
                        ridx = MOVE_TO_IDX[rev]
                        if legal[ridx]:
                            q[ridx] -= 45.0

                best_idx = int(np.argmax(q))
                best_mv = IDX_TO_MOVE[best_idx]

                # final safety
                if best_mv != Move.STAY:
                    dr, dc = best_mv.value
                    nxt = (my_position[0] + dr, my_position[1] + dc)
                    if not self._walkable(nxt[0], nxt[1]):
                        best_mv = Move.STAY
                        best_idx = MOVE_TO_IDX[Move.STAY]

                self._update_action_memory(best_idx)
                self.last_move = best_mv
                return best_mv

        # -------------------------------------------------
        # Fallback heuristic: Safe Evade (with last_move)
        # -------------------------------------------------
        moves = ALL_MOVES[:]
        self.rng.shuffle(moves)
        moves.append(Move.STAY)

        best_mv = Move.STAY
        best_score = -1e18
        bm = self.bt.get_map()
        prev2 = self.prev_positions[-2] if len(self.prev_positions) >= 2 else None

        for mv in moves:
            if mv == Move.STAY:
                nxt = my_position
            else:
                dr, dc = mv.value
                nxt = (my_position[0] + dr, my_position[1] + dc)
                if not self._walkable(nxt[0], nxt[1]):
                    continue

            score = 0.0

            # hard ABAB block
            if prev2 is not None and nxt == prev2:
                score -= 200.0

            # reverse penalty
            if mv == OPPOSITE.get(self.last_move, Move.STAY):
                score -= 70.0

            if pac_est is not None:
                d = manhattan(nxt, pac_est)
                score += 3.0 * d
                if d <= 2:
                    score -= 140.0
                if self._has_los(nxt, pac_est, radius=5):
                    score -= 80.0

            deg = self._degree(nxt)
            if deg <= 1:
                score -= 40.0
            elif deg >= 3:
                score += 10.0

            if nxt in self.prev_positions:
                score -= 18.0

            score -= 1.2 * float(self.visit[nxt[0], nxt[1]])

            if int(self.memory_map[nxt[0], nxt[1]]) == -1:
                score += 6.0

            if bm is not None:
                score -= 60.0 * float(bm[nxt[0], nxt[1]])

            if mv == Move.STAY:
                score -= 3.0

            if score > best_score:
                best_score = score
                best_mv = mv

        chosen_idx = MOVE_TO_IDX.get(best_mv, MOVE_TO_IDX[Move.STAY])
        self._update_action_memory(chosen_idx)
        self.last_move = best_mv
        return best_mv
