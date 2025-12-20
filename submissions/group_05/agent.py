"""
agent.py (PATCHED + IMPROVED) — based on YOUR latest code, with the 5 fixes you requested.

✅ Fixes applied (as you asked):
1) BeliefTracker._propagate() now NORMALIZES (prevents belief dying / drifting).
2) BeliefTracker.best_guess() now RANDOM-CHOICES among top candidates (breaks deterministic loops).
3) Pacman intercept prediction is AXIS-ONLY (no diagonal impossible targets).
4) Tie-break reversal penalty uses the CURRENT state's last_mv (not start_last_move).
5) 2-step momentum bonus is CONTEXT-AWARE (only strong when it actually helps; near capture it won’t overshoot).

➕ Also fixed (important to stop “chasing fake”):
6) allow_one_unknown_finish no longer “lies” about reaching target; it only marks success if the STRICT move also reaches target.
   (So Pacman won’t pick an action that doesn't actually progress.)
7) Frontier scoring uses TIME-TURN estimate (not Manhattan), so it matches teacher rule better.

Teacher rule enforced:
- Turn (mv != last_mv): must be 1 step.
- Straight (mv == last_mv): may be 1 or 2 steps (agent decides).
- Never exceed pacman_speed.

Run target:
python3 arena.py --seek example_student --hide group_05 --pacman-speed 2 --capture-distance 2 --pacman-obs-radius 5 --ghost-obs-radius 5 --step-timeout 1 --delay 0.05
"""

import sys
from pathlib import Path
from collections import deque
import random
import numpy as np

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move

ALL_MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
DIRS = [(Move.UP, (-1, 0)), (Move.DOWN, (1, 0)), (Move.LEFT, (0, -1)), (Move.RIGHT, (0, 1))]
OPPOSITE = {Move.UP: Move.DOWN, Move.DOWN: Move.UP, Move.LEFT: Move.RIGHT, Move.RIGHT: Move.LEFT, Move.STAY: Move.STAY}


def manhattan(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# =========================================================
# Belief tracker (FIXED normalize + random tie-break)
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

        # downweight visible region (enemy not seen there)
        visible = (obs_map != -1)
        self.belief[visible] *= self.visible_decay
        self.belief[memory_map == 1] = 0.0

        # normalize (and re-init if degenerate)
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

        # ✅ FIX #1: normalize here (prevents belief dying/drifting)
        s = float(newb.sum())
        if s <= 1e-9:
            # fallback uniform later in update()
            self.belief = newb
        else:
            self.belief = newb / s

    def best_guess(self, last_known=None):
        if self.belief is None or float(self.belief.sum()) <= 1e-9:
            return None
        mx = float(self.belief.max())
        if mx <= 0:
            return None

        coords = np.argwhere(self.belief >= 0.90 * mx)
        cands = [tuple(x) for x in coords]
        if not cands:
            return None

        if last_known is not None:
            # keep top few closest to last_known, then random
            cands.sort(key=lambda p: manhattan(p, last_known))
            cands = cands[:8]

        # ✅ FIX #2: random among candidates
        return self.rng.choice(cands)

    def get_map(self):
        return None if self.belief is None else self.belief.copy()


# =========================================================
# PACMAN (Seeker)
# =========================================================

class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self.capture_threshold = int(kwargs.get("capture_distance", 2))
        # engine uses distance < capture_distance => if capture_distance=2 => win when manhattan<=1
        self.capture_ring = max(0, self.capture_threshold - 1)

        self.rng = random.Random(kwargs.get("seed", None))

        self.memory_map = None
        self.visit = None

        self.bt = BeliefTracker(decay=0.95, visible_decay=0.05, seed=kwargs.get("seed", None))

        self.last_move = Move.STAY
        self.prev_positions = deque(maxlen=6)

        self.last_seen_enemy = None
        self.last_seen_step = None

        self.name = "Pacman (Momentum BFS + Capture-ring Intercept) [Fixed]"

    # ---------- Map Helpers ----------
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

    # ---------- Frontier Helpers ----------
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

    # ---------- Teacher Rule Logic ----------
    def _allowed_steps(self, mv: Move, last_mv: Move):
        if mv == Move.STAY:
            return [1]
        if mv != last_mv:
            return [1]  # turning constraint
        if self.pacman_speed >= 2:
            return [2, 1]  # straight: agent may choose
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

    # ---------- Capture Logic ----------
    def _capture_cells(self, ghost_pos):
        """cells where pacman wins if it ends there (within capture_ring of ghost)."""
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

    # ---------- Time-turn BFS helpers ----------
    def _estimate_turns_single(self, start, last_mv, goal, max_turns=18):
        """Turn-based BFS using teacher rule; strict_known only (fast + safe)."""
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
            """
            ✅ FIX #4: reversal penalty uses curr_last_mv (state), not start_last_move.
            ✅ FIX #5: momentum bonus depends on whether it actually helps (distance & not overshoot near capture).
            """
            val = 0.0

            # anti-backtrack
            if prev_cell is not None and moved_pos == prev_cell:
                val -= 120.0

            # reversal relative to CURRENT state's last move
            if mv == OPPOSITE.get(curr_last_mv, Move.STAY):
                val -= 60.0

            # visit penalty
            val -= 2.0 * float(self.visit[moved_pos[0], moved_pos[1]])

            # momentum bonus (context-aware)
            # strong only if:
            # - actual moved_steps==2
            # - and we're not "near capture" (to avoid overshoot)
            if moved_steps == 2:
                # approximate closeness: near any target => don't over-reward 2-steps
                # (target_set small often: capture-ring)
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

                    # propagate first action
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

                    # ✅ IMPORTANT: do NOT "fake success" with unknown peek.
                    # allow_one_unknown_finish only relaxes STRICTNESS for scoring IF the strict move ALSO lands in target.
                    is_target = (nxt_pos in target_set)

                    if is_target:
                        tv = tie_value(pos, last_mv, mv, st, nxt_pos, moved)
                        if best_turns is None or nd < best_turns or (nd == best_turns and tv > best_tie):
                            best_turns = nd
                            best_tie = tv
                            best_act = fa

            # optional: if enabled, allow exploring UNKNOWN as intermediate (still real movement)
            # but must not claim success unless strict path reaches target.
            if allow_one_unknown_finish and strict_known:
                for mv in ALL_MOVES:
                    # only try 1-step unknown as a controlled risk
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

    # ---------- Intercept prediction (AXIS-ONLY) ----------
    def _axis_escape_predictions(self, ghost_pos, my_pos, steps_ahead=(2, 3, 4)):
        """
        ✅ FIX #3: predict along ONE axis only (no diagonal impossible targets).
        We choose axis by which delta is larger.
        """
        if ghost_pos is None:
            return []

        gr, gc = ghost_pos
        pr, pc = my_pos
        dr = gr - pr
        dc = gc - pc

        preds = []

        # choose dominant axis
        if abs(dr) >= abs(dc):
            step_r = int(np.sign(dr)) if dr != 0 else 0
            step_c = 0
        else:
            step_r = 0
            step_c = int(np.sign(dc)) if dc != 0 else 0

        # If both 0 (same cell), no prediction
        if step_r == 0 and step_c == 0:
            return preds

        for k in steps_ahead:
            rr, cc = gr + step_r * k, gc + step_c * k
            if self._in_bounds(rr, cc):
                preds.append((int(rr), int(cc)))
        return preds

    # ---------- Main Step ----------
    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int):
        self._update_memory(map_state)
        self.visit[my_position[0], my_position[1]] += 1
        self.prev_positions.append(my_position)

        if enemy_position is not None:
            self.last_seen_enemy = enemy_position
            self.last_seen_step = step_number

        self.bt.update(map_state, self.memory_map, enemy_position)

        # immediate capture (engine will check distance < capture_threshold)
        if enemy_position is not None and manhattan(my_position, enemy_position) <= self.capture_ring:
            return (Move.STAY, 1)

        ghost_est = enemy_position if enemy_position is not None else self.bt.best_guess(self.last_seen_enemy)

        # 1) Intercept / chase capture-ring
        if ghost_est is not None:
            raw_targets = self._capture_cells(ghost_est)
            target_cells = {t for t in raw_targets if self._walkable_possible(t[0], t[1])}

            # if visible and close, add axis-only escape predictions
            dist_to_est = manhattan(my_position, ghost_est)
            if enemy_position is not None and dist_to_est <= 8:
                for p in self._axis_escape_predictions(ghost_est, my_position):
                    if self._walkable_possible(p[0], p[1]):
                        target_cells.add(p)

            allow_unknown = (dist_to_est <= 4)  # controlled risk only when close

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

                # enforce teacher rule strictly at execution time
                if mv != self.last_move:
                    steps = 1
                else:
                    steps = min(steps, 2, self.pacman_speed)

                # cap by speed
                steps = min(steps, self.pacman_speed)

                # execute on strict-known (to avoid requesting illegal movement through unknown walls)
                nxt, moved = self._apply_action(my_position, mv, steps, strict_known=True)
                if moved > 0:
                    self.last_move = mv
                    return (mv, moved)

        # 2) Exploration with frontiers (TIME-TURN not Manhattan)
        frontiers = self._frontiers()
        if frontiers:
            self.rng.shuffle(frontiers)
            sample = frontiers[:30]  # small for timeout=1s

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

                # anti-backtrack
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

                    nxt, moved = self._apply_action(my_position, mv, steps, strict_known=True)
                    if moved > 0:
                        self.last_move = mv
                        return (mv, moved)

        # 3) Fallback greedy (safe + anti-loop)
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
                    score -= 30.0  # don't overshoot near capture
                candidates.append((score, mv, moved))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            _, mv, moved = candidates[0]
            self.last_move = mv
            return (mv, moved)

        return (Move.STAY, 1)


# =========================================================
# GHOST (Hider) — kept close to your version, small fixes only
# =========================================================

class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rng = random.Random(kwargs.get("seed", None))
        self.memory_map = None
        self.visit = None
        self.prev_positions = deque(maxlen=10)
        self.bt = BeliefTracker(decay=0.95, visible_decay=0.10, seed=kwargs.get("seed", None))
        self.last_seen_enemy = None
        self.name = "Ghost (Safe Evade)"

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
        for _, (dr, dc) in DIRS:
            if self._walkable(pos[0] + dr, pos[1] + dc):
                deg += 1
        return deg

    def _has_los(self, a, b, radius=5):
        # same row/col, no wall in between, within radius
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

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        self._update_memory(map_state)
        self.visit[my_position[0], my_position[1]] += 1
        self.prev_positions.append(my_position)

        if enemy_position is not None:
            self.last_seen_enemy = enemy_position

        self.bt.update(map_state, self.memory_map, enemy_position)
        pac_est = enemy_position if enemy_position is not None else self.bt.best_guess(self.last_seen_enemy)

        moves = ALL_MOVES[:]
        self.rng.shuffle(moves)
        moves.append(Move.STAY)

        best_mv = Move.STAY
        best_score = -1e18

        bm = self.bt.get_map()

        for mv in moves:
            if mv == Move.STAY:
                nxt = my_position
            else:
                dr, dc = mv.value
                nxt = (my_position[0] + dr, my_position[1] + dc)
                if not self._walkable(nxt[0], nxt[1]):
                    continue

            score = 0.0

            if pac_est is not None:
                d = manhattan(nxt, pac_est)
                score += 3.0 * d
                if d <= 2:
                    score -= 120.0
                if self._has_los(nxt, pac_est, radius=5):
                    score -= 80.0

            deg = self._degree(nxt)
            if deg <= 1:
                score -= 40.0
            elif deg >= 3:
                score += 10.0

            if nxt in self.prev_positions:
                score -= 15.0

            score -= 1.0 * float(self.visit[nxt[0], nxt[1]])

            if int(self.memory_map[nxt[0], nxt[1]]) == -1:
                score += 6.0

            if bm is not None:
                score -= 60.0 * float(bm[nxt[0], nxt[1]])

            if mv == Move.STAY:
                score -= 2.0

            if score > best_score:
                best_score = score
                best_mv = mv

        return best_mv
