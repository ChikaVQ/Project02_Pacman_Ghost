"""
agent.py
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


# ============================
# TIỆN ÍCH DÙNG CHUNG
# ============================

DIRS = [
    (Move.UP,    (-1, 0)),
    (Move.DOWN,  (1, 0)),
    (Move.LEFT,  (0, -1)),
    (Move.RIGHT, (0, 1)),
]
ALL_MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]

GRID_H = 21
GRID_W = 21


def in_bounds(r: int, c: int) -> bool:
    return 0 <= r < GRID_H and 0 <= c < GRID_W


def manhattan(a: tuple, b: tuple) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def argmax_cells(mat: np.ndarray, ratio: float = 0.95):
    """
    Lấy danh sách cell có giá trị >= ratio * max(mat).
    Dùng để lấy nhiều ứng viên khi belief có nhiều đỉnh.
    """
    m = float(np.max(mat)) if mat.size else 0.0
    if m <= 0:
        return []
    coords = np.argwhere(mat >= m * ratio)
    return [tuple(x) for x in coords]


# ============================
# BELIEF TRACKER (DÙNG CHUNG)
# ============================

class BeliefTracker:
    """
    Theo dõi belief enemy có thể ở đâu (random-walk + mask vùng quan sát).
    - Nếu thấy enemy: belief=1 tại đó
    - Nếu không thấy: propagate + mask vùng visible (obs != -1) + mask tường
    """

    def __init__(self, decay: float = 0.95):
        self.belief = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        self.decay = float(decay)

    def reset_to(self, pos: tuple):
        self.belief.fill(0.0)
        r, c = pos
        if in_bounds(r, c):
            self.belief[r, c] = 1.0

    def init_uniform(self, memory_map: np.ndarray, walkable_fn):
        """
        Nếu chưa có thông tin gì: khởi tạo belief đồng đều trên các ô không phải tường.
        walkable_fn(r,c) dùng để loại tường.
        """
        self.belief.fill(0.0)
        for r in range(GRID_H):
            for c in range(GRID_W):
                if walkable_fn(r, c):
                    self.belief[r, c] = 1.0
        self._normalize()

    def update(self, obs: np.ndarray, memory_map: np.ndarray, enemy_pos: tuple, walkable_fn):
        """
        walkable_fn(r,c): trả True nếu ô có thể đứng (không tường), cho phép -1.
        """
        if enemy_pos is not None:
            self.reset_to(enemy_pos)
            return

        # Không thấy enemy
        if np.sum(self.belief) > 0:
            self._propagate(walkable_fn)
        else:
            self.init_uniform(memory_map, walkable_fn)

        # Mask vùng visible (đã nhìn thấy mà không thấy enemy)
        visible = (obs != -1)
        self.belief[visible] = 0.0

        # Mask tường đã biết
        walls = (memory_map == 1)
        self.belief[walls] = 0.0

        self._normalize()

    def _propagate(self, walkable_fn):
        """
        Random-walk: mỗi cell phân phối đều cho:
        - chính nó (enemy có thể STAY)
        - 4 ô kề (nếu walkable)
        """
        new_b = np.zeros((GRID_H, GRID_W), dtype=np.float32)

        for r in range(GRID_H):
            for c in range(GRID_W):
                p = float(self.belief[r, c])
                if p <= 0:
                    continue

                nxts = [(r, c)]  # STAY
                for _, (dr, dc) in DIRS:
                    nr, nc = r + dr, c + dc
                    if in_bounds(nr, nc) and walkable_fn(nr, nc):
                        nxts.append((nr, nc))

                share = (p * self.decay) / max(1, len(nxts))
                for nr, nc in nxts:
                    new_b[nr, nc] += share

        self.belief = new_b
        self._normalize()

    def _normalize(self):
        s = float(np.sum(self.belief))
        if s > 0:
            self.belief /= s

    def get_target(self, last_known: tuple = None):
        """
        Trả cell belief cao nhất (hoặc một trong các cell gần max).
        Nếu có last_known: ưu tiên cell gần last_known.
        """
        if float(np.sum(self.belief)) <= 0:
            return None
        cands = argmax_cells(self.belief, ratio=0.95)
        if not cands:
            return None
        if last_known is not None:
            cands.sort(key=lambda p: manhattan(p, last_known))
        return cands[0]

    def get_map(self):
        return self.belief.copy()


# ============================
# PACMAN AGENT (SEEKER)
# ============================

class PacmanAgent(BasePacmanAgent):
    """
    Pacman (kẻ đi tìm)

    Pipeline:
    - Update memory + visit_count + belief
    - Nếu thấy Ghost:
        + Intercept target (2–4 bước) -> BFS tới điểm chặn
    - Nếu không thấy:
        + Ưu tiên last_known (nếu mới) hoặc belief_target (nếu mất dấu)
        + Nếu không có -> frontier scoring explore
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))

        self.memory_map = np.full((GRID_H, GRID_W), -1, dtype=np.int8)
        self.visit_count = np.zeros((GRID_H, GRID_W), dtype=np.int16)

        self.last_known_enemy_pos = None
        self.last_seen_step = None

        self.prev_positions = deque(maxlen=8)

        # Belief tracking (Ghost position)
        self.bt = BeliefTracker(decay=0.95)

        seed = kwargs.get("seed", None)
        self.rng = random.Random(seed)

        self.name = "Lead Pacman (Frontier + Intercept + Belief)"

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int):
        # 1) Update memory
        self._update_memory(map_state)

        # 2) Update visit_count
        r, c = my_position
        if in_bounds(r, c):
            self.visit_count[r, c] += 1

        # 3) Update last_known
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            self.last_seen_step = step_number

        # 3b) Update belief (Ghost)
        self.bt.update(
            obs=map_state,
            memory_map=self.memory_map,
            enemy_pos=enemy_position,
            walkable_fn=lambda rr, cc: self._walkable(rr, cc, allow_unknown=True),
        )

        self.prev_positions.append(my_position)

        # ============================
        # 4) CHỌN TARGET
        # ============================

        target = None

        if enemy_position is not None:
            # Thấy Ghost -> Intercept
            target = self._intercept_target(my_position, enemy_position)
        else:
            # Không thấy Ghost:
            if self.last_known_enemy_pos is not None:
                stale = (self.last_seen_step is not None and (step_number - self.last_seen_step) >= 4)
                if stale:
                    target = self.bt.get_target(last_known=self.last_known_enemy_pos) or self.last_known_enemy_pos
                else:
                    target = self.last_known_enemy_pos
            else:
                target = self.bt.get_target(last_known=None)

        # 4a) Có target -> BFS
        if target is not None:
            path = self._bfs_path(my_position, target, allow_unknown=True)
            if path and len(path) >= 2:
                mv = self._move_from_to(path[0], path[1])
                return self._pack_move_with_steps(my_position, mv)

            # Nếu đã tới last_known mà vẫn không thấy -> bỏ
            if enemy_position is None and self.last_known_enemy_pos is not None and my_position == self.last_known_enemy_pos:
                self.last_known_enemy_pos = None

        # 4b) Explore bằng frontier scoring
        best_frontier = self._best_frontier(my_position)
        if best_frontier is not None and best_frontier != my_position:
            path = self._bfs_path(my_position, best_frontier, allow_unknown=True)
            if path and len(path) >= 2:
                mv = self._move_from_to(path[0], path[1])
                return self._pack_move_with_steps(my_position, mv)

        # 5) Fallback move
        for mv in self._ordered_moves(my_position):
            if self._can_step(my_position, mv):
                return (mv, 1)

        return (Move.STAY, 1)

    # ============================
    # MEMORY & MOVE
    # ============================

    def _update_memory(self, obs: np.ndarray):
        visible = obs != -1
        self.memory_map[visible] = obs[visible]

    def _walkable(self, r: int, c: int, allow_unknown: bool) -> bool:
        if not in_bounds(r, c):
            return False
        v = int(self.memory_map[r, c])
        if v == 1:
            return False
        if v == -1:
            return bool(allow_unknown)
        return True

    def _is_ghost_walkable(self, r: int, c: int) -> bool:
        # Ghost giả định: chỉ tránh tường, cho phép -1
        return self._walkable(r, c, allow_unknown=True)

    def _can_step(self, pos: tuple, mv: Move) -> bool:
        dr, dc = mv.value
        return self._walkable(pos[0] + dr, pos[1] + dc, allow_unknown=True)

    def _pack_move_with_steps(self, pos: tuple, mv: Move):
        steps = 0
        r, c = pos
        dr, dc = mv.value
        for _ in range(self.pacman_speed):
            nr, nc = r + dr, c + dc
            if not self._walkable(nr, nc, allow_unknown=True):
                break
            steps += 1
            r, c = nr, nc
        if steps <= 0:
            return (Move.STAY, 1)
        return (mv, steps)

    def _move_from_to(self, a: tuple, b: tuple) -> Move:
        dr, dc = b[0] - a[0], b[1] - a[1]
        for mv, (r, c) in DIRS:
            if (dr, dc) == (r, c):
                return mv
        return Move.STAY

    def _pacman_time(self, d_pacman: int) -> int:
        # ceil(d / pacman_speed)
        if d_pacman <= 0:
            return 0
        return (d_pacman + self.pacman_speed - 1) // self.pacman_speed

    # ============================
    # BFS
    # ============================

    def _bfs_path(self, start: tuple, goal: tuple, allow_unknown: bool):
        if start == goal:
            return [start]
        q = deque([start])
        parent = {start: None}

        while q:
            cur = q.popleft()
            if cur == goal:
                break
            for _, (dr, dc) in DIRS:
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt in parent:
                    continue
                if not self._walkable(nxt[0], nxt[1], allow_unknown):
                    continue
                parent[nxt] = cur
                q.append(nxt)

        if goal not in parent:
            return None

        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path

    def _bfs_dist(self, start: tuple, goal: tuple, allow_unknown: bool) -> int:
        if start == goal:
            return 0
        q = deque([start])
        dist = {start: 0}

        while q:
            cur = q.popleft()
            dcur = dist[cur]
            for _, (dr, dc) in DIRS:
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt in dist:
                    continue
                if not self._walkable(nxt[0], nxt[1], allow_unknown):
                    continue
                dist[nxt] = dcur + 1
                if nxt == goal:
                    return dist[nxt]
                q.append(nxt)

        return -1

    # ============================
    # FRONTIER SCORING
    # ============================

    def _is_frontier(self, pos: tuple) -> bool:
        r, c = pos
        if not in_bounds(r, c):
            return False
        if int(self.memory_map[r, c]) == 1:
            return False
        for _, (dr, dc) in DIRS:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc) and int(self.memory_map[nr, nc]) == -1:
                return True
        return False

    def _unknown_gain(self, pos: tuple, radius: int = 2) -> int:
        r, c = pos
        r0 = max(0, r - radius)
        r1 = min(GRID_H, r + radius + 1)
        c0 = max(0, c - radius)
        c1 = min(GRID_W, c + radius + 1)
        window = self.memory_map[r0:r1, c0:c1]
        return int(np.sum(window == -1))

    def _visit_penalty(self, pos: tuple) -> int:
        r, c = pos
        if not in_bounds(r, c):
            return 999
        return int(self.visit_count[r, c])

    def _best_frontier(self, start: tuple):
        q = deque([start])
        seen = {start}
        candidates = []

        while q:
            cur = q.popleft()
            if self._is_frontier(cur):
                candidates.append(cur)
                if len(candidates) >= 20:
                    break
            for _, (dr, dc) in DIRS:
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt in seen:
                    continue
                if not self._walkable(nxt[0], nxt[1], allow_unknown=True):
                    continue
                seen.add(nxt)
                q.append(nxt)

        if not candidates:
            return None

        best = None
        best_score = None

        for f in candidates:
            d = self._bfs_dist(start, f, allow_unknown=True)
            if d < 0:
                continue
            gain = self._unknown_gain(f, radius=2)
            vpen = self._visit_penalty(f)
            score = gain - 2 * d - 1 * vpen
            if best_score is None or score > best_score:
                best_score = score
                best = f

        return best

    # ============================
    # ORDER MOVES (ANTI-LOOP)
    # ============================

    def _ordered_moves(self, pos: tuple):
        moves = ALL_MOVES[:]
        self.rng.shuffle(moves)

        def score(mv: Move):
            dr, dc = mv.value
            nxt = (pos[0] + dr, pos[1] + dc)
            if not self._walkable(nxt[0], nxt[1], allow_unknown=True):
                return 10_000
            penalty = 0
            if nxt in self.prev_positions:
                penalty += 10
            penalty += self._visit_penalty(nxt)
            if in_bounds(nxt[0], nxt[1]) and int(self.memory_map[nxt[0], nxt[1]]) == -1:
                penalty -= 2
            return penalty

        moves.sort(key=score)
        return moves

    # ============================
    # INTERCEPT
    # ============================

    def _is_junction(self, pos: tuple) -> bool:
        r, c = pos
        if not in_bounds(r, c) or int(self.memory_map[r, c]) == 1:
            return False
        deg = 0
        for _, (dr, dc) in DIRS:
            if self._walkable(r + dr, c + dc, allow_unknown=False):
                deg += 1
        return deg >= 3

    def _intercept_target(self, my_pos: tuple, enemy_pos: tuple) -> tuple:
        """
        Dự đoán Ghost chạy 2–4 bước (cho phép đi qua -1), chọn điểm chặn tối ưu.
        """
        max_ghost_steps = 4

        q = deque([enemy_pos])
        ghost_dist = {enemy_pos: 0}
        potential = []

        while q:
            cur = q.popleft()
            dcur = ghost_dist[cur]
            if dcur >= max_ghost_steps:
                continue
            if dcur >= 1:
                potential.append(cur)

            for _, (dr, dc) in DIRS:
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt in ghost_dist:
                    continue
                if not self._is_ghost_walkable(nxt[0], nxt[1]):
                    continue
                ghost_dist[nxt] = dcur + 1
                q.append(nxt)

        if not potential:
            return enemy_pos

        best_target = enemy_pos
        best_score = -float("inf")

        for t in potential:
            d_pac = self._bfs_dist(my_pos, t, allow_unknown=True)
            d_gho = ghost_dist.get(t, -1)
            if d_pac < 0 or d_gho < 0:
                continue

            t_pac = self._pacman_time(d_pac)
            t_gho = d_gho

            diff_penalty = 2 * abs(t_pac - t_gho)
            late_penalty = 30 if t_pac > t_gho else 0
            d_cost = 3 * d_pac
            junction_bonus = 6 if self._is_junction(t) else 0

            cost = diff_penalty + late_penalty + d_cost - junction_bonus
            score = -cost

            if score > best_score:
                best_score = score
                best_target = t

        # fallback nếu chặn quá tệ và quá xa
        if best_score < -35 and self._bfs_dist(my_pos, best_target, allow_unknown=True) > 6:
            return enemy_pos

        return best_target


# ============================
# GHOST AGENT (HIDER)
# ============================

class GhostAgent(BaseGhostAgent):
    """
    Ghost (kẻ trốn)

    - Update memory + visit_count + belief (Pacman position)
    - Nếu thấy Pacman: dùng threat thật
    - Nếu không thấy: dùng threat ước lượng từ belief_target để né danger/occlusion
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.memory_map = np.full((GRID_H, GRID_W), -1, dtype=np.int8)
        self.visit_count = np.zeros((GRID_H, GRID_W), dtype=np.int16)

        self.last_known_enemy_pos = None
        self.prev_positions = deque(maxlen=8)

        # Belief tracking (Pacman position)
        self.bt = BeliefTracker(decay=0.95)

        seed = kwargs.get("seed", None)
        self.rng = random.Random(seed)

        self.name = "Lead Ghost (Dead-end + Occlusion + Belief)"

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        # 1) Update memory
        self._update_memory(map_state)

        # 2) Update visit_count
        r, c = my_position
        if in_bounds(r, c):
            self.visit_count[r, c] += 1

        # 3) Update last_known
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        # 3b) Update belief (Pacman)
        self.bt.update(
            obs=map_state,
            memory_map=self.memory_map,
            enemy_pos=enemy_position,
            walkable_fn=lambda rr, cc: self._walkable(rr, cc),
        )

        self.prev_positions.append(my_position)

        # threat: ưu tiên visible, nếu không thì lấy ước lượng belief
        threat = enemy_position
        if threat is None:
            threat = self.bt.get_target(last_known=self.last_known_enemy_pos) or self.last_known_enemy_pos

        if threat is not None:
            mv = self._evade_move(my_position, threat)
            if mv is not None:
                return mv

        mv = self._roam_move(my_position, threat_est=threat)
        if mv is not None:
            return mv

        return Move.STAY

    # ============================
    # MEMORY / WALKABLE / DEGREE
    # ============================

    def _update_memory(self, obs: np.ndarray):
        visible = obs != -1
        self.memory_map[visible] = obs[visible]

    def _walkable(self, r: int, c: int) -> bool:
        # Ghost cho phép đi vào -1 để trốn, chỉ tránh tường
        return in_bounds(r, c) and int(self.memory_map[r, c]) != 1

    def _visit_penalty(self, pos: tuple) -> int:
        r, c = pos
        if not in_bounds(r, c):
            return 999
        return int(self.visit_count[r, c])

    def _degree(self, pos: tuple) -> int:
        r, c = pos
        if not in_bounds(r, c) or int(self.memory_map[r, c]) == 1:
            return 0
        deg = 0
        for _, (dr, dc) in DIRS:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc) and int(self.memory_map[nr, nc]) != 1:
                deg += 1
        return deg

    # ============================
    # OCCLUSION HELPER
    # ============================

    def _has_line_of_sight(self, pos1: tuple, pos2: tuple) -> bool:
        """
        True nếu:
        - cùng hàng hoặc cùng cột
        - khoảng cách Manhattan <= 5
        - không có tường chặn giữa
        """
        r1, c1 = pos1
        r2, c2 = pos2

        if r1 != r2 and c1 != c2:
            return False

        dist = abs(r1 - r2) + abs(c1 - c2)
        if dist > 5:
            return False

        if r1 == r2:
            c_min, c_max = (c1, c2) if c1 < c2 else (c2, c1)
            for c in range(c_min + 1, c_max):
                if int(self.memory_map[r1, c]) == 1:
                    return False
        else:
            r_min, r_max = (r1, r2) if r1 < r2 else (r2, r1)
            for r in range(r_min + 1, r_max):
                if int(self.memory_map[r, c1]) == 1:
                    return False

        return True

    # ============================
    # GHOST POLICY
    # ============================

    def _evade_move(self, pos: tuple, threat: tuple) -> Move:
        """
        Score tổng:
          + xa threat
          + thưởng vào fog
          - anti-loop
          - dead-end
          - danger-aware (gần threat)
          - occlusion (cùng hàng/cột <=5 không có tường che)
          - danger_map (belief): phạt cell có xác suất Pacman cao
        """
        W_DANGER = 5
        DANGER_RADIUS = 5
        W_OCCLUDE = 20
        W_BELIEF = 80  # tăng/giảm tuỳ bạn (phạt mạnh khi belief cao)

        danger_map = self.bt.get_map()

        best_mv = Move.STAY
        best_score = None

        moves = ALL_MOVES[:]
        self.rng.shuffle(moves)

        for mv in moves:
            dr, dc = mv.value
            nxt = (pos[0] + dr, pos[1] + dc)
            if not self._walkable(nxt[0], nxt[1]):
                continue

            score = 0

            # 1) Xa threat
            score += 3 * manhattan(nxt, threat)

            # 2) Thưởng vào fog
            if int(self.memory_map[nxt[0], nxt[1]]) == -1:
                score += 4

            # 3) Anti-loop
            if nxt in self.prev_positions:
                score -= 15
            score -= 2 * self._visit_penalty(nxt)

            # 4) Dead-end avoidance
            deg = self._degree(nxt)
            if deg <= 1:
                score -= 25
            elif deg >= 3:
                score += 6

            # 5) Danger-aware theo khoảng cách
            d = manhattan(nxt, threat)
            if d <= DANGER_RADIUS:
                score -= W_DANGER * (DANGER_RADIUS + 1 - d)

            # 6) Occlusion (tránh bị nhìn thấy theo hình dấu +)
            if self._has_line_of_sight(nxt, threat):
                score -= W_OCCLUDE

            # 7) Belief danger map (khi Pacman không visible thì rất hữu ích)
            score -= W_BELIEF * float(danger_map[nxt[0], nxt[1]])

            if best_score is None or score > best_score:
                best_score = score
                best_mv = mv

        return best_mv

    def _roam_move(self, pos: tuple, threat_est: tuple = None) -> Move:
        """
        Khi không thấy threat rõ ràng:
        - ưu tiên vào fog
        - tránh loop
        - tránh dead-end nếu có thể
        - nếu có threat_est (từ belief): cũng né occlusion/danger nhẹ
        """
        moves = self._ordered_moves(pos, threat_est=threat_est)
        for mv in moves:
            dr, dc = mv.value
            nxt = (pos[0] + dr, pos[1] + dc)
            if not self._walkable(nxt[0], nxt[1]):
                continue
            if self._degree(nxt) <= 1:
                continue
            return mv

        for mv in moves:
            dr, dc = mv.value
            nxt = (pos[0] + dr, pos[1] + dc)
            if self._walkable(nxt[0], nxt[1]):
                return mv

        return None

    def _ordered_moves(self, pos: tuple, threat_est: tuple = None):
        """
        Ưu tiên: vào fog, ít loop, ít visit, tránh dead-end.
        Nếu có threat_est: phạt thêm occlusion/danger nhẹ.
        """
        W_DANGER = 3
        DANGER_RADIUS = 5
        W_OCCLUDE = 10

        moves = ALL_MOVES[:]
        self.rng.shuffle(moves)

        def penalty(mv: Move):
            dr, dc = mv.value
            nxt = (pos[0] + dr, pos[1] + dc)

            if not self._walkable(nxt[0], nxt[1]):
                return 10_000

            pen = 0
            if nxt in self.prev_positions:
                pen += 10
            pen += self._visit_penalty(nxt)

            if int(self.memory_map[nxt[0], nxt[1]]) == -1:
                pen -= 3

            if self._degree(nxt) <= 1:
                pen += 5

            if threat_est is not None:
                d = manhattan(nxt, threat_est)
                if d <= DANGER_RADIUS:
                    pen += W_DANGER * (DANGER_RADIUS + 1 - d)
                if self._has_line_of_sight(nxt, threat_est):
                    pen += W_OCCLUDE

            return pen

        moves.sort(key=penalty)
        return moves