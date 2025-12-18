"""
agent.py – Bản LEAD cho CSC14003 Project 2 (Limited Vision)

Mục tiêu thiết kế (baseline mạnh + leo sớm):
- Đúng interface (PacmanAgent / GhostAgent, chữ ký hàm step)
- Có bộ nhớ bản đồ (memory_map) 21x21: {-1: chưa biết, 0: trống, 1: tường}
- Có visit_count + anti-loop để giảm chạy vòng
- Pacman:
    + Thấy Ghost -> BFS đuổi
    + Mất dấu -> tới vị trí cuối cùng đã thấy
    + Không có thông tin -> chọn frontier tốt nhất (frontier scoring) để khám phá
- Ghost:
    + Thấy Pacman -> chạy xa (điểm số tránh loop, ưu tiên vào fog)
    + Tránh góc chết (dead-end) bằng degree(cell)
    + Không thấy -> roam ưu tiên vùng ít bị lặp và có cơ hội vào fog
- Tối ưu thời gian (< 1 giây / step): BFS trên 21x21 là đủ nhẹ

LƯU Ý:
- map_state mỗi step là quan sát cục bộ:
    1 = tường (luôn thấy)
    0 = ô trống (chỉ thấy trong tầm nhìn)
   -1 = ô chưa quan sát
- Agents phải coi -1 là "chưa biết", không được mặc định là đi được,
  nhưng để khám phá, baseline này CHO PHÉP bước vào -1 (tức treat as traversable, trừ tường).
"""

import sys
from pathlib import Path
from collections import deque
import random
import numpy as np

# Thêm thư mục src vào sys.path để import framework của Arena
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


# ============================
# PACMAN AGENT (SEEKER)
# ============================

class PacmanAgent(BasePacmanAgent):
    """
    Pacman (kẻ đi tìm)
    Mục tiêu: bắt Ghost

    Chiến lược:
    1) Update memory_map + visit_count
    2) Nếu thấy Ghost -> BFS đuổi theo đường ngắn nhất
    3) Nếu không thấy nhưng có last_known -> BFS tới đó
    4) Nếu không có thông tin -> chọn frontier tốt nhất bằng frontier scoring để khám phá
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))

        # Bản đồ nhớ
        self.memory_map = np.full((GRID_H, GRID_W), -1, dtype=np.int8)

        # Đếm số lần ghé thăm (anti-loop + scoring)
        self.visit_count = np.zeros((GRID_H, GRID_W), dtype=np.int16)

        # Theo dõi đối thủ
        self.last_known_enemy_pos = None
        self.last_seen_step = None

        # Lưu vài vị trí gần đây để tránh loop ngắn
        self.prev_positions = deque(maxlen=8)

        # ============================
        # BELIEF TRACKING (Tâm)
        # ============================
        # Belief map: xác suất enemy có thể ở đâu khi không nhìn thấy
        self.belief_map = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        self.belief_decay = 0.95  # Hệ số giảm belief khi lan truyền

        seed = kwargs.get("seed", None)
        self.rng = random.Random(seed)

        self.name = "Lead Pacman (Frontier + Anti-loop)"

    def step(self,
             map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int):
        # 1) Update memory
        self._update_memory(map_state)

        # 2) Update visit_count
        r, c = my_position
        if in_bounds(r, c):
            self.visit_count[r, c] += 1

        # 3) Update enemy memory
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            self.last_seen_step = step_number

        # 3b) Update belief map (Tâm's work)
        self._update_belief(map_state, my_position, enemy_position)

        self.prev_positions.append(my_position)

        # 4) Xác định mục tiêu: visible -> last_known -> explore
        target = enemy_position or self.last_known_enemy_pos

        # 4a) Có target -> BFS chase
        if target is not None:
            path = self._bfs_path(my_position, target, allow_unknown=True)
            if path and len(path) >= 2:
                mv = self._move_from_to(path[0], path[1])
                return self._pack_move_with_steps(my_position, mv)

            # Nếu đã tới last_known mà vẫn không thấy -> bỏ mục tiêu
            if enemy_position is None and my_position == target:
                self.last_known_enemy_pos = None

        # 4b) Explore bằng frontier scoring (leo sớm)
        best_frontier = self._best_frontier(my_position)
        if best_frontier is not None and best_frontier != my_position:
            path = self._bfs_path(my_position, best_frontier, allow_unknown=True)
            if path and len(path) >= 2:
                mv = self._move_from_to(path[0], path[1])
                return self._pack_move_with_steps(my_position, mv)

        # 5) Fallback: chọn move hợp lệ tốt nhất theo score nhỏ
        for mv in self._ordered_moves(my_position):
            if self._can_step(my_position, mv):
                return (mv, 1)

        return (Move.STAY, 1)

    # ============================
    # MEMORY & MOVE
    # ============================

    def _update_memory(self, obs: np.ndarray):
        """Ghi các ô nhìn thấy vào memory_map"""
        visible = obs != -1
        self.memory_map[visible] = obs[visible]

    def _walkable(self, r: int, c: int, allow_unknown: bool) -> bool:
        """Ô đi được = không phải tường; -1 chỉ đi nếu allow_unknown=True"""
        if not in_bounds(r, c):
            return False
        v = self.memory_map[r, c]
        if v == 1:
            return False
        if v == -1:
            return allow_unknown
        return True  # 0

    def _can_step(self, pos: tuple, mv: Move) -> bool:
        dr, dc = mv.value
        return self._walkable(pos[0] + dr, pos[1] + dc, allow_unknown=True)

    def _pack_move_with_steps(self, pos: tuple, mv: Move):
        """
        Pacman trả về (Move, steps) với steps <= pacman_speed.
        Baseline này cho phép bước vào -1 để khám phá (trừ tường).
        """
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

    # ============================
    # BFS
    # ============================

    def _bfs_path(self, start: tuple, goal: tuple, allow_unknown: bool):
        """BFS tìm đường trên memory_map (nhẹ vì map 21x21)"""
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
        """BFS distance (trả về số bước, -1 nếu không tới được). Nhanh hơn reconstruct path."""
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
    # FRONTIER SCORING (LEO SỚM)
    # ============================

    def _is_frontier(self, pos: tuple) -> bool:
        """Frontier: ô không phải tường và kề ít nhất 1 ô -1"""
        r, c = pos
        if not in_bounds(r, c):
            return False
        if self.memory_map[r, c] == 1:
            return False

        for _, (dr, dc) in DIRS:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc) and self.memory_map[nr, nc] == -1:
                return True
        return False

    def _unknown_gain(self, pos: tuple, radius: int = 2) -> int:
        """Đếm số ô -1 trong cửa sổ quanh pos (unknown gain)."""
        r, c = pos
        r0 = max(0, r - radius)
        r1 = min(GRID_H, r + radius + 1)
        c0 = max(0, c - radius)
        c1 = min(GRID_W, c + radius + 1)
        window = self.memory_map[r0:r1, c0:c1]
        return int(np.sum(window == -1))

    def _visit_penalty(self, pos: tuple) -> int:
        """Phạt ô ghé thăm nhiều (giảm loop)."""
        r, c = pos
        if not in_bounds(r, c):
            return 999
        return int(self.visit_count[r, c])

    def _best_frontier(self, start: tuple):
        """
        Chọn frontier tốt nhất theo điểm:
            score = + unknown_gain(radius=2)
                    - 2*path_len
                    - 1*visit_penalty
        Lấy tối đa một số lượng frontier gần start để giữ tốc độ.
        """
        # Thu thập candidate frontier bằng BFS lan ra (tối đa 20 cái là đủ)
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

            score = (gain) - 2 * d - 1 * vpen

            if best_score is None or score > best_score:
                best_score = score
                best = f

        return best

    # ============================
    # ORDER MOVES (ANTI-LOOP)
    # ============================

    def _ordered_moves(self, pos: tuple):
        """Sắp xếp move để giảm loop: tránh quay về vị trí vừa đi, ưu tiên vào fog."""
        moves = ALL_MOVES[:]
        self.rng.shuffle(moves)

        def score(mv: Move):
            dr, dc = mv.value
            nxt = (pos[0] + dr, pos[1] + dc)

            # Move không hợp lệ -> phạt rất nặng
            if not self._walkable(nxt[0], nxt[1], allow_unknown=True):
                return 10_000

            penalty = 0

            # tránh loop ngắn
            if nxt in self.prev_positions:
                penalty += 10

            # phạt ghé thăm nhiều
            penalty += self._visit_penalty(nxt)

            # thưởng nhẹ nếu bước vào fog (khám phá)
            if in_bounds(nxt[0], nxt[1]) and self.memory_map[nxt[0], nxt[1]] == -1:
                penalty -= 2

            return penalty

        moves.sort(key=score)
        return moves

    # ============================
    # BELIEF TRACKING (Tâm's work)
    # ============================

    def _update_belief(self, obs: np.ndarray, my_pos: tuple, enemy_pos: tuple):
        """
        Cập nhật belief map:
        - Nếu enemy_pos != None: belief = 1 tại vị trí đó, các ô khác = 0
        - Nếu enemy_pos == None:
            + Lan truyền belief sang các ô kề (random-walk)
            + Mask belief = 0 tại các ô Pacman quan sát mà không thấy enemy
        """
        if enemy_pos is not None:
            # Thấy enemy -> reset belief, chỉ tập trung vào vị trí đó
            self.belief_map.fill(0.0)
            r, c = enemy_pos
            if in_bounds(r, c):
                self.belief_map[r, c] = 1.0
        else:
            # Không thấy enemy -> lan truyền belief
            if np.sum(self.belief_map) > 0:
                self._propagate_belief()
            else:
                # Chưa có thông tin gì -> khởi tạo belief đồng đều vào các ô không phải tường
                for r in range(GRID_H):
                    for c in range(GRID_W):
                        if self.memory_map[r, c] != 1:  # không phải tường
                            self.belief_map[r, c] = 1.0
                # Normalize
                total = np.sum(self.belief_map)
                if total > 0:
                    self.belief_map /= total

            # Mask các ô Pacman quan sát mà không thấy enemy
            # obs != -1 là các ô nhìn thấy
            visible = obs != -1
            self.belief_map[visible] = 0.0

            # Normalize lại belief
            total = np.sum(self.belief_map)
            if total > 0:
                self.belief_map /= total

    def _propagate_belief(self):
        """
        Lan truyền belief theo mô hình random-walk:
        Mỗi ô có belief sẽ phân phối đều cho các ô kề có thể đi được
        """
        new_belief = np.zeros((GRID_H, GRID_W), dtype=np.float32)

        for r in range(GRID_H):
            for c in range(GRID_W):
                if self.belief_map[r, c] <= 0:
                    continue

                # Tìm các ô kề có thể đi được
                neighbors = []
                for _, (dr, dc) in DIRS:
                    nr, nc = r + dr, c + dc
                    if self._walkable(nr, nc, allow_unknown=True):
                        neighbors.append((nr, nc))

                # Phân phối belief cho các ô kề (random walk)
                if len(neighbors) > 0:
                    distributed_belief = self.belief_map[r, c] * self.belief_decay / len(neighbors)
                    for nr, nc in neighbors:
                        new_belief[nr, nc] += distributed_belief

        self.belief_map = new_belief

        # Normalize
        total = np.sum(self.belief_map)
        if total > 0:
            self.belief_map /= total

    def get_belief_target(self) -> tuple:
        """
        API cho Pacman: trả về vị trí có belief cao nhất
        Dùng khi mất dấu enemy để biết nên đi tìm ở đâu
        """
        if np.sum(self.belief_map) <= 0:
            return None

        # Tìm ô có belief cao nhất
        max_belief = np.max(self.belief_map)
        if max_belief <= 0:
            return None

        # Lấy tất cả các ô có belief gần bằng max (trong trường hợp có nhiều ô)
        candidates = []
        for r in range(GRID_H):
            for c in range(GRID_W):
                if self.belief_map[r, c] >= max_belief * 0.95:  # 95% của max
                    candidates.append((r, c))

        if not candidates:
            return None

        # Chọn ô gần nhất với last_known (nếu có)
        if self.last_known_enemy_pos is not None:
            candidates.sort(key=lambda pos: manhattan(pos, self.last_known_enemy_pos))
            return candidates[0]

        return candidates[0]

    def get_danger_map(self) -> np.ndarray:
        """
        API cho Ghost: trả về danger map dựa trên belief
        Các ô có belief cao = nguy hiểm cho Ghost
        """
        return self.belief_map.copy()


# ============================
# GHOST AGENT (HIDER)
# ============================

class GhostAgent(BaseGhostAgent):
    """
    Ghost (kẻ trốn)
    Mục tiêu: tránh Pacman càng lâu càng tốt

    Chiến lược:
    - Nếu thấy Pacman -> chọn move tăng khoảng cách, ưu tiên vào fog, tránh loop, tránh dead-end
    - Nếu không thấy -> roam: ưu tiên vào fog và tránh loop
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.memory_map = np.full((GRID_H, GRID_W), -1, dtype=np.int8)

        # Đếm ghé thăm để anti-loop
        self.visit_count = np.zeros((GRID_H, GRID_W), dtype=np.int16)

        self.last_known_enemy_pos = None
        self.prev_positions = deque(maxlen=8)

        # ============================
        # BELIEF TRACKING (Tâm)
        # ============================
        # Belief map: xác suất Pacman có thể ở đâu khi không nhìn thấy
        self.belief_map = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        self.belief_decay = 0.95  # Hệ số giảm belief khi lan truyền

        seed = kwargs.get("seed", None)
        self.rng = random.Random(seed)

        self.name = "Lead Ghost (Dead-end + Anti-loop)"

    def step(self,
             map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int) -> Move:

        # 1) Update memory
        self._update_memory(map_state)

        # 2) Update visit_count
        r, c = my_position
        if in_bounds(r, c):
            self.visit_count[r, c] += 1

        # 3) Update threat memory
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        # 3b) Update belief map (Tâm's work)
        self._update_belief(map_state, my_position, enemy_position)

        self.prev_positions.append(my_position)

        threat = enemy_position or self.last_known_enemy_pos

        if threat is not None:
            mv = self._evade_move(my_position, threat)
            if mv is not None:
                return mv

        mv = self._roam_move(my_position)
        if mv is not None:
            return mv

        return Move.STAY

    # ============================
    # MEMORY / WALKABLE
    # ============================

    def _update_memory(self, obs: np.ndarray):
        visible = obs != -1
        self.memory_map[visible] = obs[visible]

    def _walkable(self, r: int, c: int) -> bool:
        """Ghost chỉ cần tránh tường; cho phép đi vào -1 để trốn."""
        return in_bounds(r, c) and self.memory_map[r, c] != 1

    def _visit_penalty(self, pos: tuple) -> int:
        r, c = pos
        if not in_bounds(r, c):
            return 999
        return int(self.visit_count[r, c])

    def _degree(self, pos: tuple) -> int:
        """
        degree(cell) = số hàng xóm không phải tường (cho phép -1).
        Dùng để tránh dead-end (degree=1).
        """
        r, c = pos
        if not in_bounds(r, c) or self.memory_map[r, c] == 1:
            return 0
        deg = 0
        for _, (dr, dc) in DIRS:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc) and self.memory_map[nr, nc] != 1:
                deg += 1
        return deg

    # ============================
    # GHOST POLICY
    # ============================

    def _evade_move(self, pos: tuple, threat: tuple) -> Move:
        """
        Chọn move tối đa hóa điểm:
          + khoảng cách tới threat
          + thưởng vào fog (-1)
          - phạt loop (prev_positions + visit_count)
          - phạt dead-end (degree=1), thưởng giao lộ (degree>=3)
        """
        best_mv = Move.STAY
        best_score = None

        moves = ALL_MOVES[:]  # không include STAY để tránh đứng yên vô ích
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
            if self.memory_map[nxt[0], nxt[1]] == -1:
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

            if best_score is None or score > best_score:
                best_score = score
                best_mv = mv

        # Nếu không có move hợp lệ, đứng yên
        return best_mv

    def _roam_move(self, pos: tuple) -> Move:
        """
        Khi không thấy threat:
        - Ưu tiên move vào fog
        - Tránh loop
        - Tránh dead-end nếu có thể
        """
        moves = self._ordered_moves(pos)
        for mv in moves:
            dr, dc = mv.value
            nxt = (pos[0] + dr, pos[1] + dc)
            if not self._walkable(nxt[0], nxt[1]):
                continue
            # tránh dead-end nếu có lựa chọn
            if self._degree(nxt) <= 1:
                continue
            return mv

        # Nếu toàn dead-end hoặc kẹt, lấy move tốt nhất còn lại
        for mv in moves:
            dr, dc = mv.value
            nxt = (pos[0] + dr, pos[1] + dc)
            if self._walkable(nxt[0], nxt[1]):
                return mv

        return None

    def _ordered_moves(self, pos: tuple):
        """Sắp xếp move theo ưu tiên: vào fog, ít loop, ít visit."""
        moves = ALL_MOVES[:]
        self.rng.shuffle(moves)

        def score(mv: Move):
            dr, dc = mv.value
            nxt = (pos[0] + dr, pos[1] + dc)

            if not self._walkable(nxt[0], nxt[1]):
                return 10_000

            penalty = 0

            if nxt in self.prev_positions:
                penalty += 10

            penalty += self._visit_penalty(nxt)

            # thưởng nếu vào fog
            if self.memory_map[nxt[0], nxt[1]] == -1:
                penalty -= 3

            # phạt nếu dead-end
            if self._degree(nxt) <= 1:
                penalty += 5

            return penalty

        moves.sort(key=score)
        return moves

    # ============================
    # BELIEF TRACKING (Tâm's work)
    # ============================

    def _update_belief(self, obs: np.ndarray, my_pos: tuple, enemy_pos: tuple):
        """
        Cập nhật belief map:
        - Nếu enemy_pos != None: belief = 1 tại vị trí đó, các ô khác = 0
        - Nếu enemy_pos == None:
            + Lan truyền belief sang các ô kề (random-walk)
            + Mask belief = 0 tại các ô Ghost quan sát mà không thấy Pacman
        """
        if enemy_pos is not None:
            # Thấy Pacman -> reset belief, chỉ tập trung vào vị trí đó
            self.belief_map.fill(0.0)
            r, c = enemy_pos
            if in_bounds(r, c):
                self.belief_map[r, c] = 1.0
        else:
            # Không thấy Pacman -> lan truyền belief
            if np.sum(self.belief_map) > 0:
                self._propagate_belief()
            else:
                # Chưa có thông tin gì -> khởi tạo belief đồng đều vào các ô không phải tường
                for r in range(GRID_H):
                    for c in range(GRID_W):
                        if self.memory_map[r, c] != 1:  # không phải tường
                            self.belief_map[r, c] = 1.0
                # Normalize
                total = np.sum(self.belief_map)
                if total > 0:
                    self.belief_map /= total

            # Mask các ô Ghost quan sát mà không thấy Pacman
            # obs != -1 là các ô nhìn thấy
            visible = obs != -1
            self.belief_map[visible] = 0.0

            # Normalize lại belief
            total = np.sum(self.belief_map)
            if total > 0:
                self.belief_map /= total

    def _propagate_belief(self):
        """
        Lan truyền belief theo mô hình random-walk:
        Mỗi ô có belief sẽ phân phối đều cho các ô kề có thể đi được
        """
        new_belief = np.zeros((GRID_H, GRID_W), dtype=np.float32)

        for r in range(GRID_H):
            for c in range(GRID_W):
                if self.belief_map[r, c] <= 0:
                    continue

                # Tìm các ô kề có thể đi được
                neighbors = []
                for _, (dr, dc) in DIRS:
                    nr, nc = r + dr, c + dc
                    if self._walkable(nr, nc):
                        neighbors.append((nr, nc))

                # Phân phối belief cho các ô kề (random walk)
                if len(neighbors) > 0:
                    distributed_belief = self.belief_map[r, c] * self.belief_decay / len(neighbors)
                    for nr, nc in neighbors:
                        new_belief[nr, nc] += distributed_belief

        self.belief_map = new_belief

        # Normalize
        total = np.sum(self.belief_map)
        if total > 0:
            self.belief_map /= total

    def get_belief_target(self) -> tuple:
        """
        API cho Ghost: trả về vị trí có belief cao nhất
        (Pacman có thể ở đâu) để Ghost tránh xa
        """
        if np.sum(self.belief_map) <= 0:
            return None

        # Tìm ô có belief cao nhất
        max_belief = np.max(self.belief_map)
        if max_belief <= 0:
            return None

        # Lấy tất cả các ô có belief gần bằng max
        candidates = []
        for r in range(GRID_H):
            for c in range(GRID_W):
                if self.belief_map[r, c] >= max_belief * 0.95:
                    candidates.append((r, c))

        if not candidates:
            return None

        # Chọn ô gần nhất với last_known (nếu có)
        if self.last_known_enemy_pos is not None:
            candidates.sort(key=lambda pos: manhattan(pos, self.last_known_enemy_pos))
            return candidates[0]

        return candidates[0]

    def get_danger_map(self) -> np.ndarray:
        """
        API cho Ghost: trả về danger map dựa trên belief
        Các ô có belief cao = nguy hiểm cho Ghost (Pacman có thể ở đó)
        """
        return self.belief_map.copy()
