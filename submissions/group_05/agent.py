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

        self.prev_positions.append(my_position)

        # 4) Xác định mục tiêu: visible -> last_known -> explore
        target = None
        if enemy_position is not None:
            # Khi thấy Ghost, tìm điểm chặn thay vì đuổi thẳng tới enemy_position
            target = self._intercept_target(my_position, enemy_position)
        elif self.last_known_enemy_pos is not None:
            # Nếu không thấy, đuổi tới vị trí cuối cùng nhìn thấy
            target = self.last_known_enemy_pos
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
    def _is_ghost_walkable(self, r: int, c: int) -> bool:
        """
        Kiểm tra ô có đi được (không phải tường, có thể là -1) theo logic của Ghost
        khi Pacman đang dự đoán đường trốn của nó.
        """
        # Sử dụng lại _walkable với allow_unknown=True
        return self._walkable(r, c, allow_unknown=True)
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
    def _pacman_time(self, d_pacman: int) -> int:
        """
        Tính thời gian (số bước của Ghost) cần để Pacman di chuyển d_pacman ô.
        Dùng phép chia làm tròn lên (ceil) vì Pacman chỉ di chuyển theo đơn vị bước/lần.
        """
        if d_pacman <= 0:
            return 0
        # Chia làm tròn lên: ceil(d_pacman / pacman_speed)
        return (d_pacman + self.pacman_speed - 1) // self.pacman_speed

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
    # INTERCEPT LOGIC (MỚI)
    # ============================

    def _is_junction(self, pos: tuple) -> bool:
        """Kiểm tra xem một vị trí có phải là giao lộ (có 3 hướng đi trở lên) không."""
        r, c = pos
        if not in_bounds(r, c) or self.memory_map[r, c] == 1:
            return False
        
        walkable_neighbors = 0
        for _, (dr, dc) in DIRS:
            if self._walkable(r + dr, c + dc, allow_unknown=False):
                walkable_neighbors += 1
        
        # Coi 3 hướng đi trở lên là giao lộ
        return walkable_neighbors >= 3

    def _intercept_target(self, my_pos: tuple, enemy_pos: tuple) -> tuple:
        """
        Dự đoán Ghost chạy và chọn điểm chặn tối ưu bằng công thức Cost dựa trên THỜI GIAN.
        """
        max_ghost_steps = 4
        
        # 1. BFS từ Ghost để tìm các điểm đến và khoảng cách
        q = deque([enemy_pos])
        ghost_dist = {enemy_pos: 0}
        potential_targets = [] 

        while q:
            cur = q.popleft()
            d_cur = ghost_dist[cur]

            if d_cur >= max_ghost_steps:
                continue
            
            # Ghi nhận các điểm chặn tiềm năng sau bước 1
            if d_cur >= 1: 
                 potential_targets.append(cur)

            for _, (dr, dc) in DIRS:
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt in ghost_dist:
                    continue
                
                # SỬA ĐỔI 1: Đồng bộ hóa giả định đường đi của Ghost (cho phép đi qua -1)
                if not self._is_ghost_walkable(nxt[0], nxt[1]): 
                    continue
                
                ghost_dist[nxt] = d_cur + 1
                q.append(nxt)

        if not potential_targets:
            return enemy_pos

        best_target = enemy_pos
        best_score = -float('inf')

        for target in potential_targets:
            d_pacman = self._bfs_dist(my_pos, target, allow_unknown=True) 
            d_ghost = ghost_dist.get(target, -1) 
            
            if d_pacman < 0 or d_ghost < 0:
                continue
            
            # *** BƯỚC CỐT LÕI: TÍNH THỜI GIAN VÀ COST DỰA TRÊN TỐC ĐỘ PACMAN ***
            
            t_pacman = self._pacman_time(d_pacman) # Thời gian của Pacman (đã tính pacman_speed)
            t_ghost = d_ghost                     # Thời gian của Ghost (1 bước = 1 đơn vị thời gian)
            
            # --- Công thức Cost Tối ưu (Dựa trên thời gian) ---
            
            # w1 = 2: Phạt chênh lệch thời gian
            diff_penalty = 2 * abs(t_pacman - t_ghost)
            
            # w2 = 30: Hình phạt RẤT nặng nếu Pacman đến muộn
            late_penalty = 30 if t_pacman > t_ghost else 0

            # w3 = 3: Phạt khoảng cách tuyệt đối của Pacman (Ưu tiên điểm chặn gần)
            # Dùng d_pacman thay vì t_pacman để giữ cho đường đi vật lý không quá dài
            d_pacman_cost = 3 * d_pacman 

            # w4 = 6: Thưởng cho giao lộ
            junction_bonus = 6 if self._is_junction(target) else 0 

            # Cost: Càng nhỏ, càng tốt
            cost = diff_penalty + late_penalty + d_pacman_cost - junction_bonus
            score = -cost # Score: Càng lớn, càng tốt

            if score > best_score:
                best_score = score
                best_target = target
                
        # Giữ nguyên Fallback Logic (nếu điểm chặn tốt nhất quá xa và score tệ, quay về đuổi thẳng)
        # Ngưỡng -35 tương đương với việc Pacman đến muộn 2 bước (late_penalty=30 + 2*diff=4)
        if best_score < -35 and self._bfs_dist(my_pos, best_target, allow_unknown=True) > 6:
             return enemy_pos

        return best_target

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
          - [Nguyên] phạt danger (gần Pacman)
          - [Nguyên] phạt occlusion (cùng hàng/cột, không có tường che)
        """
        # Nguyên: Tunable weights
        W_DANGER = 5
        W_OCCLUDE = 20
        DANGER_RADIUS = 5
        
        # Nguyên: Helper - check line of sight (occlusion)
        def has_line_of_sight(pos1: tuple, pos2: tuple) -> bool:
            """Kiểm tra xem pos1 và pos2 có cùng hàng/cột trong phạm vi 5 và không có tường che."""
            r1, c1 = pos1
            r2, c2 = pos2
            
            # Phải cùng hàng hoặc cùng cột
            if r1 != r2 and c1 != c2:
                return False
            
            # Kiểm tra khoảng cách Manhattan <= 5
            dist = abs(r1 - r2) + abs(c1 - c2)
            if dist > 5:
                return False
            
            # Kiểm tra tường giữa hai vị trí
            if r1 == r2:  # Cùng hàng
                c_min, c_max = (c1, c2) if c1 < c2 else (c2, c1)
                for c in range(c_min + 1, c_max):
                    if self.memory_map[r1, c] == 1:  # Tường chặn
                        return False
            else:  # Cùng cột (c1 == c2)
                r_min, r_max = (r1, r2) if r1 < r2 else (r2, r1)
                for r in range(r_min + 1, r_max):
                    if self.memory_map[r, c1] == 1:  # Tường chặn
                        return False
            
            return True
        
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
            
            # Nguyên: 5) Danger-aware - phạt các ô gần Pacman
            dist_to_threat = manhattan(nxt, threat)
            if dist_to_threat <= DANGER_RADIUS:
                danger_penalty = (DANGER_RADIUS + 1 - dist_to_threat)
                score -= W_DANGER * danger_penalty
            
            # Nguyên: 6) Occlusion - phạt nếu cùng hàng/cột với Pacman (dễ bị nhìn thấy)
            if has_line_of_sight(nxt, threat):
                score -= W_OCCLUDE

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
        """
        Sắp xếp move theo ưu tiên: vào fog, ít loop, ít visit.
        [Nguyên] Thêm: tránh danger và occlusion.
        """
        # Nguyên: Tunable weights
        W_DANGER = 5
        W_OCCLUDE = 20
        DANGER_RADIUS = 5
        
        # Nguyên: Helper - check line of sight (occlusion)
        def has_line_of_sight(pos1: tuple, pos2: tuple) -> bool:
            """Kiểm tra xem pos1 và pos2 có cùng hàng/cột trong phạm vi 5 và không có tường che."""
            r1, c1 = pos1
            r2, c2 = pos2
            
            # Phải cùng hàng hoặc cùng cột
            if r1 != r2 and c1 != c2:
                return False
            
            # Kiểm tra khoảng cách Manhattan <= 5
            dist = abs(r1 - r2) + abs(c1 - c2)
            if dist > 5:
                return False
            
            # Kiểm tra tường giữa hai vị trí
            if r1 == r2:  # Cùng hàng
                c_min, c_max = (c1, c2) if c1 < c2 else (c2, c1)
                for c in range(c_min + 1, c_max):
                    if self.memory_map[r1, c] == 1:  # Tường chặn
                        return False
            else:  # Cùng cột (c1 == c2)
                r_min, r_max = (r1, r2) if r1 < r2 else (r2, r1)
                for r in range(r_min + 1, r_max):
                    if self.memory_map[r, c1] == 1:  # Tường chặn
                        return False
            
            return True
        
        # Lấy vị trí threat để tính danger/occlusion
        threat = self.last_known_enemy_pos
        
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
            
            # Nguyên: Danger-aware - phạt các ô gần Pacman
            if threat is not None:
                dist_to_threat = manhattan(nxt, threat)
                if dist_to_threat <= DANGER_RADIUS:
                    danger_penalty = (DANGER_RADIUS + 1 - dist_to_threat)
                    penalty += W_DANGER * danger_penalty
            
            # Nguyên: Occlusion - phạt nếu cùng hàng/cột với Pacman (dễ bị nhìn thấy)
            if threat is not None and has_line_of_sight(nxt, threat):
                penalty += W_OCCLUDE

            return penalty

        moves.sort(key=score)
        return moves
