# Ghost Strategy (Nguyên) - Implementation Changes

**Date:** December 20, 2025  
**File Modified:** `submissions/group_05/agent.py`  
**Author:** Nguyên

---

## Overview

Implemented danger-aware movement and occlusion avoidance strategy for Ghost agent to improve survival rate against Pacman. Changes are localized to two scoring functions only.

---

## Changes Made

### 1. Enhanced `_evade_move()` Function (Lines 745-799 → 745-837)

**Purpose:** When Ghost sees Pacman, choose the best evasion move considering danger zones and visibility.

**Added Features:**

#### A. Tunable Weights
```python
W_DANGER = 5           # Weight for danger penalty
W_OCCLUDE = 20         # Weight for occlusion penalty  
DANGER_RADIUS = 5      # Danger zone radius
```

#### B. Line-of-Sight Helper (Nested Function)
```python
def has_line_of_sight(pos1: tuple, pos2: tuple) -> bool
```
- Checks if two positions share same row/column within range 5
- Scans for walls between positions (O(5) complexity)
- Returns `True` if Ghost would be visible to Pacman (dangerous)

#### C. Danger-Aware Scoring
```python
# Nguyên: danger-aware
dist_to_threat = manhattan(nxt, threat)
if dist_to_threat <= DANGER_RADIUS:
    danger_penalty = (DANGER_RADIUS + 1 - dist_to_threat)
    score -= W_DANGER * danger_penalty
```
- Penalizes moves that bring Ghost closer to Pacman
- Penalty increases as distance decreases
- Only applies within DANGER_RADIUS (5 tiles)

#### D. Occlusion Avoidance
```python
# Nguyên: occlusion
if has_line_of_sight(nxt, threat):
    score -= W_OCCLUDE
```
- Heavy penalty (-20) for moves that expose Ghost on same row/column
- Considers Pacman's cross-shaped vision (radius 5, wall-blocked)
- Encourages Ghost to hide behind walls and avoid straight lines

**Scoring Formula (New):**
```
total_score = base_distance_score (3×d)
            + fog_bonus (+4)
            - loop_penalty (-15)
            - visit_penalty (-2×v)
            - dead_end_penalty (-25)
            + junction_bonus (+6)
            - danger_penalty (-5×(6-d))  [NEW - Nguyên]
            - occlusion_penalty (-20)    [NEW - Nguyên]
```

---

### 2. Enhanced `_ordered_moves()` Function (Lines 824-856 → 824-906)

**Purpose:** When Ghost doesn't see Pacman (roaming), order moves by safety considering belief-based threat estimation.

**Added Features:**

#### A. Same Tunable Weights
```python
W_DANGER = 5
W_OCCLUDE = 20
DANGER_RADIUS = 5
```

#### B. Same Line-of-Sight Helper (Duplicated Nested)
- Identical implementation as in `_evade_move()`
- Necessary because it's a local helper function

#### C. Threat Position Resolution
```python
threat = self.last_known_enemy_pos
if threat is None:
    threat = self.get_belief_target()  # Fallback to belief
```
- Uses last known Pacman position if available
- Falls back to belief system (Tâm's work) for probabilistic threat location
- Enables danger-aware scoring even when Pacman is not visible

#### D. Danger-Aware & Occlusion in Penalty Function
```python
# Nguyên: danger-aware
if threat is not None:
    dist_to_threat = manhattan(nxt, threat)
    if dist_to_threat <= DANGER_RADIUS:
        danger_penalty = (DANGER_RADIUS + 1 - dist_to_threat)
        penalty += W_DANGER * danger_penalty

# Nguyên: occlusion
if threat is not None and has_line_of_sight(nxt, threat):
    penalty += W_OCCLUDE
```

**Penalty Formula (New):**
```
total_penalty = loop_penalty (+10)
              + visit_penalty (+v)
              - fog_bonus (-3)
              + dead_end_penalty (+5)
              + danger_penalty (+5×(6-d))  [NEW - Nguyên]
              + occlusion_penalty (+20)    [NEW - Nguyên]
```
*(Lower penalty = better move)*

---

## Implementation Constraints Met

✅ **Only modified** `_evade_move()` and `_ordered_moves()`  
✅ **Did NOT modify:** `step()`, BFS, memory_map, frontier scoring, visit_count, degree logic, imports, class structure  
✅ **No global functions added** - only nested local helpers  
✅ **No new algorithms** - only heuristic scoring adjustments  
✅ **Complexity:** O(5) per move evaluation (line-of-sight check)  
✅ **Preserved all existing logic:** Dead-end avoidance, loop prevention, fog preference unchanged  

---

## Strategy Explanation

### 1. Danger-Aware Movement
Ghost now evaluates how close each potential move brings it to Pacman:
- **Far from Pacman (>5 tiles):** No danger penalty, normal behavior
- **Close to Pacman (≤5 tiles):** Progressive penalty, strongest when adjacent
- **Example:** At distance 2, danger penalty = 5×(6-2) = 20 points

### 2. Occlusion Strategy
Ghost avoids being spotted by staying off Pacman's cross-shaped vision:
- **Vision model:** Cross pattern (±5 in 4 directions), blocked by walls
- **High risk:** Same row/column as Pacman with clear line-of-sight (-20 penalty)
- **Safe zones:** Behind walls, diagonal positions, or far away
- **Example:** Ghost at (10,5), Pacman at (10,12), no walls between → heavy penalty

### 3. Integration with Existing Systems
- **With Tâm's belief system:** Uses probabilistic Pacman location when not visible
- **With fog preference:** Both systems coexist, Ghost balances exploration and safety
- **With dead-end avoidance:** All penalties stack, creating nuanced decision-making

---

## Tuning Recommendations

Current weights are starting values for testing:

| Weight | Value | Effect | Tuning Guidance |
|--------|-------|--------|-----------------|
| `W_DANGER` | 5 | Danger penalty strength | Increase if Ghost too aggressive, decrease if too cautious |
| `W_OCCLUDE` | 20 | Occlusion penalty strength | Increase if Ghost gets spotted often, decrease if too hiding |
| `DANGER_RADIUS` | 5 | Danger zone size | Match Pacman's vision radius, increase for more caution |

**Testing scenarios:**
1. Ghost survival time vs baseline
2. Ghost getting caught frequency
3. Balance between hiding and exploration

---

## Performance Impact

- **Added operations per move:** 1 LOS check (O(5)) + 1 distance check (O(1))
- **Total complexity per decision:** O(4 moves × 5 LOS) = O(20) worst case
- **Expected overhead:** <0.1ms per decision, negligible for game loop
- **Memory impact:** None (local variables only)

---

## Testing Checklist

- [ ] Ghost successfully evades when Pacman approaches
- [ ] Ghost avoids standing in same row/column as Pacman
- [ ] Ghost still explores fog areas when safe
- [ ] Ghost doesn't get stuck in loops
- [ ] Ghost avoids dead-ends
- [ ] No runtime errors or exceptions
- [ ] Game completes without timeout
- [ ] Improved survival rate vs baseline

---

## Code Quality

- **Comments:** Clear markers with "Nguyên: danger-aware" and "Nguyên: occlusion"
- **Readability:** Helper function isolated, weights clearly defined
- **Maintainability:** Easy to tune weights, clear logic flow
- **Compatibility:** No breaking changes to existing codebase

---

## Future Improvements (Not Implemented)

Potential enhancements if more complex changes allowed:
- Predictive Pacman movement based on velocity
- Multi-step lookahead for trap avoidance
- Adaptive weights based on game phase
- Cooperative strategies for multiple ghosts

---

## Summary

Successfully implemented Ghost Strategy (Nguyên) with:
- **Danger awareness** to avoid Pacman's reach
- **Occlusion avoidance** to stay hidden from cross-shaped vision
- **Minimal invasiveness** - only 2 functions modified
- **Performance efficiency** - O(5) overhead per move
- **Full compatibility** with existing systems

The Ghost agent now makes more intelligent survival decisions while maintaining all existing behaviors and constraints.
