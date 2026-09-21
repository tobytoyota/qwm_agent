# small MLP to predict extra penalties for bad candidate paths
# stops agent from doing dumb back-and-forth loops in 2-tile hallways or waiting when danger is near
# 56 features per path: one-hot + path length/stats + loop history + tactical flags + 26 state features

from collections import deque
import numpy as np
import torch
import torch.nn as nn

from .model import (
    ACTIONS,
    ACTION_TO_IDX,
    DIRS,
    N_ACTIONS,
    state_to_features,
    valid_action_mask,
    get_blast_zones,
    dynActivation,
    INF,
)

try:
    from .plan_overlay import walk_path
except ImportError:
    try:
        from plan_overlay import walk_path
    except ImportError:
        walk_path = None

FEATURE_DIM = 56
# magic indices extracted from the 33-dim state feature vector (don't touch these!)
STATE_SUB_INDICES = [0, 1, 2, 3, 4, 5, 16, 17, 15, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23, 29, 0]


# small 2-layer MLP with dynActivation to score candidate paths
class SmallPathRewardModel(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            dynActivation(),
            nn.Linear(hidden_dim, 32),
            dynActivation(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        if x.dim() == 3:
            b = x.shape[0]
            n = x.shape[1]
            d = x.shape[2]
            flat_x = x.reshape(-1, d)
            out_flat = self.net(flat_x)
            return out_flat.reshape(b, n)
        res = self.net(x)
        return res

    def predict_deltas(self, path_features_np, device=None):
        self.eval()
        if device is not None:
            dev = device
        else:
            dev = next(self.parameters()).device

        with torch.no_grad():
            tensor = torch.as_tensor(path_features_np, dtype=torch.float32, device=dev)
            out = self.forward(tensor)
            if out.dim() == 2 and out.size(-1) == 1:
                out = out.squeeze(-1)
            return out.cpu().numpy()


# check if agent is pacing back and forth in a loop (e.g. up-down-up-down)
def detect_cycle_helper(sequence, max_k=12):
    n = len(sequence)
    max_p = n // 2 + 1
    if max_p > max_k + 1:
        max_p = max_k + 1

    for period in range(2, max_p):
        matches = 0
        max_reps = n // period + 1
        for repeat_count in range(1, max_reps):
            chunk1 = sequence[-repeat_count * period:]
            chunk2 = sequence[-period:] * repeat_count
            if chunk1 == chunk2:
                matches = repeat_count
            else:
                break
        if matches >= 2:
            return True, period, matches
    return False, 0, 0


# BFS to find shortest escape route to any safe tile
def find_escape_moves_helper(arena, bombs, explosion_map, pos, others=()):
    x = pos[0]
    y = pos[1]
    blast_map = get_blast_zones(bombs, arena)

    in_danger = False
    if blast_map[x, y] < INF:
        in_danger = True
    elif explosion_map is not None and explosion_map[x, y] > 0:
        in_danger = True

    if in_danger == False:
        return False, []

    bomb_positions = set()
    for b in bombs:
        bomb_positions.add((b[0][0], b[0][1]))

    other_positions = set()
    for o in others:
        other_positions.add((o[0], o[1]))

    safe_tiles = set()
    for tx in range(arena.shape[0]):
        for ty in range(arena.shape[1]):
            if arena[tx, ty] == 0 and blast_map[tx, ty] == INF:
                is_fire = False
                if explosion_map is not None and explosion_map[tx, ty] > 0:
                    is_fire = True
                if is_fire == False:
                    if (tx, ty) not in bomb_positions and (tx, ty) not in other_positions:
                        safe_tiles.add((tx, ty))

    if len(safe_tiles) == 0:
        return True, []  # no safe tile exists

    queue = deque()
    visited = set()
    visited.add((x, y))
    escape_routes = []

    for action_index in range(len(DIRS)):
        dx = DIRS[action_index][1][0]
        dy = DIRS[action_index][1][1]
        nx = x + dx
        ny = y + dy
        if 0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1]:
            if arena[nx, ny] == 0:
                if (nx, ny) not in bomb_positions and (nx, ny) not in other_positions:
                    has_exp = False
                    if explosion_map is not None and explosion_map[nx, ny] > 0:
                        has_exp = True
                    if has_exp == False and blast_map[nx, ny] > 0:
                        visited.add((nx, ny))
                        if (nx, ny) in safe_tiles:
                            escape_routes.append((action_index, 1))
                        else:
                            queue.append((nx, ny, 1, action_index))

    if len(escape_routes) > 0:
        return True, escape_routes

    min_dist = INF
    while len(queue) > 0:
        node = queue.popleft()
        cx = node[0]
        cy = node[1]
        dist = node[2]
        first_act = node[3]

        if dist >= min_dist or dist >= 6:
            continue

        for dir_entry in DIRS:
            dx = dir_entry[1][0]
            dy = dir_entry[1][1]
            nx = cx + dx
            ny = cy + dy
            if 0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1]:
                if arena[nx, ny] == 0:
                    if (nx, ny) not in bomb_positions and (nx, ny) not in other_positions:
                        if (nx, ny) not in visited:
                            if blast_map[nx, ny] > dist:
                                visited.add((nx, ny))
                                if (nx, ny) in safe_tiles:
                                    min_dist = dist + 1
                                    escape_routes.append((first_act, dist + 1))
                                else:
                                    queue.append((nx, ny, dist + 1, first_act))

    return True, escape_routes


# check if bomb blast or fire is close to our position
def is_hazard_nearby_helper(pos, arena, bombs, explosion_map=None, radius=4):
    x = pos[0]
    y = pos[1]

    if explosion_map is not None:
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                tx = x + dx
                ty = y + dy
                dist = abs(dx) + abs(dy)
                if dist <= 2 and 0 <= tx < arena.shape[0] and 0 <= ty < arena.shape[1]:
                    if explosion_map[tx, ty] > 0:
                        return True

    blast_map = get_blast_zones(bombs, arena)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            tx = x + dx
            ty = y + dy
            dist = abs(dx) + abs(dy)
            if dist <= radius and 0 <= tx < arena.shape[0] and 0 <= ty < arena.shape[1]:
                if blast_map[tx, ty] < INF:
                    return True

    return False


# extract features for all 6 candidate action paths
def extract_path_candidate_features(
    game_state,
    plan_details,
    coordinate_history,
    consecutive_stationary_steps=0,
    steps_without_progress=0,
    base_scores=None,
):
    arena = game_state['field']
    bombs = game_state.get('bombs', [])
    exp_map = game_state.get('explosion_map')
    raw_others = game_state.get('others', [])
    others = []
    for o in raw_others:
        others.append(o[3])

    x = game_state['self'][3][0]
    y = game_state['self'][3][1]

    raw_feats = state_to_features(game_state, coordinate_history)
    valid_mask = valid_action_mask(raw_feats)

    in_danger, _ = find_escape_moves_helper(arena, bombs, exp_map, (x, y), tuple(others))
    hazard_nearby = is_hazard_nearby_helper((x, y), arena, bombs, exp_map, radius=4)

    crate_cnt = 0
    for rx in range(arena.shape[0]):
        for ry in range(arena.shape[1]):
            if arena[rx, ry] == 1:
                crate_cnt = crate_cnt + 1

    has_targets = False
    if crate_cnt > 0 or len(game_state.get('coins', [])) > 0 or len(raw_others) > 0:
        has_targets = True

    blast_map = get_blast_zones(bombs, arena)
    has_safe_move = False
    for a_i in range(len(DIRS)):
        dx = DIRS[a_i][1][0]
        dy = DIRS[a_i][1][1]
        nx = x + dx
        ny = y + dy
        if valid_mask[a_i] == True:
            if 0 <= nx < arena.shape[0] and 0 <= ny < arena.shape[1]:
                if blast_map[nx, ny] == INF:
                    if exp_map is None or exp_map[nx, ny] == 0:
                        has_safe_move = True
                        break

    # would placing a bomb trap us?
    hypo_bombs = []
    for b in bombs:
        hypo_bombs.append(b)
    hypo_bombs.append(((x, y), 4))
    _, bomb_escapes = find_escape_moves_helper(arena, hypo_bombs, exp_map, (x, y), tuple(others))
    bomb_traps_self = (len(bomb_escapes) == 0)

    prev_tile = None
    if len(coordinate_history) > 0:
        prev_tile = coordinate_history[-1]
    stagnation_factor = 1.0 + 0.15 * float(steps_without_progress)

    # take the 26 state features
    state_sub = []
    for s_idx in STATE_SUB_INDICES:
        state_sub.append(float(raw_feats[s_idx]))

    # simulate where each path goes
    path_data = []
    if plan_details is not None and walk_path is not None:
        for cand in plan_details:
            tiles, blocked, bomb_steps = walk_path((x, y), cand['actions'], arena=arena, bombs=bombs)
            path_data.append((cand['actions'], tiles, blocked, bomb_steps))
    elif walk_path is not None:
        for act in ACTIONS:
            tiles, blocked, bomb_steps = walk_path((x, y), [act], arena=arena, bombs=bombs)
            path_data.append(([act], tiles, blocked, bomb_steps))
    else:
        for act in ACTIONS:
            path_data.append(([act], [(x, y)], None, []))

    feature_matrix = np.zeros((N_ACTIONS, FEATURE_DIM), dtype=np.float32)

    for k in range(N_ACTIONS):
        acts = path_data[k][0]
        tiles = path_data[k][1]
        blocked = path_data[k][2]

        if len(tiles) > 1:
            target_pos = tiles[1]
        else:
            target_pos = (x, y)

        row = []

        # 1. action one-hot
        for act_i in range(6):
            if act_i == k:
                row.append(1.0)
            else:
                row.append(0.0)

        # 2. path stats
        n_tiles = len(tiles)
        if n_tiles < 1:
            n_tiles = 1

        n_acts = len(acts)
        if n_acts < 1:
            n_acts = 1

        wait_cnt = 0
        bomb_cnt = 0
        for a in acts:
            if a == 'WAIT':
                wait_cnt = wait_cnt + 1
            elif a == 'BOMB':
                bomb_cnt = bomb_cnt + 1

        if blocked is not None:
            is_bl = 1.0
            bl_ratio = float(blocked) / float(n_acts)
        else:
            is_bl = 0.0
            bl_ratio = 0.0

        dx_disp = float(tiles[-1][0] - x) / 10.0
        dy_disp = float(tiles[-1][1] - y) / 10.0

        unique_tiles = set()
        for t in tiles:
            unique_tiles.add((t[0], t[1]))
        reps = len(tiles) - len(unique_tiles)
        if reps < 0:
            reps = 0
        repeat_ratio = float(reps) / float(n_tiles)

        row.append(float(len(tiles)) / 8.0)
        row.append(float(wait_cnt) / float(n_acts))
        row.append(float(bomb_cnt) / float(n_acts))
        row.append(repeat_ratio)
        row.append(is_bl)
        row.append(bl_ratio)
        row.append(dx_disp)
        row.append(dy_disp)

        # 3. history (revisits & pacing back and forth)
        is_reversal = 0.0
        if prev_tile is not None and k < 4:
            if target_pos[0] == prev_tile[0] and target_pos[1] == prev_tile[1]:
                is_reversal = 1.0

        revisits = 0
        if k < 4:
            for p in coordinate_history:
                if p[0] == target_pos[0] and p[1] == target_pos[1]:
                    revisits = revisits + 1

        cand_traj = []
        for p in coordinate_history:
            cand_traj.append(p)
        cand_traj.append((x, y))
        cand_traj.append(target_pos)

        if k < 4:
            is_cycle, _, cycle_reps = detect_cycle_helper(cand_traj, max_k=12)
        else:
            is_cycle = False
            cycle_reps = 0

        if k == ACTION_TO_IDX['WAIT']:
            is_w = 1.0
        else:
            is_w = 0.0

        row.append(is_reversal)
        row.append(min(float(revisits) / 5.0, 2.0))
        row.append(1.0 if is_cycle == True else 0.0)
        row.append(min(float(cycle_reps) / 5.0, 2.0))
        row.append(min(float(consecutive_stationary_steps) / 5.0, 2.0))
        row.append(min(float(steps_without_progress) / 10.0, 3.0))
        row.append(stagnation_factor / 3.0)
        row.append(is_w)

        # 4. tactical flags
        if k == ACTION_TO_IDX['BOMB']:
            is_b = 1.0
        else:
            is_b = 0.0

        if valid_mask[k] == True:
            leg = 1.0
        else:
            leg = 0.0

        row.append(1.0 if in_danger == True else 0.0)
        row.append(1.0 if hazard_nearby == True else 0.0)
        row.append(1.0 if has_targets == True else 0.0)
        row.append(1.0 if has_safe_move == True else 0.0)
        row.append(1.0 if bomb_traps_self == True else 0.0)
        row.append(is_b)
        row.append(leg)

        # 5. base score
        if base_scores is not None and np.isfinite(base_scores[k]):
            base_val = float(base_scores[k]) / 10.0
        else:
            base_val = 0.0
        row.append(base_val)

        # 6. state features
        for s_val in state_sub:
            row.append(s_val)

        for col_i in range(FEATURE_DIM):
            feature_matrix[k, col_i] = row[col_i]

    return feature_matrix


def load_small_model(filepath, device=None):
    if device is not None:
        dev = device
    else:
        dev = "cpu"
    checkpoint = torch.load(filepath, map_location=dev)
    if 'input_dim' in checkpoint:
        input_dim = checkpoint['input_dim']
    else:
        input_dim = FEATURE_DIM

    model = SmallPathRewardModel(input_dim=input_dim)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(dev)
    model.eval()
    return model
