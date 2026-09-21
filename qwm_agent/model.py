# bomberman spatial qwm agent
# dueling cnn dqn + latent world model for lookahead
# based on QWM paper (Dong et al. 2026)
# TODO: clean up code before submitting!!

from collections import deque
from pathlib import Path
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

AGENT_DIR = Path(__file__).resolve().parent

# ===========================================================================
# Board settings & constants (17x17 grid)
# ===========================================================================
BOARD_W = 17
BOARD_H = 17
COLS = 17  # redundant but some code uses cols/rows
ROWS = 17
BOMB_POWER = 3
BOMB_TIMER = 4
EXPLOSION_TIMER = 2
INF = 999  # large number for unreachable stuff
HORIZON = BOMB_TIMER + EXPLOSION_TIMER + 1  # 4 + 2 + 1 = 7 steps lookahead

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
N_ACTIONS = 6
MOVE_ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']
DIR_VECS = {'UP': (0, -1), 'RIGHT': (1, 0), 'DOWN': (0, 1), 'LEFT': (-1, 0)}
DIRS = [(a, DIR_VECS[a]) for a in MOVE_ACTIONS]
DIR_VECS_LIST = [(0, -1), (1, 0), (0, 1), (-1, 0)]
ACTION_TO_IDX = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'WAIT': 4, 'BOMB': 5}

N_SPATIAL_CHANNELS = 10
N_FEATURES = 33

# indices for the 33-dim features array (lots of magic numbers here...)
F_SAFE_MOVE = 0
F_SAFE_WAIT = 4
F_IN_DANGER = 5
F_COIN_DIR = 6
F_COIN_PROX = 10
F_CRATE_DIR = 11
F_BOMB_UTIL = 15
F_TRAPS_SELF = 16
F_BOMBS_LEFT = 17
F_OPP_DIR = 18
F_OPP_PROX = 22
F_BLOCKED = 23
F_VISITED = 27
F_BIAS = 28
F_NBR_VISITED = 29


def get_device(explicit_device=None):
    if explicit_device is not None:
        return torch.device(explicit_device)
    if torch.cuda.is_available() == True:
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def agent_path(*parts):
    res = AGENT_DIR
    for p in parts:
        res = res / p
    return str(res)


# ---------------------------------------------------------------------------
# helper functions for board physics & feature extraction
# ---------------------------------------------------------------------------

# helper to calculate tiles hit by a bomb at pos (stops at stone walls)
def blast_coords(pos, field, power=BOMB_POWER):
    # print("blast_coords called for", pos)
    x = pos[0]
    y = pos[1]
    coords = []
    coords.append((x, y))

    for d in DIR_VECS_LIST:
        dx = d[0]
        dy = d[1]
        for i in range(1, power + 1):
            nx = x + dx * i
            ny = y + dy * i
            # check if outside grid bounds
            if nx < 0 or nx >= field.shape[0]:
                break
            if ny < 0 or ny >= field.shape[1]:
                break
            if field[nx, ny] == -1:  # wall blocks explosion!
                break
            coords.append((nx, ny))

    return coords


# blast map: min countdown of any bomb hitting tile (x, y)
def get_blast_zones(bombs, field):
    blast_map = np.full(field.shape, INF, dtype=np.int32)
    for b in bombs:
        bx = b[0][0]
        by = b[0][1]
        t = b[1]
        # get tiles hit by this bomb
        c_list = blast_coords((bx, by), field)
        for coord in c_list:
            cx = coord[0]
            cy = coord[1]
            if t < blast_map[cx, cy]:
                blast_map[cx, cy] = t
    return blast_map


# 3D bool array: danger[t, x, y] = True if tile explodes at step t
def danger_tensor(bombs, field, explosion_map=None, horizon=HORIZON):
    danger = np.zeros((horizon + 1, field.shape[0], field.shape[1]), dtype=bool)

    for b in bombs:
        bx = b[0][0]
        by = b[0][1]
        t = b[1]
        k0 = t + 1
        k1 = t + EXPLOSION_TIMER
        if k1 > horizon:
            k1 = horizon

        if k0 <= horizon:
            b_coords = blast_coords((bx, by), field)
            for c in b_coords:
                cx = c[0]
                cy = c[1]
                for step in range(k0, k1 + 1):
                    danger[step, cx, cy] = True

    if explosion_map is not None:
        for x in range(field.shape[0]):
            for y in range(field.shape[1]):
                if explosion_map[x, y] > 0:
                    e = int(explosion_map[x, y])
                    if e > horizon:
                        e = horizon
                    for step in range(1, e + 1):
                        danger[step, x, y] = True

    return danger


# space-time BFS to check if we can run away from explosions
# bitmask magic: (1 << i) tracks which first move was used to reach safety
def bfs_safe_escape(pos, field, bombs, explosion_map, others=(), horizon=HORIZON):
    danger = danger_tensor(bombs, field, explosion_map, horizon)
    passable = np.zeros(field.shape, dtype=bool)
    for x in range(field.shape[0]):
        for y in range(field.shape[1]):
            if field[x, y] == 0:
                passable[x, y] = True

    for b in bombs:
        bx = b[0][0]
        by = b[0][1]
        passable[bx, by] = False

    x0 = pos[0]
    y0 = pos[1]

    opp_set = set()
    for o in others:
        opp_set.add((o[0], o[1]))

    reach = np.zeros((horizon + 1, field.shape[0], field.shape[1]), dtype=np.int16)
    for i in range(len(DIRS)):
        dx = DIRS[i][1][0]
        dy = DIRS[i][1][1]
        nx = x0 + dx
        ny = y0 + dy
        if passable[nx, ny] == True:
            if (nx, ny) not in opp_set:
                if danger[1, nx, ny] == False:
                    reach[1, nx, ny] = reach[1, nx, ny] | (1 << i)

    if danger[1, x0, y0] == False:
        reach[1, x0, y0] = reach[1, x0, y0] | (1 << 4)  # 4 = WAIT

    future_danger = np.flip(np.cumsum(np.flip(danger, 0), axis=0), 0) > 0
    if future_danger[1, x0, y0] == False:
        min_escape = 0
    else:
        min_escape = INF

    for k in range(1, horizon):
        xs, ys = np.nonzero(reach[k])
        if len(xs) == 0:
            break
        for j in range(len(xs)):
            x = xs[j]
            y = ys[j]
            mask = reach[k, x, y]
            if future_danger[k, x, y] == False and k < min_escape:
                min_escape = k
            if danger[k + 1, x, y] == False:
                reach[k + 1, x, y] = reach[k + 1, x, y] | mask
            for dir_tuple in DIRS:
                dx = dir_tuple[1][0]
                dy = dir_tuple[1][1]
                nx = x + dx
                ny = y + dy
                if passable[nx, ny] == True and danger[k + 1, nx, ny] == False:
                    reach[k + 1, nx, ny] = reach[k + 1, nx, ny] | mask

    final = 0
    xs, ys = np.nonzero(reach[horizon])
    for j in range(len(xs)):
        x = xs[j]
        y = ys[j]
        final = final | int(reach[horizon, x, y])
        if future_danger[horizon, x, y] == False and horizon < min_escape:
            min_escape = horizon

    safe_actions = {}
    for i in range(len(MOVE_ACTIONS)):
        a = MOVE_ACTIONS[i]
        if (final & (1 << i)) != 0:
            safe_actions[a] = True
        else:
            safe_actions[a] = False

    if (final & (1 << 4)) != 0:
        safe_actions['WAIT'] = True
    else:
        safe_actions['WAIT'] = False

    has_any = False
    for v in safe_actions.values():
        if v == True:
            has_any = True
            break
    if has_any == False:
        min_escape = INF

    return safe_actions, min_escape


# simple BFS queue to find direction to nearest target (coin, crate, etc.)
def bfs_direction(pos, field, targets, obstacles=None):
    if targets is None or len(targets) == 0:
        return None, INF

    target_set = set()
    for t in targets:
        target_set.add((t[0], t[1]))

    if (pos[0], pos[1]) in target_set:
        return 'WAIT', 0

    passable = (field == 0)
    if obstacles is not None:
        for ob in obstacles:
            ox = ob[0]
            oy = ob[1]
            if 0 <= ox < field.shape[0] and 0 <= oy < field.shape[1]:
                passable[ox, oy] = False

    dist = np.full(field.shape, -1, dtype=np.int32)
    dist[pos[0], pos[1]] = 0
    first = {}
    q = deque()

    for dir_entry in DIRS:
        a = dir_entry[0]
        dx = dir_entry[1][0]
        dy = dir_entry[1][1]
        nx = pos[0] + dx
        ny = pos[1] + dy
        if (nx, ny) in target_set:
            return a, 1
        if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
            if passable[nx, ny] == True and dist[nx, ny] < 0:
                dist[nx, ny] = 1
                first[(nx, ny)] = a
                q.append((nx, ny))

    while len(q) > 0:
        node = q.popleft()
        x = node[0]
        y = node[1]
        d = dist[x, y]
        for dir_entry in DIRS:
            dx = dir_entry[1][0]
            dy = dir_entry[1][1]
            nx = x + dx
            ny = y + dy
            if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
                if dist[nx, ny] >= 0:
                    continue
                if (nx, ny) in target_set:
                    return first[(x, y)], d + 1
                if passable[nx, ny] == True:
                    dist[nx, ny] = d + 1
                    first[(nx, ny)] = first[(x, y)]
                    q.append((nx, ny))

    return None, INF


# find walkable tiles next to soft crates so we know where to place bomb
def crate_adjacent_tiles(field):
    res = []
    w = field.shape[0]
    h = field.shape[1]
    for x in range(w):
        for y in range(h):
            if field[x, y] == 0:  # must be free tile
                is_adj = False
                for d in DIR_VECS_LIST:
                    nx = x + d[0]
                    ny = y + d[1]
                    if 0 <= nx < w and 0 <= ny < h:
                        if field[nx, ny] == 1:  # crate!
                            is_adj = True
                            break
                if is_adj == True:
                    res.append((x, y))
    return res


# check if placing bomb here hits crates/enemies and doesnt trap us
def bomb_utility(pos, field, others, coins, bombs=(), explosion_map=None):
    coords = blast_coords(pos, field)
    crates_hit = 0
    for c in coords:
        if field[c[0], c[1]] == 1:
            crates_hit = crates_hit + 1

    opponents_hit = 0
    for o in others:
        for c in coords:
            if o[0] == c[0] and o[1] == c[1]:
                opponents_hit = opponents_hit + 1
                break

    hypo_bombs = []
    for b in bombs:
        hypo_bombs.append(b)
    hypo_bombs.append(((pos[0], pos[1]), BOMB_TIMER))

    safe, _ = bfs_safe_escape(pos, field, hypo_bombs, explosion_map, others=others)
    traps = (safe['WAIT'] == False)

    res = {}
    res['crates_hit'] = crates_hit
    res['opponents_hit'] = opponents_hit
    res['coins_uncovered'] = 0
    res['traps_self'] = traps
    return res


# turns game state dict into 33 features for the network
def state_to_features(game_state, coordinate_history=None):
    if game_state is None:
        return None
    if 'self' not in game_state:
        return None

    f = np.zeros(N_FEATURES, dtype=np.float32)
    field = game_state['field']
    bombs = game_state.get('bombs', [])
    explosion_map = game_state.get('explosion_map')
    my_info = game_state['self']
    bombs_left = my_info[2]
    x = my_info[3][0]
    y = my_info[3][1]

    others = []
    for o in game_state.get('others', []):
        others.append(o[3])

    coins = game_state.get('coins', [])

    bomb_xy = set()
    for b in bombs:
        bomb_xy.add(b[0])

    opp_set = set()
    for o in others:
        opp_set.add(o)

    # 1. safe moves from BFS
    safe, _ = bfs_safe_escape((x, y), field, bombs, explosion_map, others)
    for i in range(len(MOVE_ACTIONS)):
        a = MOVE_ACTIONS[i]
        if safe[a] == True:
            f[F_SAFE_MOVE + i] = 1.0
    if safe['WAIT'] == True:
        f[F_SAFE_WAIT] = 1.0

    blast_map = get_blast_zones(bombs, field)
    in_danger = False
    if blast_map[x, y] < INF:
        in_danger = True
    elif explosion_map is not None and explosion_map[x, y] > 0:
        in_danger = True
    if in_danger == True:
        f[F_IN_DANGER] = 1.0

    burning = set()
    if explosion_map is not None:
        for bx in range(field.shape[0]):
            for by in range(field.shape[1]):
                if explosion_map[bx, by] > 0:
                    burning.add((bx, by))

    path_obstacles = set()
    for pt in bomb_xy:
        path_obstacles.add(pt)
    for pt in burning:
        path_obstacles.add(pt)
    for pt in opp_set:
        path_obstacles.add(pt)

    # 2. coins
    d, dist = bfs_direction((x, y), field, coins, path_obstacles)
    if d == 'UP':
        f[F_COIN_DIR + 0] = 1.0
    elif d == 'RIGHT':
        f[F_COIN_DIR + 1] = 1.0
    elif d == 'DOWN':
        f[F_COIN_DIR + 2] = 1.0
    elif d == 'LEFT':
        f[F_COIN_DIR + 3] = 1.0

    if dist < INF:
        f[F_COIN_PROX] = 1.0 / (1.0 + float(dist))

    # 3. crates & bomb utility
    cr_tiles = crate_adjacent_tiles(field)
    d, dist = bfs_direction((x, y), field, cr_tiles, path_obstacles)
    if d == 'UP':
        f[F_CRATE_DIR + 0] = 1.0
    elif d == 'RIGHT':
        f[F_CRATE_DIR + 1] = 1.0
    elif d == 'DOWN':
        f[F_CRATE_DIR + 2] = 1.0
    elif d == 'LEFT':
        f[F_CRATE_DIR + 3] = 1.0

    util = bomb_utility((x, y), field, others, coins, bombs, explosion_map)
    if util['traps_self'] == False:
        score_val = (util['crates_hit'] + 3.0 * util['opponents_hit']) / 6.0
        if score_val > 1.0:
            score_val = 1.0
        f[F_BOMB_UTIL] = score_val

    # 4. traps self & bombs left
    if util['traps_self'] == True:
        f[F_TRAPS_SELF] = 1.0
    if bombs_left:
        f[F_BOMBS_LEFT] = 1.0

    # 5. opponents
    d, dist = bfs_direction((x, y), field, others, bomb_xy | burning)
    if d == 'UP':
        f[F_OPP_DIR + 0] = 1.0
    elif d == 'RIGHT':
        f[F_OPP_DIR + 1] = 1.0
    elif d == 'DOWN':
        f[F_OPP_DIR + 2] = 1.0
    elif d == 'LEFT':
        f[F_OPP_DIR + 3] = 1.0

    if dist < INF:
        f[F_OPP_PROX] = 1.0 / (1.0 + float(dist))

    # 6. blocked neighbors
    for i in range(len(DIRS)):
        dx = DIRS[i][1][0]
        dy = DIRS[i][1][1]
        nx = x + dx
        ny = y + dy
        if field[nx, ny] != 0:
            f[F_BLOCKED + i] = 1.0
        elif (nx, ny) in bomb_xy:
            f[F_BLOCKED + i] = 1.0
        elif (nx, ny) in opp_set:
            f[F_BLOCKED + i] = 1.0

    # 7. loop history (so we don't pace back and forth forever)
    if coordinate_history is not None:
        cnt = 0
        for p in coordinate_history:
            if p[0] == x and p[1] == y:
                cnt = cnt + 1
        if cnt > 2:
            f[F_VISITED] = 1.0

        for i in range(len(DIRS)):
            dx = DIRS[i][1][0]
            dy = DIRS[i][1][1]
            target_pt = (x + dx, y + dy)
            nbr_cnt = 0
            for p in coordinate_history:
                if p[0] == target_pt[0] and p[1] == target_pt[1]:
                    nbr_cnt = nbr_cnt + 1
            if nbr_cnt >= 2:
                f[F_NBR_VISITED + i] = 1.0

    f[F_BIAS] = 1.0
    return f


# mask out illegal actions (blocked moves or bomb when no ammo)
def valid_action_mask(features):
    mask = [True, True, True, True, True, True]

    if features[F_BLOCKED] >= 0.5:
        mask[0] = False
    if features[F_BLOCKED + 1] >= 0.5:
        mask[1] = False
    if features[F_BLOCKED + 2] >= 0.5:
        mask[2] = False
    if features[F_BLOCKED + 3] >= 0.5:
        mask[3] = False
    if features[F_BOMBS_LEFT] <= 0.5:
        mask[5] = False

    return np.array(mask, dtype=bool)


# creates 10x17x17 spatial tensor for CNN
# channel 0: walls, 1: crates, 2: free, 3: bomb danger, 4: fire, 5: coins, 6: self, 7: others, 8: visited heatmap, 9: target dist
def state_to_spatial_tensor(game_state, coordinate_history=None):
    spatial = np.zeros((N_SPATIAL_CHANNELS, BOARD_W, BOARD_H), dtype=np.float32)
    if game_state is None:
        return spatial
    if 'field' not in game_state:
        return spatial

    field = game_state['field']
    W = field.shape[0]
    H = field.shape[1]
    bombs = game_state.get('bombs', [])
    explosion_map = game_state.get('explosion_map')
    coins = game_state.get('coins', [])
    others = game_state.get('others', [])
    self_info = game_state.get('self')

    # walls, crates, empty - simple grid loops
    for r in range(W):
        for c in range(H):
            if field[r, c] == -1:
                spatial[0, r, c] = 1.0
            elif field[r, c] == 1:
                spatial[1, r, c] = 1.0
            elif field[r, c] == 0:
                spatial[2, r, c] = 1.0

    # bombs danger
    for b in bombs:
        bx = b[0][0]
        by = b[0][1]
        timer = b[1]
        if 0 <= bx < W and 0 <= by < H:
            danger_val = (5.0 - float(timer)) / 5.0
            if danger_val > spatial[3, bx, by]:
                spatial[3, bx, by] = danger_val
            spatial[2, bx, by] = 0.0
            for d in DIR_VECS_LIST:
                dx = d[0]
                dy = d[1]
                for dist in range(1, 4):
                    nx = bx + dx * dist
                    ny = by + dy * dist
                    if nx < 0 or nx >= W or ny < 0 or ny >= H:
                        break
                    if field[nx, ny] == -1:
                        break
                    if danger_val > spatial[3, nx, ny]:
                        spatial[3, nx, ny] = danger_val
                    if field[nx, ny] == 1:
                        break

    # fire
    if explosion_map is not None:
        for r in range(W):
            for c in range(H):
                if explosion_map[r, c] > 0:
                    v = float(explosion_map[r, c]) / 2.0
                    if v > 1.0:
                        v = 1.0
                    spatial[4, r, c] = v

    # coins
    for coin in coins:
        cx = coin[0]
        cy = coin[1]
        if 0 <= cx < W and 0 <= cy < H:
            spatial[5, cx, cy] = 1.0

    # self
    self_x = 1
    self_y = 1
    if self_info is not None:
        if len(self_info) >= 4:
            self_x = self_info[3][0]
            self_y = self_info[3][1]
            if 0 <= self_x < W and 0 <= self_y < H:
                spatial[6, self_x, self_y] = 1.0

    # enemies
    for other in others:
        if other is not None:
            if len(other) >= 4:
                ox = other[3][0]
                oy = other[3][1]
                if 0 <= ox < W and 0 <= oy < H:
                    spatial[7, ox, oy] = 1.0
                    spatial[2, ox, oy] = 0.0

    # position history heatmap (visited tiles)
    if coordinate_history is not None:
        for p in coordinate_history:
            if isinstance(p, (tuple, list)):
                if len(p) >= 2:
                    px = p[0]
                    py = p[1]
                    if 0 <= px < W and 0 <= py < H:
                        val = spatial[8, px, py] + 0.25
                        if val > 1.0:
                            val = 1.0
                        spatial[8, px, py] = val

    # goal gradient (BFS to coins or crates)
    passable = (field == 0)
    for b in bombs:
        bx = b[0][0]
        by = b[0][1]
        if 0 <= bx < W and 0 <= by < H:
            passable[bx, by] = False
    if explosion_map is not None:
        passable &= (explosion_map == 0)
    for other in others:
        if other is not None and len(other) >= 4:
            ox = other[3][0]
            oy = other[3][1]
            if 0 <= ox < W and 0 <= oy < H:
                passable[ox, oy] = False

    if len(coins) > 0:
        targets = set()
        for c in coins:
            targets.add((c[0], c[1]))
    else:
        cr_list = crate_adjacent_tiles(field)
        targets = set()
        for cr in cr_list:
            targets.add((cr[0], cr[1]))

    if len(targets) > 0:
        if 0 <= self_x < W and 0 <= self_y < H:
            dist_map = np.full((W, H), -1, dtype=np.int32)
            q = deque()
            q.append((self_x, self_y))
            dist_map[self_x, self_y] = 0
            while len(q) > 0:
                cur = q.popleft()
                cx = cur[0]
                cy = cur[1]
                d = dist_map[cx, cy]
                for d_vec in DIR_VECS_LIST:
                    nx = cx + d_vec[0]
                    ny = cy + d_vec[1]
                    if 0 <= nx < W and 0 <= ny < H:
                        if dist_map[nx, ny] < 0 and passable[nx, ny] == True:
                            dist_map[nx, ny] = d + 1
                            q.append((nx, ny))
            for t in targets:
                tx = t[0]
                ty = t[1]
                if 0 <= tx < W and 0 <= ty < H:
                    if dist_map[tx, ty] >= 0:
                        spatial[9, tx, ty] = 1.0 / (1.0 + float(dist_map[tx, ty]))

    return spatial


# ---------------------------------------------------------------------------
# neural net architectures
# ---------------------------------------------------------------------------

# dynActivation: mish with learnable alpha and beta (from paper eq 4)
class DynActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        diff = self.alpha - self.beta
        m = F.mish(x)
        term1 = m * diff
        term2 = self.beta * x
        return term1 + term2


# aliases so both spellings work
dynActivation = DynActivation
dynActvation = DynActivation


# loads weights into model, ignores mismatches if partial=True
def load_matching_weights(model, state_dict, partial=True):
    model_dict = model.state_dict()
    matched = {}
    loaded = []
    uninit = []
    for k in model_dict.keys():
        if k in state_dict:
            val = state_dict[k]
            if val.shape == model_dict[k].shape:
                matched[k] = val
                loaded.append(k)
            else:
                matched[k] = model_dict[k]
                uninit.append(k)
        else:
            matched[k] = model_dict[k]
            uninit.append(k)
    if partial == True:
        model.load_state_dict(matched)
    else:
        model.load_state_dict(state_dict)
    return loaded, uninit


# dueling cnn dqn
class SpatialDuelingDQN(nn.Module):
    def __init__(self, in_channels=N_SPATIAL_CHANNELS, n_features=N_FEATURES, n_actions=N_ACTIONS, fused_dim=256):
        super().__init__()
        self.fused_dim = fused_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(),
        )
        self.spatial_proj = nn.Sequential(
            nn.Linear(32 * BOARD_W * BOARD_H, 256),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(256 + n_features, fused_dim),
            nn.ReLU(),
        )
        self.val_stream = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def encode_fused(self, spatial, features):
        b = spatial.size(0)
        c = self.conv(spatial).view(b, -1)
        s_embed = self.spatial_proj(c)
        cat_feat = torch.cat([s_embed, features], dim=1)
        res = self.fusion(cat_feat)
        return res

    def forward_from_fused(self, fused):
        val = self.val_stream(fused)
        adv = self.adv_stream(fused)
        adv_mean = adv.mean(dim=-1, keepdim=True)
        adv_centered = adv - adv_mean
        q_vals = val + adv_centered
        return q_vals

    def forward(self, spatial, features):
        fused = self.encode_fused(spatial, features)
        out = self.forward_from_fused(fused)
        return out


# latent world model: predicts next latent z (residual), reward, value, and action legality
class LatentWorldModel(nn.Module):
    def __init__(self, fused_dim=256, n_actions=N_ACTIONS, hidden_dim=256):
        super().__init__()
        self.fused_dim = fused_dim
        self.n_actions = n_actions

        in_dim = fused_dim + n_actions
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            DynActivation(),
            nn.Linear(hidden_dim, hidden_dim),
            DynActivation(),
        )
        self.trans_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            DynActivation(),
            nn.Linear(hidden_dim, fused_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            DynActivation(),
            nn.Linear(64, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            DynActivation(),
            nn.Linear(128, 1),
        )
        self.legal_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            DynActivation(),
            nn.Linear(128, n_actions),
        )

    def _prepare_input(self, z, action):
        if isinstance(action, int):
            a_idx = torch.tensor([action], device=z.device, dtype=torch.int64)
        elif isinstance(action, np.ndarray):
            a_idx = torch.as_tensor(action, device=z.device, dtype=torch.int64)
        elif isinstance(action, torch.Tensor):
            a_idx = action.to(device=z.device, dtype=torch.int64)
        else:
            raise TypeError(f"bad action type: {type(action)}")

        if a_idx.dim() == 0:
            a_idx = a_idx.unsqueeze(0)
        if a_idx.dim() == 2 and a_idx.size(1) == 1:
            a_idx = a_idx.squeeze(1)

        a_one_hot = F.one_hot(a_idx, num_classes=self.n_actions).float()
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if z.size(0) == 1 and a_one_hot.size(0) > 1:
            z = z.expand(a_one_hot.size(0), -1)

        inp = torch.cat([z, a_one_hot], dim=-1)
        return inp

    def forward(self, z, action):
        x = self._prepare_input(z, action)
        h = self.trunk(x)
        delta_z = self.trans_head(h)
        if z.size(0) == delta_z.size(0):
            next_z = z + delta_z
        else:
            next_z = z.expand(delta_z.size(0), -1) + delta_z
        rew = self.reward_head(h)
        val = self.value_head(h)
        leg = self.legal_head(next_z)
        return next_z, rew, val, leg


# ---------------------------------------------------------------------------
# main agent class
# ---------------------------------------------------------------------------

class SpatialQWMAgent:
    def __init__(self, device=None, **kwargs):
        if device is not None:
            self.device = device
        else:
            self.device = get_device()
        self.kind = "spatial_qwm"
        self.policy_net = SpatialDuelingDQN().to(self.device)
        self.world_model = LatentWorldModel().to(self.device)
        self.policy_net.eval()
        self.world_model.eval()

    @torch.no_grad()
    def q_values(self, spatial, features):
        s = torch.as_tensor(spatial, dtype=torch.float32, device=self.device)
        f = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        if s.dim() == 3:
            s = s.unsqueeze(0)
        if f.dim() == 1:
            f = f.unsqueeze(0)
        out = self.policy_net(s, f)
        res = out.squeeze(0).cpu().numpy()
        return res

    # vectorized world model tree search!
    # old python loops took 600ms per step, this takes ~25ms on GPU
    @torch.no_grad()
    def tree_search_action_scores(
        self,
        spatial,
        features,
        depth=4,
        beam_size=12,
        discount=0.12,
        alpha_vq=0.45,
        valid_mask=None,
        return_details=False,
    ):
        s = torch.as_tensor(spatial, dtype=torch.float32, device=self.device)
        f = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        if s.dim() == 3:
            s = s.unsqueeze(0)
        if f.dim() == 1:
            f = f.unsqueeze(0)

        # encode root state z0
        z0 = self.policy_net.encode_fused(s, f)
        out_q = self.policy_net.forward_from_fused(z0)
        q0 = out_q.squeeze(0)

        # bounds for tree search params
        depth = max(1, min(int(depth), 16))
        beam_size = max(6, int(beam_size))
        K = max(1, beam_size // N_ACTIONS)

        # step 1: evaluate all 6 root actions
        a0 = torch.arange(N_ACTIONS, device=self.device, dtype=torch.int64)
        z1, r0, v0, legal0 = self.world_model(z0.expand(N_ACTIONS, -1), a0)
        # clamp to prevent crazy outliers
        r0_val = torch.nan_to_num(r0.squeeze(-1).clamp(-10.0, 10.0), nan=0.0)
        v0_val = torch.nan_to_num(v0.squeeze(-1).clamp(-50.0, 50.0), nan=0.0)

        q1 = self.policy_net.forward_from_fused(z1)
        legal0_mask = (legal0 >= 0.0)
        legal0_mask[:, ACTION_TO_IDX['WAIT']] = True  # wait is always allowed
        v_q1 = torch.where(legal0_mask, q1, torch.tensor(-1e9, device=self.device)).max(dim=-1)[0]
        # combine critic value and world model value prediction (paper eq 8)
        v1_comb = alpha_vq * v_q1 + (1.0 - alpha_vq) * v0_val

        # depth 1 fast return
        if depth == 1:
            v1 = v1_comb
            q_ts = 0.5 * (q0 + (r0_val + discount * v1))
            if valid_mask is not None:
                vm = torch.as_tensor(valid_mask, dtype=torch.bool, device=self.device)
                q_ts = torch.where(vm, q_ts, torch.tensor(-1e9, device=self.device))
            scores = torch.nan_to_num(q_ts, nan=-1e9).cpu().numpy()
            if return_details == False:
                return scores
            details = []
            for i in range(N_ACTIONS):
                item = {}
                item['a0'] = i
                item['a0_name'] = ACTIONS[i]
                item['a1'] = None
                item['a1_name'] = None
                item['actions'] = [ACTIONS[i]]
                item['q_ts'] = float(scores[i])
                item['q_root'] = float(q0[i].item())
                item['r0'] = float(r0_val[i].item())
                item['v1'] = float(v1[i].item())
                if valid_mask is not None:
                    item['is_valid'] = bool(valid_mask[i])
                else:
                    item['is_valid'] = True
                details.append(item)
            return scores, details

        # rollout deeper steps (beam search over K candidate branches per root action)
        curr_z = z1.unsqueeze(1)
        curr_legal = legal0.unsqueeze(1)
        curr_cum_r = r0_val.unsqueeze(1)
        curr_r_hist = [r0_val.unsqueeze(1)]
        curr_vq_hist = [v1_comb.unsqueeze(1)]
        path_actions = [torch.arange(6, device=self.device).unsqueeze(1)]

        for d in range(1, depth):
            K_curr = curr_z.size(1)
            z_exp = curr_z.unsqueeze(2).expand(6, K_curr, 6, 256).reshape(-1, 256)
            a_exp = torch.arange(6, device=self.device).repeat(6 * K_curr)
            legal_exp = curr_legal.reshape(-1)
            cand_legal_mask = (legal_exp >= 0.0) | (a_exp == ACTION_TO_IDX['WAIT'])

            next_z, next_r, next_v, next_legal = self.world_model(z_exp, a_exp)
            next_r_val = torch.nan_to_num(next_r.squeeze(-1).clamp(-10.0, 10.0), nan=0.0).view(6, K_curr * 6)
            next_v_val = torch.nan_to_num(next_v.squeeze(-1).clamp(-50.0, 50.0), nan=0.0).view(6, K_curr * 6)

            next_q = self.policy_net.forward_from_fused(next_z)
            next_legal_mask = (next_legal >= 0.0)
            next_legal_mask[:, ACTION_TO_IDX['WAIT']] = True
            next_vq = torch.where(next_legal_mask, next_q, torch.tensor(-1e9, device=self.device)).max(dim=-1)[0].view(6, K_curr * 6)
            next_comb = alpha_vq * next_vq + (1.0 - alpha_vq) * next_v_val

            cand_cum_r = curr_cum_r.unsqueeze(-1).expand(6, K_curr, 6).reshape(6, K_curr * 6) + (discount ** d) * next_r_val
            branch_score = cand_cum_r + (discount ** (d + 1)) * next_comb
            branch_score = torch.where(cand_legal_mask.view(6, K_curr * 6), branch_score, torch.tensor(-1e9, device=self.device))

            if d < depth - 1:
                surv_k = min(K, K_curr * 6)
                _, topk_idx = branch_score.topk(surv_k, dim=-1)
                parent_idx = topk_idx // 6
                action_taken = topk_idx % 6

                next_z_reshaped = next_z.view(6, K_curr * 6, 256)
                next_legal_reshaped = next_legal.view(6, K_curr * 6, 6)

                curr_z = next_z_reshaped.gather(1, topk_idx.unsqueeze(-1).expand(6, surv_k, 256))
                curr_legal = next_legal_reshaped.gather(1, topk_idx.unsqueeze(-1).expand(6, surv_k, 6))
                curr_cum_r = cand_cum_r.gather(1, topk_idx)

                curr_r_hist = [h.gather(1, parent_idx) for h in curr_r_hist] + [next_r_val.gather(1, topk_idx)]
                curr_vq_hist = [h.gather(1, parent_idx) for h in curr_vq_hist] + [next_comb.gather(1, topk_idx)]
                path_actions = [p.gather(1, parent_idx) for p in path_actions] + [action_taken]
            else:
                best_idx = branch_score.argmax(dim=-1, keepdim=True)
                parent_idx = best_idx // 6
                action_taken = best_idx % 6

                final_r_hist = [h.gather(1, parent_idx).squeeze(-1) for h in curr_r_hist] + [next_r_val.gather(1, best_idx).squeeze(-1)]
                final_vq_hist = [h.gather(1, parent_idx).squeeze(-1) for h in curr_vq_hist] + [next_comb.gather(1, best_idx).squeeze(-1)]
                all_actions = [p.gather(1, parent_idx).squeeze(-1) for p in path_actions] + [action_taken.squeeze(-1)]

                V_curr = final_vq_hist[-1]
                for step in range(depth - 1, 0, -1):
                    V_curr = alpha_vq * final_vq_hist[step - 1] + (1.0 - alpha_vq) * (final_r_hist[step] + discount * V_curr)
                v1 = V_curr

        q_ts = 0.5 * (q0 + (r0_val + discount * v1))
        if valid_mask is not None:
            vm = torch.as_tensor(valid_mask, dtype=torch.bool, device=self.device)
            q_ts = torch.where(vm, q_ts, torch.tensor(-1e9, device=self.device))

        scores = torch.nan_to_num(q_ts, nan=-1e9).cpu().numpy()
        if return_details == False:
            return scores

        action_matrix = torch.stack(all_actions, dim=1).cpu().numpy()
        details = []
        for i in range(N_ACTIONS):
            acts = []
            for a in action_matrix[i]:
                acts.append(ACTIONS[a])

            if depth > 1:
                a1_val = int(action_matrix[i, 1])
                a1_name = acts[1]
            else:
                a1_val = None
                a1_name = None

            item = {}
            item['a0'] = i
            item['a0_name'] = ACTIONS[i]
            item['a1'] = a1_val
            item['a1_name'] = a1_name
            item['actions'] = acts
            item['q_ts'] = float(scores[i])
            item['q_root'] = float(q0[i].item())
            item['r0'] = float(r0_val[i].item())
            item['v1'] = float(v1[i].item())
            if valid_mask is not None:
                item['is_valid'] = bool(valid_mask[i])
            else:
                item['is_valid'] = True
            details.append(item)

        return scores, details

    # loads checkpoint weights from file
    def load(self, filepath, need_training=False, partial=True, **kwargs):
        checkpoint = torch.load(filepath, map_location=self.device)
        loaded_keys = []
        uninit_keys = []

        if 'policy_state_dict' in checkpoint:
            pol_dict = checkpoint['policy_state_dict']
        else:
            pol_dict = checkpoint

        l_pol, u_pol = load_matching_weights(self.policy_net, pol_dict, partial=partial)
        for k in l_pol:
            loaded_keys.append("policy." + str(k))
        for k in u_pol:
            uninit_keys.append("policy." + str(k))

        if 'world_model_state_dict' in checkpoint:
            wm_dict = checkpoint['world_model_state_dict']
            l_wm, u_wm = load_matching_weights(self.world_model, wm_dict, partial=partial)
            for k in l_wm:
                loaded_keys.append("wm." + str(k))
            for k in u_wm:
                uninit_keys.append("wm." + str(k))
        else:
            for k in self.world_model.state_dict().keys():
                uninit_keys.append("wm." + str(k))

        self.policy_net.eval()
        self.world_model.eval()
        return loaded_keys, uninit_keys


def load_model(filepath, need_training=False, partial=False, device=None):
    agent = SpatialQWMAgent(device=device)
    agent.load(filepath, need_training=need_training, partial=partial)
    return agent
