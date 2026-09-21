# callbacks for qwm bomberman agent
# uses spatial dueling dqn + world model tree search + small path mlp
# basically Dong et al 2026 tree search + small network to fix dumb corridor loops

from collections import deque
import json
import logging
import os
import time

import numpy as np

from .model import (
    ACTIONS,
    agent_path,
    load_model,
    state_to_features,
    state_to_spatial_tensor,
    valid_action_mask,
)
from .path_penalty_model import (
    load_small_model,
    extract_path_candidate_features,
)

try:
    from .plan_overlay import (
        CandidatePlanTrace,
        PlannerOverlayTrace,
        publish_overlay_trace,
        clear_overlay_trace,
        walk_path,
        install_render_hook,
    )
except ImportError:
    CandidatePlanTrace = None
    PlannerOverlayTrace = None
    publish_overlay_trace = None
    clear_overlay_trace = lambda: None
    walk_path = None
    install_render_hook = lambda: None

DEFAULT_CONFIG = {
    'model_file': 'my-saved-model-spatial-qwm.pt',
    'tree_search': True,
    'search_depth': 5,
    'beam_size': 18,
    'tree_discount': 0.08,  # paper suggested ~0.1, 0.08 works well
    'alpha_vq': 0.16,
    'use_small_model': True,
    'small_model_file': 'small_model.pt',
    'small_model_weight': 0.25,
}


def load_config():
    cfg = {}
    for k, v in DEFAULT_CONFIG.items():
        cfg[k] = v

    p = agent_path('config.json')
    if os.path.isfile(p):
        with open(p, 'r') as f:
            data = json.load(f)
            for k in data:
                cfg[k] = data[k]
    return cfg


def setup(self):
    # agent setup stuff at match start
    if hasattr(self, 'logger') and self.logger is not None:
        pass
    else:
        self.logger = logging.getLogger('qwm_agent_veryclean')

    self.cfg = load_config()

    # load the main DQN + world model checkpoint
    if 'MY_QWM_MODEL_FILE' in os.environ:
        model_name = os.environ['MY_QWM_MODEL_FILE']
    else:
        model_name = self.cfg['model_file']

    if os.path.isfile(model_name):
        model_path = os.path.abspath(model_name)
    else:
        model_path = agent_path(model_name)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")

    self.model = load_model(model_path, need_training=False, partial=False)
    self.model_file = model_path
    self.logger.info("Loaded trained Spatial QWM model from %s", os.path.basename(model_path))

    # keep track of recent steps & stagnation
    self.coordinate_history = deque(maxlen=40)
    self._last_round = None
    self._consecutive_stationary_steps = 0
    self._steps_without_progress = 0
    self._last_score = 0
    self._last_crates = None

    # tree search settings
    if 'MY_QWM_TREE_SEARCH' in os.environ:
        ts_env = os.environ['MY_QWM_TREE_SEARCH']
        if ts_env.lower() in ('1', 'true', 'yes'):
            self.tree_search = True
        else:
            self.tree_search = False
    else:
        if 'tree_search' in self.cfg:
            self.tree_search = bool(self.cfg['tree_search'])
        else:
            self.tree_search = True

    if 'MY_QWM_SEARCH_DEPTH' in os.environ:
        sd = int(os.environ['MY_QWM_SEARCH_DEPTH'])
    elif 'search_depth' in self.cfg:
        sd = int(self.cfg['search_depth'])
    else:
        sd = 5

    if sd < 1:
        sd = 1
    if sd > 16:
        sd = 16
    self.search_depth = sd

    if 'MY_QWM_BEAM_SIZE' in os.environ:
        bs = int(os.environ['MY_QWM_BEAM_SIZE'])
    elif 'beam_size' in self.cfg:
        bs = int(self.cfg['beam_size'])
    else:
        bs = 18

    if bs < 6:
        bs = 6
    if bs > 60:
        bs = 60
    self.beam_size = bs

    if 'MY_QWM_TREE_DISCOUNT' in os.environ:
        self.tree_discount = float(os.environ['MY_QWM_TREE_DISCOUNT'])
    elif 'tree_discount' in self.cfg:
        self.tree_discount = float(self.cfg['tree_discount'])
    else:
        self.tree_discount = 0.08

    if 'MY_QWM_ALPHA_VQ' in os.environ:
        self.alpha_vq = float(os.environ['MY_QWM_ALPHA_VQ'])
    elif 'alpha_vq' in self.cfg:
        self.alpha_vq = float(self.cfg['alpha_vq'])
    else:
        self.alpha_vq = 0.16

    # small mlp model for corridor loop penalty
    if 'MY_QWM_USE_SMALL_MODEL' in os.environ:
        sm_env = os.environ['MY_QWM_USE_SMALL_MODEL']
        if sm_env.lower() in ('1', 'true', 'yes'):
            self.use_small_model = True
        else:
            self.use_small_model = False
    else:
        if 'use_small_model' in self.cfg:
            self.use_small_model = bool(self.cfg['use_small_model'])
        else:
            self.use_small_model = True

    if 'MY_QWM_SMALL_MODEL_FILE' in os.environ:
        sm_file = os.environ['MY_QWM_SMALL_MODEL_FILE']
    elif 'small_model_file' in self.cfg:
        sm_file = self.cfg['small_model_file']
    else:
        sm_file = 'small_model.pt'

    if os.path.isfile(sm_file):
        sm_path = os.path.abspath(sm_file)
    else:
        sm_path = agent_path(sm_file)

    if self.use_small_model == True and os.path.isfile(sm_path):
        self.small_model = load_small_model(sm_path, device=self.model.device)
        self.logger.info("Loaded Small Path Reward Model from %s", os.path.basename(sm_path))
    else:
        self.small_model = None

    if 'MY_QWM_SMALL_MODEL_WEIGHT' in os.environ:
        self.small_model_weight = float(os.environ['MY_QWM_SMALL_MODEL_WEIGHT'])
    elif 'small_model_weight' in self.cfg:
        self.small_model_weight = float(self.cfg['small_model_weight'])
    else:
        self.small_model_weight = 0.25

    # install render hook for pygame gui if active
    if install_render_hook is not None:
        try:
            install_render_hook()
        except Exception:
            pass


def act(self, game_state):
    start_time = time.perf_counter()
    if game_state is None:
        return 'WAIT'

    # reset tracking if new round started
    step = game_state.get('step', 1)
    round_num = game_state.get('round', 1)

    if step == 1 or round_num != self._last_round:
        self.coordinate_history.clear()
        self._last_round = round_num
        self._consecutive_stationary_steps = 0
        self._steps_without_progress = 0
        self._last_score = game_state['self'][1]

        crates_count = 0
        if 'field' in game_state and game_state['field'] is not None:
            f = game_state['field']
            for rx in range(f.shape[0]):
                for ry in range(f.shape[1]):
                    if f[rx, ry] == 1:
                        crates_count = crates_count + 1
        self._last_crates = crates_count

        if clear_overlay_trace is not None:
            clear_overlay_trace()

    x = game_state['self'][3][0]
    y = game_state['self'][3][1]

    # check if standing still
    if len(self.coordinate_history) > 0:
        prev_pos = self.coordinate_history[-1]
        if prev_pos[0] == x and prev_pos[1] == y:
            self._consecutive_stationary_steps = self._consecutive_stationary_steps + 1
        else:
            self._consecutive_stationary_steps = 0
    else:
        self._consecutive_stationary_steps = 0

    # check if game progressed
    current_score = game_state['self'][1]
    current_crates = 0
    if 'field' in game_state and game_state['field'] is not None:
        f = game_state['field']
        for rx in range(f.shape[0]):
            for ry in range(f.shape[1]):
                if f[rx, ry] == 1:
                    current_crates = current_crates + 1

    progress_made = False
    if current_score > self._last_score:
        progress_made = True
    elif self._last_crates is not None:
        if current_crates < self._last_crates:
            progress_made = True

    if progress_made == True:
        self._steps_without_progress = 0
    else:
        self._steps_without_progress = self._steps_without_progress + 1

    self._last_score = current_score
    self._last_crates = current_crates

    # state feature extraction
    spatial = state_to_spatial_tensor(game_state, self.coordinate_history)
    features = state_to_features(game_state, self.coordinate_history)
    valid_mask = valid_action_mask(features)

    # don't drop bombs in coin heaven (task 1) or we blow ourselves up like an idiot
    field = game_state.get('field')
    others = game_state.get('others', [])
    if field is not None:
        crates_total = 0
        for rx in range(field.shape[0]):
            for ry in range(field.shape[1]):
                if field[rx, ry] == 1:
                    crates_total = crates_total + 1
        if crates_total == 0 and len(others) == 0:
            valid_mask[5] = False  # index 5 = BOMB

    # 1. compute action scores using tree search
    if self.tree_search == True:
        scores, plan_details = self.model.tree_search_action_scores(
            spatial,
            features,
            depth=self.search_depth,
            beam_size=self.beam_size,
            discount=self.tree_discount,
            alpha_vq=self.alpha_vq,
            valid_mask=valid_mask,
            return_details=True,
        )
    else:
        scores = self.model.q_values(spatial, features)
        plan_details = None

    scores = np.asarray(scores, dtype=np.float64)
    for i in range(len(ACTIONS)):
        if valid_mask[i] == False:
            scores[i] = -np.inf

    # 2. small MLP adjustments to penalize silly corridor loops
    if self.use_small_model == True and self.small_model is not None:
        path_feats = extract_path_candidate_features(
            game_state,
            plan_details,
            self.coordinate_history,
            consecutive_stationary_steps=self._consecutive_stationary_steps,
            steps_without_progress=self._steps_without_progress,
            base_scores=scores,
        )
        deltas = self.small_model.predict_deltas(path_feats)
        for act_idx in range(len(ACTIONS)):
            scores[act_idx] = scores[act_idx] + self.small_model_weight * deltas[act_idx]

    # 3. select best action with loop
    best_idx = -1
    best_score = -999999999.0
    for i in range(len(ACTIONS)):
        if valid_mask[i] == True:
            if scores[i] > best_score:
                best_score = scores[i]
                best_idx = i

    if best_idx == -1:
        for i in range(len(ACTIONS)):
            if valid_mask[i] == True:
                best_idx = i
                break

    if best_idx == -1:
        best_idx = 4  # fallback to WAIT

    action = ACTIONS[best_idx]

    # publish plan trace to pygame overlay if gui is running
    if plan_details is not None and publish_overlay_trace is not None and CandidatePlanTrace is not None:
        path_data = []
        if walk_path is not None:
            for cand in plan_details:
                tiles, blocked, bomb_steps = walk_path(
                    (x, y),
                    cand['actions'],
                    arena=game_state.get('field'),
                    bombs=game_state.get('bombs'),
                )
                path_data.append((tiles, blocked, bomb_steps))

        candidate_traces = []
        sorted_indices = []
        # sort indices by score descending
        order = np.argsort(-scores)
        for idx_val in order:
            sorted_indices.append(int(idx_val))

        for rank in range(len(sorted_indices)):
            idx = sorted_indices[rank]
            cand = plan_details[idx]
            if len(path_data) > idx:
                tiles = path_data[idx][0]
                blocked = path_data[idx][1]
                bomb_steps = path_data[idx][2]
            else:
                tiles = [(x, y)]
                blocked = None
                bomb_steps = []

            ret_val = 0.0
            if 'r0' in cand and cand['r0'] is not None:
                ret_val = float(cand['r0'])

            v_val = 0.0
            if 'v1' in cand and cand['v1'] is not None:
                v_val = float(cand['v1'])

            candidate_traces.append(CandidatePlanTrace(
                rank=rank,
                actions=cand['actions'],
                tiles=tiles,
                total_score=float(scores[idx]),
                predicted_return=ret_val,
                value_bootstrap=v_val,
                blocked_from=blocked,
                bomb_steps=bomb_steps,
                first_action=ACTIONS[idx],
                first_action_legal=bool(valid_mask[idx]),
            ))

        time_ms = (time.perf_counter() - start_time) * 1000.0
        trace = PlannerOverlayTrace(
            round_id=int(game_state.get('round', 0)),
            env_step=int(game_state.get('step', 0)),
            agent_position=(x, y),
            planner_elapsed_ms=time_ms,
            actor_action=action,
            planner_action=action,
            candidates=candidate_traces,
        )
        publish_overlay_trace(trace)

    self.coordinate_history.append((x, y))
    return action
