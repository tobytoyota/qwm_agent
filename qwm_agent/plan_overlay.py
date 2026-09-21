# visual plan overlay for pygame gui
# draws the green path line on the board so we can see what the agent is planning during the demo
# monkey patches pygame.display.flip/update so we don't have to edit the main game engine code

from dataclasses import dataclass, field
from pathlib import Path
import sys
import numpy as np

import settings as s

try:
    import pygame
    HAVE_PYGAME = True
except ImportError:
    HAVE_PYGAME = False

# colors for drawing: green for top path, yellow for bomb, red for blocked
TOP_PATH_COLOR = (0, 255, 128)      # emerald green
TOP_PATH_WIDTH = 5
BOMB_RING_COLOR = (255, 210, 0)     # yellow circle where bomb is placed
BLOCKED_COLOR = (255, 60, 60)       # red dot if move is blocked by wall/crate

# board dimensions from settings.py
GRID_OFFSET = getattr(s, 'GRID_OFFSET', (30, 30))
GRID_SIZE = getattr(s, 'GRID_SIZE', 30)
COLS = getattr(s, 'COLS', 17)
ROWS = getattr(s, 'ROWS', 17)


@dataclass
class CandidatePlanTrace:
    rank: int
    actions: list
    tiles: list
    total_score: float
    predicted_return: float = None
    value_bootstrap: float = None
    blocked_from: int = None
    bomb_steps: list = field(default_factory=list)
    first_action: str = 'WAIT'
    first_action_legal: bool = True


@dataclass
class PlannerOverlayTrace:
    round_id: int
    env_step: int
    agent_position: tuple
    planner_elapsed_ms: float
    actor_action: str
    planner_action: str
    planner_changed_action: bool = False
    fallback_used: bool = False
    candidates: list = field(default_factory=list)


# simulates walking an action sequence on the board
def walk_path(start_pos, actions, arena=None, bombs=None):
    cx = start_pos[0]
    cy = start_pos[1]
    tiles = []
    tiles.append((cx, cy))
    blocked_from = None
    bomb_steps = []

    if arena is not None:
        dynamic_arena = np.copy(arena)
    else:
        dynamic_arena = None

    active_bombs = []
    if bombs is not None:
        for b in bombs:
            if isinstance(b, (tuple, list)):
                if isinstance(b[0], (tuple, list)):
                    active_bombs.append([b[0][0], b[0][1], int(b[1])])
                elif len(b) >= 3:
                    active_bombs.append([b[0], b[1], int(b[2])])

    for step_idx in range(len(actions)):
        act = actions[step_idx]
        if act == 'BOMB':
            bomb_steps.append(step_idx)
            active_bombs.append([cx, cy, 4])  # 4 tick fuse

        newly_active = []
        for b in active_bombs:
            b[2] = b[2] - 1
            if b[2] <= 0:
                bx = b[0]
                by = b[1]
                if dynamic_arena is not None:
                    blast_dirs = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
                    for bd in blast_dirs:
                        dx = bd[0]
                        dy = bd[1]
                        for r in range(1, 4):
                            tx = bx + dx * r
                            ty = by + dy * r
                            if tx < 0 or tx >= dynamic_arena.shape[0]:
                                break
                            if ty < 0 or ty >= dynamic_arena.shape[1]:
                                break
                            if dynamic_arena[tx, ty] == -1:
                                break  # stone wall
                            if dynamic_arena[tx, ty] == 1:
                                dynamic_arena[tx, ty] = 0  # broke crate
                                break
            else:
                newly_active.append(b)
        active_bombs = newly_active

        # move agent position using simple if-else
        if act == 'UP':
            dx = 0
            dy = -1
        elif act == 'DOWN':
            dx = 0
            dy = 1
        elif act == 'LEFT':
            dx = -1
            dy = 0
        elif act == 'RIGHT':
            dx = 1
            dy = 0
        else:
            dx = 0
            dy = 0

        nx = cx + dx
        ny = cy + dy

        if dynamic_arena is not None:
            is_blocked = False
            if nx < 0 or nx >= dynamic_arena.shape[0]:
                is_blocked = True
            elif ny < 0 or ny >= dynamic_arena.shape[1]:
                is_blocked = True
            elif dynamic_arena[nx, ny] != 0:
                is_blocked = True

            if is_blocked == True:
                if blocked_from is None:
                    blocked_from = step_idx
                nx = cx
                ny = cy

        cx = nx
        cy = ny
        tiles.append((cx, cy))

    return tiles, blocked_from, bomb_steps


# global variables for overlay state
_LATEST_TRACE = None
_RENDERED_ON_CURRENT_FRAME = False
_HOOK_INSTALLED = False
_OVERLAY_FONTS = {}


def publish_overlay_trace(trace):
    global _LATEST_TRACE
    _LATEST_TRACE = trace
    if _HOOK_INSTALLED == False and HAVE_PYGAME == True:
        install_render_hook()


def clear_overlay_trace():
    global _LATEST_TRACE, _RENDERED_ON_CURRENT_FRAME
    _LATEST_TRACE = None
    _RENDERED_ON_CURRENT_FRAME = False


def get_latest_overlay_trace():
    return _LATEST_TRACE


# calculate screen pixel coordinates for tile (x, y)
def tile_center(x, y):
    px = GRID_OFFSET[0] + GRID_SIZE * x + GRID_SIZE // 2
    py = GRID_OFFSET[1] + GRID_SIZE * y + GRID_SIZE // 2
    return (px, py)


def _get_overlay_font(size='small'):
    if HAVE_PYGAME == False:
        return None
    if size not in _OVERLAY_FONTS:
        font_file = getattr(s, 'ASSET_DIR', Path('.')) / 'emulogic.ttf'
        if size == 'small':
            font_size = 8
        else:
            font_size = 10
        try:
            if hasattr(font_file, 'is_file') and font_file.is_file():
                _OVERLAY_FONTS[size] = pygame.font.Font(str(font_file), font_size)
            else:
                _OVERLAY_FONTS[size] = pygame.font.SysFont('consolas', font_size)
        except Exception:
            _OVERLAY_FONTS[size] = pygame.font.Font(None, font_size + 4)
    return _OVERLAY_FONTS.get(size)


def _render_text(screen, gui, text, x, y, color, size='small'):
    if gui is not None and hasattr(gui, 'render_text'):
        gui.render_text(text, x, y, color, size=size)
    elif HAVE_PYGAME == True and isinstance(screen, pygame.Surface):
        font = _get_overlay_font(size)
        if font is not None:
            surf = font.render(text, False, color)
            screen.blit(surf, (x, y))


def _draw_path(screen, path, color, width):
    if HAVE_PYGAME == False or not isinstance(screen, pygame.Surface):
        return
    if not hasattr(path, 'tiles') or path.tiles is None or len(path.tiles) == 0:
        return

    points = []
    for t in path.tiles:
        points.append(tile_center(t[0], t[1]))

    if len(points) > 1:
        pygame.draw.lines(screen, color, False, points, width)

    # start circle
    pygame.draw.circle(screen, color, points[0], max(4, width + 1))

    # waypoints along path
    for i in range(1, len(points)):
        pt = points[i]
        pygame.draw.circle(screen, color, pt, max(2, width - 2))
        if path.blocked_from is not None:
            if i == path.blocked_from + 1:
                pygame.draw.circle(screen, BLOCKED_COLOR, pt, max(5, width + 2), 2)

    # bomb drop marker
    for step_idx in path.bomb_steps:
        if step_idx < len(points):
            pygame.draw.circle(screen, BOMB_RING_COLOR, points[step_idx], GRID_SIZE // 3, 2)

    # arrowhead at the end
    if len(points) > 1:
        if points[-1] != points[-2]:
            _draw_arrowhead(screen, points[-2], points[-1], color, width)


def _draw_arrowhead(screen, from_pt, to_pt, color, width):
    dx = to_pt[0] - from_pt[0]
    dy = to_pt[1] - from_pt[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1.0:
        length = 1.0
    ux = dx / length
    uy = dy / length
    size = 5 + width
    tip = (to_pt[0] + ux * size * 0.6, to_pt[1] + uy * size * 0.6)
    left = (to_pt[0] - uy * size * 0.5, to_pt[1] + ux * size * 0.5)
    right = (to_pt[0] + uy * size * 0.5, to_pt[1] - ux * size * 0.5)
    polygon_points = [(int(tip[0]), int(tip[1])), (int(left[0]), int(left[1])), (int(right[0]), int(right[1]))]
    pygame.draw.polygon(screen, color, polygon_points)


def _format_actions(actions, limit=10):
    res = ""
    upper = min(len(actions), limit)
    for i in range(upper):
        a = actions[i]
        if a == 'UP':
            res = res + 'U'
        elif a == 'DOWN':
            res = res + 'D'
        elif a == 'LEFT':
            res = res + 'L'
        elif a == 'RIGHT':
            res = res + 'R'
        elif a == 'WAIT':
            res = res + '.'
        elif a == 'BOMB':
            res = res + 'B'
        else:
            res = res + '?'
    if len(actions) > limit:
        res = res + "+" + str(len(actions) - limit)
    return res


def _find_active_agent_pos(gui, trace):
    if gui is not None and hasattr(gui, 'world') and hasattr(gui.world, 'active_agents'):
        for a in gui.world.active_agents:
            code_name = getattr(a, 'code_name', '').lower()
            name = getattr(a, 'name', '').lower()
            if 'veryclean' in code_name or 'qwm' in code_name or 'qwm' in name:
                return (a.x, a.y)
        if len(gui.world.active_agents) > 0:
            return (gui.world.active_agents[0].x, gui.world.active_agents[0].y)

    try:
        env_mod = sys.modules.get('environment')
        if env_mod:
            w = getattr(env_mod, 'world', None) or getattr(env_mod, '_world', None)
            if w and hasattr(w, 'active_agents'):
                for a in w.active_agents:
                    code_name = getattr(a, 'code_name', '').lower()
                    if 'veryclean' in code_name or 'qwm' in code_name:
                        return (a.x, a.y)
    except Exception:
        pass

    if trace is not None:
        return trace.agent_position
    return None


# draws top path and info box
def render_overlay(screen, gui=None):
    global _RENDERED_ON_CURRENT_FRAME
    if HAVE_PYGAME == False or _LATEST_TRACE is None:
        return

    trace = _LATEST_TRACE
    candidates = getattr(trace, 'candidates', []) or []
    if len(candidates) == 0:
        return

    # find best evaluated path
    top_path = min(candidates, key=lambda p: p.rank)
    agent_pos = _find_active_agent_pos(gui, trace)

    path_to_draw = top_path
    if agent_pos is not None and hasattr(top_path, 'tiles') and len(top_path.tiles) > 1:
        if agent_pos != top_path.tiles[0] and agent_pos == top_path.tiles[1]:
            # slice if agent already stepped forward
            if len(top_path.actions) > 1:
                new_actions = top_path.actions[1:]
                new_first_act = top_path.actions[1]
            else:
                new_actions = top_path.actions
                new_first_act = top_path.first_action

            if top_path.blocked_from is not None:
                new_blocked = max(0, top_path.blocked_from - 1)
            else:
                new_blocked = None

            new_bombs = []
            for b in top_path.bomb_steps:
                if b >= 1:
                    new_bombs.append(max(0, b - 1))

            path_to_draw = CandidatePlanTrace(
                rank=top_path.rank,
                actions=new_actions,
                tiles=top_path.tiles[1:],
                total_score=top_path.total_score,
                predicted_return=top_path.predicted_return,
                value_bootstrap=top_path.value_bootstrap,
                blocked_from=new_blocked,
                bomb_steps=new_bombs,
                first_action=new_first_act,
                first_action_legal=top_path.first_action_legal,
            )

    _draw_path(screen, path_to_draw, TOP_PATH_COLOR, TOP_PATH_WIDTH)
    _render_legend(screen, gui, trace, top_path)
    _RENDERED_ON_CURRENT_FRAME = True


# draws info box on right sidebar
def _render_legend(screen, gui, trace, top_path):
    if HAVE_PYGAME == False or not isinstance(screen, pygame.Surface):
        return

    x = GRID_OFFSET[0] + COLS * GRID_SIZE + 15
    y = 250

    _render_text(screen, gui, "QWM TOP PLAN", x, y, (220, 220, 220), size='small')
    y = y + 16

    pygame.draw.rect(screen, TOP_PATH_COLOR, pygame.Rect(x, y + 2, 12, 5))
    act_str = _format_actions(top_path.actions)
    plan_label = f"#1 {act_str}  Q:{top_path.total_score:+.2f}"
    _render_text(screen, gui, plan_label, x + 18, y, TOP_PATH_COLOR, size='small')
    y = y + 14

    details = []
    if top_path.predicted_return is not None:
        details.append(f"R:{top_path.predicted_return:+.2f}")
    if top_path.value_bootstrap is not None:
        details.append(f"V:{top_path.value_bootstrap:+.2f}")
    if len(details) > 0:
        det_text = "  " + " | ".join(details)
        _render_text(screen, gui, det_text, x + 18, y, (160, 160, 160), size='small')
        y = y + 14

    plan_act = getattr(trace, 'planner_action', 'WAIT')
    timing = getattr(trace, 'planner_elapsed_ms', 0.0)
    msg = f"Act: {plan_act} ({timing:.1f}ms)"
    _render_text(screen, gui, msg, x, y, (180, 180, 180), size='small')


# hooks pygame display flip/update and GUI.render so overlay draws automatically
def install_render_hook():
    global _HOOK_INSTALLED
    if _HOOK_INSTALLED == True or HAVE_PYGAME == False:
        return

    try:
        env_mod = sys.modules.get('environment')
        if env_mod and hasattr(env_mod, 'GUI'):
            GUI_cls = getattr(env_mod, 'GUI')
            if not getattr(GUI_cls, '_qwm_overlay_hooked', False):
                orig_gui_render = GUI_cls.render

                def hooked_gui_render(self_gui, *args, **kwargs):
                    res = orig_gui_render(self_gui, *args, **kwargs)
                    try:
                        render_overlay(self_gui.screen, self_gui)
                    except Exception:
                        pass
                    return res

                GUI_cls.render = hooked_gui_render
                GUI_cls._qwm_overlay_hooked = True
    except Exception:
        pass

    try:
        if hasattr(pygame.display, 'flip') and not getattr(pygame.display, '_qwm_flip_hooked', False):
            orig_flip = pygame.display.flip

            def hooked_flip(*args, **kwargs):
                global _RENDERED_ON_CURRENT_FRAME
                try:
                    surface = pygame.display.get_surface()
                    if surface is not None and not _RENDERED_ON_CURRENT_FRAME:
                        render_overlay(surface, gui=None)
                except Exception:
                    pass
                _RENDERED_ON_CURRENT_FRAME = False
                return orig_flip(*args, **kwargs)

            pygame.display.flip = hooked_flip
            pygame.display._qwm_flip_hooked = True
    except Exception:
        pass

    try:
        if hasattr(pygame.display, 'update') and not getattr(pygame.display, '_qwm_update_hooked', False):
            orig_update = pygame.display.update

            def hooked_update(*args, **kwargs):
                global _RENDERED_ON_CURRENT_FRAME
                try:
                    surface = pygame.display.get_surface()
                    if surface is not None and not _RENDERED_ON_CURRENT_FRAME:
                        render_overlay(surface, gui=None)
                except Exception:
                    pass
                _RENDERED_ON_CURRENT_FRAME = False
                return orig_update(*args, **kwargs)

            pygame.display.update = hooked_update
            pygame.display._qwm_update_hooked = True
    except Exception:
        pass

    _HOOK_INSTALLED = True


if HAVE_PYGAME == True:
    try:
        install_render_hook()
    except Exception:
        pass
