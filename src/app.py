from lib.connect4 import Connect4, Connect4Board
from lib.mcts import MCTS, MCTSNode, Optional, List
from lib.d_tree import predict, load_tree
from src.window import Window
import numpy as np

import pygame
import time

class App:
    def __init__(self) -> None:
        self.window:      Window          = Window(0.70)
        self.running:     bool            = True
        self.game_start:  bool            = False
        self.gamemode:    int             = 0
        self.opponent:    int             = 0
        self.mouse_click: tuple[int, int] = None

        font = pygame.font.Font(None, int(self.window.scale * self.window.info.current_w) // 25)
        self.font = pygame.font.Font(None, int(self.window.scale * self.window.info.current_w) // 30)

        # width: int  = int(self.window.scale * self.window.info.current_w) // 20
        # height: int = int(self.window.scale * self.window.info.current_h) // 20
        # esc: pygame.Rect = pygame.Rect(
            # int(self.window.scale * self.window.info.current_w) - width - 10,
            # int(self.window.scale * self.window.info.current_h) - height - 10,
            # width,
            # height
        # )
        # text_esc = font.render("Menu", True, (255, 255, 255))
        # self.esc_button = (esc, text_esc)

        button_pvp: pygame.Rect = pygame.Rect(
            int(self.window.scale * self.window.info.current_w) // 4,
            int((0.5 - 0.2 - 0.2 / 2 - 0.05) * self.window.scale * self.window.info.current_h),
            # half the screen minus the button height minus half the center button height
            int(self.window.scale * self.window.info.current_w) // 2,
            int(0.2 * self.window.scale * self.window.info.current_h)
        )
        button_pvb: pygame.Rect = pygame.Rect(
            int(self.window.scale * self.window.info.current_w) // 4,
            int((0.5 - 0.2 / 2) * self.window.scale * self.window.info.current_h),
            # half the screen minus the half center button height
            int(self.window.scale * self.window.info.current_w) // 2,
            int(0.2 * self.window.scale * self.window.info.current_h)
        )
        button_bvb: pygame.Rect = pygame.Rect(
            int(self.window.scale * self.window.info.current_w) // 4,
            int((0.5 + 0.2 / 2 + 0.05) * self.window.scale * self.window.info.current_h),
            # half the screen plus half the center button height
            int(self.window.scale * self.window.info.current_w) // 2,
            int(0.2 * self.window.scale * self.window.info.current_h)
        )

        text_pvp = font.render("P VS P", True, (255, 255, 255))
        text_pvb = font.render("P VS B", True, (255, 255, 255))
        text_bvb = font.render("B VS B", True, (255, 255, 255))

        self.menu_objects = [
            (button_pvp, text_pvp),
            (button_pvb, text_pvb),
            (button_bvb, text_bvb)
        ]

        button1: pygame.Rect = pygame.Rect(
            int(self.window.scale * self.window.info.current_w) // 4,
            int((0.5 - 0.2 - 0.05) * self.window.scale * self.window.info.current_h),
            # half the screen minus the button height minus half the center button height
            int(self.window.scale * self.window.info.current_w) // 2,
            int(0.2 * self.window.scale * self.window.info.current_h)
        )
        button2: pygame.Rect = pygame.Rect(
            int(self.window.scale * self.window.info.current_w) // 4,
            int((0.5 + 0.05) * self.window.scale * self.window.info.current_h),
            # half the screen minus the half center button height
            int(self.window.scale * self.window.info.current_w) // 2,
            int(0.2 * self.window.scale * self.window.info.current_h)
        )

        text_mcts = font.render("Monte Carlo", True, (255, 255, 255))
        text_dt = font.render("Decision Tree", True, (255, 255, 255))
        text_mcts_vs_dt = font.render("Monte Carlo VS Decision Tree", True, (255, 255, 255))
        text_dt_vs_mcts = font.render("Decision Tree VS Monte Carlo", True, (255, 255, 255))

        self.pick_opponnent_objects = (
            [button1, button2],
            [text_mcts, text_dt, text_mcts_vs_dt, text_dt_vs_mcts]
        )

        self.player1_win = font.render("Player 1 Wins", True, (242, 233, 78))
        self.player2_win = font.render("Player 2 Wins", True, (218, 62, 82))
        self.draw = font.render("Draw", True, (218, 62, 82))

        width: int = int(0.33 * self.window.scale * self.window.info.current_w)
        height: int = int(self.window.scale * self.window.info.current_h)
        eval_base_rect: pygame.Rect = pygame.Rect(
            int(self.window.scale * self.window.info.current_w) - width,
            0,
            width,
            height
        )
        self.eval = [True, eval_base_rect]

def draw_base(window: Window) -> None:
    color: tuple[int, int, int] = (0, 31, 63)
    window.window.fill(color)

def draw_menu(app: App) -> int:
    draw_base(app.window)
    for i, (rect, text) in enumerate(app.menu_objects):
        pygame.draw.rect(app.window.window, (19, 75, 112), rect, border_radius = 10)
        app.window.window.blit(
            text,
            text.get_rect(
                center = (rect.x + rect.width // 2, rect.y + rect.height // 2)
            )
        )
        if app.mouse_click != None and rect.collidepoint(app.mouse_click[0], app.mouse_click[1]):
            return i + 1
    return 0

def pick_opponnent(app: App) -> int:
    draw_base(app.window)
    for i, rect in enumerate(app.pick_opponnent_objects[0]):
        pygame.draw.rect(app.window.window, (19, 75, 112), rect, border_radius = 10)
        text = app.pick_opponnent_objects[1][i + (2 if app.gamemode == 3 else 0)]
        app.window.window.blit(
            text,
            text.get_rect(center = (rect.x + rect.width // 2, rect.y + rect.height // 2))
        )
        if app.mouse_click != None and rect.collidepoint(app.mouse_click[0], app.mouse_click[1]):
            return i + 1
    return 0

def draw_result(app: App, result: int) -> None:
    label = app.player1_win if result == 1 else (app.player2_win if result == 2 else app.draw)
    app.window.window.blit(
        label,
        label.get_rect(
            center = (
                int(.5 * app.window.scale * (app.window.info.current_w - label.get_rect().width)),
                int(.5 * app.window.scale * (app.window.info.current_h - label.get_rect().height))
            )
        )
    )

def draw_game(app: App, state: Connect4Board, root: MCTSNode, move_count: int, player: int) -> None:
    draw_base(app.window)

    # pygame.draw.rect(app.window.window, (47, 69, 80), app.esc_button[0], border_radius = 10)
    # app.window.window.blit(
        # app.esc_button[1],
        # app.esc_button[1].get_rect(
            # center = (
                # (app.esc_button[0].x + app.esc_button[0].width - app.esc_button[1].get_rect().width) // 2,
                # (app.esc_button[0].y + app.esc_button[0].height - app.esc_button[1].get_rect().height) // 2
            # )
        # )
    # )

    rows, cols = state.rows, state.cols
    s_row, s_col = (
        int(app.window.scale * app.window.info.current_h) // (rows + 1), # extra row for piece to be placed
        int(app.window.scale * app.window.info.current_h) // cols        # square board
    )

    base_rect: pygame.Rect = pygame.Rect(
        0, # int(app.window.scale * app.window.info.current_w - cols * s_col) // 2,
        s_row,
        cols * s_col + 10,
        app.window.info.current_h - s_row
    )
    pygame.draw.rect(app.window.window, (19, 75, 112), base_rect, border_radius = 10)
    
    color: tuple[int, int, int] = (0, 31, 63)
    radius: int = int(0.90 * min(s_row, s_col)) // 2

    for i in range(rows):
        for j in range(cols):
            if state.get_piece(i, j) == 1:
                color = (242, 233, 78)
            elif state.get_piece(i, j) == 2:
                color = (218, 62, 82)
            else:
                color = (0, 31, 63)

            center: tuple[int, int] = (
                base_rect.x + j * (s_col + 1) + s_col // 2, base_rect.y + i * s_row + s_row // 2
            )
            pygame.draw.circle(app.window.window, color, center, radius)
            if color != (0, 31, 63):
                pygame.draw.circle(
                    app.window.window, 
                    (color[0] * 0.90, color[1] * 0.90, color[2] * 0.90), 
                    center, 
                    radius * 0.85
                )

            if i == rows - 1:
                col_name = app.font.render(f"{chr(ord('A') + j)}", True, (255, 255, 255))
                app.window.window.blit(col_name, col_name.get_rect(center = center))

    mouse_x, _ = pygame.mouse.get_pos()
    center: tuple[int, int] = (
        base_rect.x + radius 
        if mouse_x - radius < base_rect.x 
        else (
            base_rect.x + base_rect.width - radius
            if mouse_x + radius > base_rect.x + base_rect.width
            else mouse_x
        ),
        s_row // 2
    )
    color = (242, 233, 78) if state.player == 1 else (218, 62, 82)
    pygame.draw.circle(app.window.window, color, center, radius)
    pygame.draw.circle(
        app.window.window, 
        (color[0] * 0.90, color[1] * 0.90, color[2] * 0.90), 
        center, 
        radius * 0.85
    )

    draw_mcts_eval(app, root, cols * s_col + 10, s_row, move_count, player)

def draw_mcts_eval(app: App, root: MCTSNode, x: int, y: int, move_count: int, player: int) -> None:
    if app is None or root is None:
        return

    eval_child = lambda child: (
        (
            MCTS._convert_eval(child.reward / child.visits)
            if child.visits > 0 else
            float("inf")
        )
        if child.terminal == -1 
        else MCTS._convert_terminal(child.terminal)
    )

    width: int = int(0.99 * app.window.scale * app.window.info.current_w - x)
    height: int = int(app.window.scale * app.window.info.current_h) // 12

    for i, child in enumerate(sorted(root.get_children(), key = eval_child, reverse = True)):
        rect: pygame.Rect = pygame.Rect(
            x + (int(app.window.scale * app.window.info.current_w) - x - width) // 2, y + i * (height + 3), width, height
        )
        eval_rect: pygame.Rect = pygame.Rect(
            x + int(0.2 * height), y + i * (height + 3) + int(0.05 * height), width // 6, int(0.9 * height)
        )

        eval: float = eval_child(child)  * (1 if player == 1 else -1)
        text: str = (
            (f"{eval:.2f}" if abs(eval) < float("inf") else "U") 
            if child.terminal == -1
            else ("W" if child.terminal != .5 else "D")
        )
        eval_text = app.font.render(
            text,
            True,
            (255, 255, 255)
        )

        best_path: str = ""
        node: MCTSNode = child
        best_path_text = app.font.render(
            best_path,
            True,
            (255, 255, 255)
        )

        while True:
            if node.s_children == 0:
                break

            best_path += f"{move_count + node.depth // 2}. "

            if node.depth == 1 and player == 2:
                best_path += ".. "
            elif node.s_children > 0:
                best_path += f"{chr(ord('A') + node.action)} "
                node = max(node.get_children(), key = eval_child)

            if node.s_children > 0:
                best_path += f"{chr(ord('A') + node.action)} "
                node = max(node.get_children(), key = eval_child)

            best_path_text = app.font.render(
                best_path,
                True,
                (255, 255, 255)
            )

            if eval_rect.x + eval_rect.width + best_path_text.get_rect().width >= rect.x + int(0.85 * rect.width) or node.s_children == 0:
                break

        pygame.draw.rect(
            app.window.window,
            (19, 75, 112),
            rect,
            border_radius = 10
        )
        pygame.draw.rect(
            app.window.window,
            (242, 233, 78) if eval >= 0 else (218, 62, 82),
            eval_rect, 
            border_radius = 5
        )
        app.window.window.blit(
            eval_text, 
            eval_text.get_rect(
                center = (
                    eval_rect.x + eval_rect.width // 2, 
                    eval_rect.y + eval_rect.height // 2
                )
            )
        )
        app.window.window.blit(
            best_path_text, 
            best_path_text.get_rect(
                center = (
                    eval_rect.x + eval_rect.width + 10 + best_path_text.get_rect().width // 2,
                    rect.y + rect.height // 2
                )
            )
        )

def player_action(app: App, state: Connect4Board) -> int:
    cols: int  = state.cols
    s_col: int = int(app.window.scale * app.window.info.current_h) // cols
    x: int     = 0

    if not 0 <= app.mouse_click[0] - x <= s_col * cols:
        return None

    col: int = int(app.mouse_click[0] - x) // s_col

    rows: int = state.rows
    for i in range(rows):
        if state.get_piece(rows - 1 - i, col) == 0:
            return col

    return None

def pick_action(app: App, state: Connect4Board, root: Optional[MCTSNode]) -> tuple[int, Optional[MCTSNode]]:
    action: int = None
    
    match app.gamemode:
        case 1:
            if app.mouse_click is not None:
                action = player_action(app, state)

                if root is not None:
                    MCTS.clear_tree(root)
                    root = None

                _, root = MCTS.mcts(
                    root_state = state, 
                    world = Connect4, 
                    s_rollout = int(1e3), 
                    max_expansion = 7, 
                    tree = True
                )

        case 2:
            if state.player == 1 and app.mouse_click is not None:
                action = player_action(app, state)

                if root is not None:
                    MCTS.clear_tree(root)
                    root = None


                _, root = MCTS.mcts(
                    root_state = state, 
                    world = Connect4, 
                    s_rollout = int(1e3), 
                    max_expansion = 7, 
                    tree = True
                )


            elif state.player == 2 and app.opponent == 1:
                if root is not None:
                    MCTS.clear_tree(root)
                    root = None

                action, root = MCTS.mcts(
                    root_state = state, 
                    world = Connect4, 
                    s_rollout = int(1e3), 
                    max_expansion = 7, 
                    tree  = True,
                    timer = True
                )

            elif state.player==2 and app.opponent==2:

                if root is not None:
                    MCTS.clear_tree(root)
                    root = None

                _, root = MCTS.mcts(
                    root_state = state, 
                    world = Connect4, 
                    s_rollout = int(1e3), 
                    max_expansion = 7, 
                    tree  = True,
                    timer = True
                )

                dt,_=load_tree("tree_weights")

                full_state = state.get_full_state() 
                action = int(predict(dt, sample=np.array(full_state)))



        case 3:
            if (state.player==1 and app.opponent==2) or (state.player==2 and app.opponent==1):
                if root is not None:
                    MCTS.clear_tree(root)
                    root = None

                _, root = MCTS.mcts(
                    root_state = state, 
                    world = Connect4, 
                    s_rollout = int(1e3), 
                    max_expansion = 7, 
                    tree  = True,
                    timer = False
                )

                dt,_=load_tree("tree_weights")

                full_state = state.get_full_state() 
                action = int(predict(dt, sample=np.array(full_state)))

            elif (state.player==2 and app.opponent==2) or (state.player==1 and app.opponent==1):
                if root is not None:
                    MCTS.clear_tree(root)
                    root = None

                action, root = MCTS.mcts(
                    root_state = state, 
                    world = Connect4, 
                    s_rollout = int(1e3), 
                    max_expansion = 7, 
                    tree  = True,
                    timer = False
                )

    return action, root

def game(app: App, state: Connect4Board, root: Optional[MCTSNode]) -> tuple[Connect4Board, int, Optional[MCTSNode]]:
    action, root = pick_action(app, state, root)
    if not type(action) == int:
        return state, 0, None
    state = Connect4.play(state, action)
    return state, Connect4.check_result(state, action), root

def main() -> None:
    pygame.init()
    app: App = App()

    state: Connect4Board = None
    root: MCTSNode = None
    move_count: int = 0

    outcome: int = 0
    timer_start = None
    duration = 3

    while app.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                app.running = False
            if event.type == pygame.KEYDOWN:
                # if event.key == pygame.K_F11:
                    # app.window.resize(1.00 if app.window.scale < 1.00 else 0.75)
                if event.key == pygame.K_ESCAPE:
                    app.gamemode = 0
            if event.type == pygame.MOUSEBUTTONDOWN:
                app.mouse_click = event.pos

        if app.gamemode > 0 and app.opponent > 0:
            if app.game_start:
                app.game_start = False
                state: Connect4Board = Connect4.init_board(6, 7)
                app.in_game = True

            if app.in_game:
                draw_game(app, state, root, move_count, Connect4.reverse_player(state.player))
                pygame.display.flip()

                state, result, new_root = game(app, state, root)
                if new_root is not None:
                    root = new_root
                    move_count += 1 if state.player == 2 else 0

                app.in_game = result == 0
                current_task = None

                app.mouse_click = None

            else:
                if result == 2:
                    timer_start = time.time()
                    outcome = Connect4.reverse_player(state.player)
                elif result > 0:
                    timer_start = time.time()
                    outcome = 3

                if timer_start != None and time.time() - timer_start <= duration:
                    draw_game(app, state, root, move_count, Connect4.reverse_player(state.player))
                    draw_result(app, outcome)
                else:
                    outcome = 0
                    app.gamemode = 0
                    app.opponent = 0

                result = 0
                pygame.display.flip()

        else:
            if app.gamemode == 0:
                app.gamemode = draw_menu(app)
                app.game_start = app.gamemode > 0
            elif app.opponent == 0:
                app.opponent = pick_opponnent(app) if app.gamemode > 1 else 1
            app.mouse_click = None

            pygame.display.flip()

    pygame.quit()
