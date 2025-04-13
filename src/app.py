from lib.connect4 import Connect4, Connect4Board
from lib.mcts import MCTS
from src.window import Window

import pygame
import asyncio
import time

class App:
    def __init__(self) -> None:
        self.window:      Window          = Window(0.90)
        self.running:     bool            = True
        self.game_start:  bool            = False
        self.gamemode:    int             = 0
        self.opponent:    int             = 0
        self.mouse_click: tuple[int, int] = None

        font = pygame.font.Font(None, int(self.window.scale * self.window.info.current_w) // 25)

        esc: pygame.Rect = pygame.Rect(
            int(0.895 * self.window.scale * self.window.info.current_w),
            int(0.90 * self.window.scale * self.window.info.current_h),
            # half the screen minus the button height minus half the center button height
            int(self.window.scale * self.window.info.current_w) // 10,
            int(self.window.scale * self.window.info.current_h) // 10
        )
        text_esc = font.render("Menu", True, (255, 255, 255))
        self.esc_button = (esc, text_esc)

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

def draw_base(window: Window) -> None:
    color: tuple[int, int, int] = (147, 175, 174)
    window.window.fill(color)

def draw_menu(app: App) -> int:
    draw_base(app.window)
    for i, (rect, text) in enumerate(app.menu_objects):
        pygame.draw.rect(app.window.window, (47, 69, 80), rect)
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
        pygame.draw.rect(app.window.window, (47, 69, 80), rect)
        text = app.pick_opponnent_objects[1][i + (2 if app.gamemode == 3 else 0)]
        app.window.window.blit(
            text,
            text.get_rect(center = (rect.x + rect.width // 2, rect.y + rect.height // 2))
        )
        if app.mouse_click != None and rect.collidepoint(app.mouse_click[0], app.mouse_click[1]):
            return i + 1
    return 0

def draw_result(app: App, result: int) -> None:
    draw_base(app.window)
    label = app.player1_win if result == 1 else (app.player2_win if result == 2 else app.draw)
    app.window.window.blit(
        label,
        label.get_rect(
            center = (
                (app.window.info.current_w - label.get_rect().width) // 2,
                (app.window.info.current_h - label.get_rect().height) // 2
            )
        )
    )

def draw_game(app: App, state: Connect4Board) -> None:
    draw_base(app.window)

    pygame.draw.rect(app.window.window, (47, 69, 80), app.esc_button[0])
    app.window.window.blit(
        app.esc_button[1],
        app.esc_button[1].get_rect(
            center = (
                (app.esc_button[0].x + app.esc_button[0].width - app.esc_button[1].get_rect().width) // 2,
                (app.esc_button[0].y + app.esc_button[0].height - app.esc_button[1].get_rect().height) // 2
            )
        )
    )

    rows, cols = state.rows, state.cols
    s_row, s_col = (
        int(app.window.scale * app.window.info.current_h) // (rows + 1), # extra row for piece to be placed
        int(app.window.scale * app.window.info.current_h) // cols        # square board
    )

    base_rect: pygame.Rect = pygame.Rect(
        int(app.window.scale * app.window.info.current_w - cols * s_col) // 2,
        s_row,
        cols * s_col,
        rows * s_row
    )
    pygame.draw.rect(app.window.window, (47, 69, 80), base_rect)

    app.esc_button[0].y = base_rect.y + base_rect.height - app.esc_button[0].height
    pygame.draw.rect(app.window.window, (47, 69, 80), app.esc_button[0])
    app.window.window.blit(
        app.esc_button[1],
        app.esc_button[1].get_rect(
            center = (
                app.esc_button[0].x + app.esc_button[0].width // 2,
                app.esc_button[0].y + app.esc_button[0].height // 2
            )
        )
    )

    color: tuple[int, int, int] = (184, 219, 217)
    radius: int = int(0.90 * min(s_row, s_col)) // 2

    for i in range(rows):
        for j in range(cols):
            if state.get_piece(i, j) == 1:
                color = (242, 233, 78)
            elif state.get_piece(i, j) == 2:
                color = (218, 62, 82)
            else:
                color = (147, 175, 174)

            center: tuple[int, int] = (
                base_rect.x + j * (s_col + 1) + s_col // 2, base_rect.y + i * s_row + s_row // 2
            )
            pygame.draw.circle(app.window.window, color, center, radius)

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

def player_action(app: App, state: Connect4Board) -> int:
    cols: int  = state.cols
    s_col: int = int(app.window.scale * app.window.info.current_h) // cols
    x: int     = int(
        int(app.window.scale * app.window.info.current_w - cols * s_col) // 2
    )

    if not 0 <= app.mouse_click[0] - x <= s_col * cols:
        return None

    col: int = int(app.mouse_click[0] - x) // s_col

    rows: int = state.rows
    for i in range(rows):
        if state.get_piece(rows - 1 - i, col) == 0:
            return col

    return None

async def pick_action(app: App, state: Connect4Board) -> int:
    action: int = None
    match app.gamemode:
        case 1:
            if app.mouse_click is not None:
                action = player_action(app, state)
        case 2:
            if state.player == 1 and app.mouse_click is not None:
                action = player_action(app, state)
            elif state.player == 2 and app.opponent == 1:
                mcts_choice = asyncio.create_task(
                    MCTS.mcts(state, Connect4, int(1e5), 7, debug = True, timer = True)
                )
                action = await mcts_choice
        case 3:
            mcts_choice = await asyncio.create_task(
                MCTS.mcts(state, Connect4, int(1e5), 7, debug = True, timer = False)
            )
            action = await mcts_choice
    return action

async def game(app: App, state: Connect4Board) -> tuple[Connect4Board, int]:
    action: int = await pick_action(app, state)
    if not type(action) == int:
        return state, 0
    state = Connect4.play(state, action)
    return state, Connect4.check_result(state, action)

async def main() -> None:
    pygame.init()
    app: App = App()

    state: Connect4Board = None

    outcome: int = 0
    timer_start = None
    duration = 3

    while app.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                app.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F11:
                    app.window.resize(1.00 if app.window.scale < 1.00 else 0.75)
            if event.type == pygame.MOUSEBUTTONDOWN:
                app.mouse_click = event.pos

        if app.gamemode > 0 and app.opponent > 0:
            if app.game_start:
                app.game_start = False
                state: Connect4Board = Connect4.init_board(6, 7)
                app.in_game = True

            app.in_game = app.in_game and not (
                    app.mouse_click != None 
                and app.esc_button[0].collidepoint(app.mouse_click[0], app.mouse_click[1])
            )

            if app.in_game:
                current_task = asyncio.create_task(game(app, state))
                state, result = await current_task
                app.in_game = result == 0
                current_task = None

                app.mouse_click = None
                draw_game(app, state)

            else:
                if result == 2:
                    timer_start = time.time()
                    outcome = Connect4.reverse_player(state.player)
                elif result > 0:
                    timer_start = time.time()
                    outcome = 3

                if timer_start != None and time.time() - timer_start <= duration:
                    draw_result(app, outcome)
                else:
                    outcome = 0
                    app.gamemode = 0

                result = 0

        else:
            if app.gamemode == 0:
                app.gamemode = draw_menu(app)
                app.game_start = app.gamemode > 0
            else:
                app.opponent = pick_opponnent(app) if app.gamemode > 1 else 1
            app.mouse_click = None

        pygame.display.flip()

    pygame.quit()
