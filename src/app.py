from mcts import *
from lib.connect4 import Connect4, Board
from lib.mcts import MCTS
from window import Window

import pygame
import asynio

class App:
    def __init__(self) -> None:
        self.window:      Window          = Window(0.75)
        self.running:     bool            = True
        self.game_start:  bool            = False
        self.gamemode:    int             = 0
        self.opponent:    int             = 0
        self.mouse_click: tuple[int, int] = None 

        button_pvp: pygame.Rect = pygame.Rect(
            window.scale * window.info.current_width // 4,
            (0.5 - 0.2 - 0.2 / 2 - 0.05) * window.scale * window.info.current_height,
            # half the screen minus the button height minus half the center button height
            window.scale * window.info.current_width // 2,
            0.2 * window.scale * window.info.current_height
        )
        button_pvb: pygame.Rect = pygame.Rect(
            window.scale * window.info.current_width // 4,
            (0.5 - 0.2 / 2) * window.scale * window.info.current_height,
            # half the screen minus the half center button height
            window.scale * window.info.current_width // 2,
            0.2 * window.scale * window.info.current_height
        )
        button_bvb: pygame.Rect = pygame.Rect(
            window.scale * window.info.current_width // 4,
            (0.5 + 0.2 / 2 + 0.05) * window.scale * window.info.current_height,
            # half the screen plus half the center button height
            window.scale * window.info.current_width // 2,
            0.2 * window.scale * window.info.current_height
        )

        font = pygame.font.Font(None, window.scale * window.info.current_width // 25)
        text_pvp = font.render("P VS P", True, (255, 255, 255))
        text_pvb = font.render("P VS B", True, (255, 255, 255))
        text_bvb = font.render("B VS B", True, (255, 255, 255))

        self.menu_objects = [
            (button_pvp, text_pvp),
            (button_pvb, text_pvb),
            (button_bvb, text_bvb)
        ]

        button1: pygame.Rect = pygame.Rect(
            window.scale * window.info.current_width // 4,
            (0.5 - 0.2 - 0.05) * window.scale * window.info.current_height,
            # half the screen minus the button height minus half the center button height
            window.scale * window.info.current_width // 2,
            0.2 * window.scale * window.info.current_height
        )
        button2: pygame.Rect = pygame.Rect(
            window.scale * window.info.current_width // 4,
            (0.5 - 0.2 / 2 + 0.05) * window.scale * window.info.current_height,
            # half the screen minus the half center button height
            window.scale * window.info.current_width // 2,
            0.2 * window.scale * window.info.current_height
        )

        font = pygame.font.Font(None, window.scale * window.info.current_width // 25)
        text_mcts = font.render("Monte Carlo", True, (255, 255, 255))
        text_dt = font.render("Decision Tree", True, (255, 255, 255))
        text_mcts_vs_dt = font.render("Monte Carlo VS Decision Tree", True, (255, 255, 255))
        text_dt_vs_mcts = font.render("Decision Tree VS Monte Carlo", True, (255, 255, 255))

        self.pick_opponnent_objects = (
            [button1, button2],
            [text_mcts, text_dt, text_mcts_vs_dt, text_dt_vs_mcts]
        )


def draw_base(window: Window) -> None:
    color: tuple[int, int, int] = (184, 219, 217)
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
        if rect.collidepoint(app.mouse_click):
            return i + 1
    return 0

def pick_opponnent(app: App) -> int:
    draw_base(app.window)
    for i, rect in enumerate(app.pick_opponnent_objects[0]):
        pygame.draw.rect(app.window.window, (47, 69, 80), rect)
        app.window.window.blit(
            app.pick_opponnent_objects[1][i + 2 if app.gamemode == 3 else 0],
            text.get_rect(center = (rect.x + rect.width // 2, rect.y + rect.height // 2))
        )
        if rect.collidepoint(app.mouse_click):
            return i + 1
    return 0

def draw_result(app: App) -> None:
    pass

def draw_game(window: Window, state: Board, is_player1: bool) -> None:
    draw_base(window)

    rows, cols = state.rows, state.cols
    s_row, s_col = (
        window.scale * window.info.current_height // (rows + 1), # extra row for piece to be placed
        window.scale * window.info.current_height // cols        # square board
    )

    base_rect: pygame.Rect = pygame.Rect(
        int(0.5 * window.scale * window.info.current_height - 0.5 * cols * s_col),
        s_row,
        cols * s_col,
        rows * s_row
    )
    pygame.draw.rect(window.window, (47, 69, 80), base_rect)

    color: tuple[int, int, int] = (184, 219, 217)
    radius: int = int(0.90 * min(s_row, s_col))

    for i in range(rows):
        for j in range(cols):
            if state.board[i][j] == 1:
                color = (242, 233, 78)
            elif state.board[i][j] == 2:
                color = (218, 62, 82)

            center: tuple[int, int] = (
                j * (s_col + 1) + s_col // 2, i * s_row + (3 * s_row) // 2
            )
            pygame.draw.circle(window.window, color, center, radius)

    mouse_x, _ = pygame.mouse.get_pos()
    center: tuple[int, int] = (
        base_rect.x + radius 
        if mouse_x % s_col - radius < base_rect.x 
        else (
            base_rect.x + base_rect.width - radius
            if mouse_x % s_col + radius > base_rect.x + base_rect.width
            else mouse_x % s_col + radius
        ),
        s_row // 2
    )
    color = (242, 233, 78) if is_player1 else (218, 62, 82)
    pygame.draw.circle(window.window, color, center, radius)

async def pick_action(app: App, state: Board, is_player1: bool) -> int:
    action: int = None

    match app.gamemode:
        case 0:
            if app.mouse_click != None:
                cols: int  = state.cols
                s_col: int = app.window.scale * app.window.info.current_height // cols
                x: int     = int(
                    0.5 * app.window.scale * app.window.info.current_height - 0.5 * cols * s_col
                )

                if app.mouse_click.x - x <= s_col * cols:
                    return None

                action = app.mouse_click % s_col

                rows: int = state.rows
                for j in range(rows):
                    if state.board[rows - 1 - j] == 0:
                        return action

                return None

        case 1: pass
        case 2: pass

    return action

async def game(app: App, state: Board, is_player1: bool) -> tuple[Board, int]:
    action: int = await pick_action(app, state, is_player1)
    state = Connect4.play(state, (action, 1 if is_player1 else 2))
    return state, Connect4.check_result(state, action)

async def main() -> None:
    app: App = App()

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
                is_player1: bool = True
                state: Board = Connect4.init_board(7, 7)

            task_game = asynio.create_task(game(state, app.mouse_click, app.opponent, is_player1))
            state, result = await task_game
            in_game = result == 0

            is_player1 = not is_player1
            app.mouse_click = None

            if in_game:
                draw_game(app.window, state, is_player1)
            else:
                draw_result(app, result)

        else:
            if app.gamemode == 0:
                app.gamemode = draw_menu(app)
                app.game_start = app.gamemode > 0
            else:
                app.opponent = pick_opponnent(app) if app.gamemode > 1 else 0
            app.mouse_click = None

        pygame.display.flip()

    pygame.quit()
