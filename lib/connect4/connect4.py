from lib.mcts import MCTSInterface

class Board:
    def __init__(self, rows: int, cols: int) -> None:
        self.player: int = 1
        self.rows: int = rows
        self.cols: int = cols
        self.board: int = 0

    def place_piece(self, player: int, row: int, col: int) -> None:
        assert 0 <= row < self.rows and 0 <= col < self.cols
        assert player == 1 or player == 2
        self.board = self.board | player << 2 * (self.cols * row + col)

    def get_piece(self, row: int, col: int) -> int:
        assert 0 <= row < self.rows and 0 <= col < self.cols
        return self.board >> 2 * (self.cols * row + col) & 0b11

class Connect4(MCTSInterface):
    DIRS: list[tuple[int, int]] = [
        (-1,  0), # E
        (-1, -1), # NE
        ( 0, -1), # N
        ( 1, -1)  # NW
    ]

    @staticmethod
    def play(state: Board, action: int) -> Board: 
        assert not Connect4.is_out_of_bounds(state, 0, action), f"Invalid Action: {action}"

        col: int = action
        row: int = Connect4.action_get_row(state, col)
        assert row > -1, f"Invalid Action: {action}"

        player: int = state.player
        assert player == 1 or player == 2, f"Invalid player: {player}"

        state.place_piece(player, row, col)
        state.player = Connect4.reverse_player(player)
        return state

    @staticmethod
    def get_actions(state: Board) -> list[int]:
        rows: int = state.rows
        cols: int = state.cols

        moves: list[int] = []
        for col in range(cols):
            for row in range(rows):
                if state.get_piece(rows - 1 - row, col) == 0:
                    moves.append(col)
                    break
        return moves

    @staticmethod
    def is_terminal_state(state: Board, action: int) -> bool:
        return action != None and Connect4.value(state, action, state.player) > -1

    @staticmethod
    def value(state: Board, action: int, player: int) -> float:
        assert player == 1 or player == 2
        result: int = Connect4.check_result(state, action)
        if result == 0:
            return -1
        # inverse because if the resulting state is final, then the reverse player won or drew
        return (result / 2) if state.player != player else (1 - result / 2)

    @staticmethod
    def heuristic(state: Board, player: int) -> float:
        score = 0

        for row in range(state.rows):
            for col in range(state.cols):
                if col + 3 < state.cols:
                    score += Connect4.evaluate_line(
                        [state.get_piece(row, col + i) for i in range(4)],
                        player
                    )
                if row + 3 < state.rows:
                    score += Connect4.evaluate_line(
                        [state.get_piece(row + i, col) for i in range(4)],
                        player
                    )
                if row + 3 < state.rows and col - 3 >= 0:
                    score += Connect4.evaluate_line(
                        [state.get_piece(row + i, col-i) for i in range(4)],
                        player
                    )
                if row + 3 < state.rows and col + 3 < state.cols:
                    score += Connect4.evaluate_line(
                        [state.get_piece(row + i, col + i) for i in range(4)],
                        player
                    )

        # Calculate the maximum possible score
        max_score = (state.rows * state.cols - 3) * 4 * 100  # Each line gives a max of 100
        return (score + max_score) / (2 * max_score)  # Normalize score to be between 0 and 1

    @staticmethod
    def evaluate_line(line: list[int], player: int) -> int:
        player_count = line.count(player)
        opponent_count = line.count(Connect4.reverse_player(player))

        if player_count == 4:
            return 100
        elif opponent_count == 4:
            return -100
        elif player_count == 3 and opponent_count == 0:
            return 10
        elif opponent_count == 3 and player_count == 0:
            return -10
        elif player_count == 2 and opponent_count == 0:
            return 1
        elif opponent_count == 2 and player_count == 0:
            return -1
        else:
            return 0

    @staticmethod
    def get_current_player(state: Board) -> int:
        return state.player

    @staticmethod
    def reverse_player(player: int) -> int:
        assert player == 1 or player == 2
        return 1 if player == 2 else 2

    @staticmethod
    def copy(state: Board) -> Board:
        rows, cols = state.rows, state.cols
        cp: Board = Board(rows, cols)
        cp.board = state.board
        cp.player = state.player
        return cp

    @staticmethod
    def print(state: Board) -> None:
        for row in range(state.rows):
            print("  [", end = "")
            for col in range(state.cols):
                print(state.get_piece(row, col), " " if col < state.cols - 1 else "", end = "")
            print("]")

    @staticmethod
    def init_board(rows: int, cols: int) -> Board:
        return Board(rows, cols)
        
    @staticmethod
    def action_get_row(state: Board, col: int) -> int:
        rows: int = state.rows
        for i in range(rows):
            if state.get_piece(rows - 1 - i, col) == 0:
                return rows - 1 - i
        return -1

    @staticmethod
    def is_out_of_bounds(state: Board, row: int, col: int) -> bool:
        return not (0 <= row < state.rows and 0 <= col < state.cols)

    @staticmethod
    def check_result(state: Board, action: int) -> int:
        player: int = Connect4.reverse_player(state.player)
        col: int    = action
        row: int    = Connect4.action_get_row(state, col) + 1

        for drow, dcol in Connect4.DIRS:
            if (Connect4.count_in_direction(state, row + drow, col + dcol,  drow,  dcol)
                + Connect4.count_in_direction(state, row - drow, col - dcol, -drow, -dcol)) >= 3:
                return 2 # win

        for col in range(state.cols):
            if state.get_piece(0, col) == 0:
                return 0 # not terminal state
        return 1 # draw

    @staticmethod
    def count_in_direction(state: Board, row: int, col: int, drow: int, dcol: int) -> int:
        count: int  = 0
        player: int = Connect4.reverse_player(state.player)
        while not Connect4.is_out_of_bounds(state, row, col) and state.get_piece(row, col) == player:
            count += 1
            row   += drow
            col   += dcol
        return count
