from mcts import MCTSInterface

class Board:
    def __init__(self, rows: int, cols: int) -> None:
        assert rows == cols
        self.rows: int = rows
        self.cols: int = cols
        self.board: list[list[int]] = [[0 for _ in range(cols)] for _ in range(rows)]

class Connect4(MCTSInterface):
    @staticmethod
    def play(state: Board, action: tuple[int, int]) -> Board: 
        assert not Connect4.is_out_of_bounds(state, 0, action[0]), f"Invalid Action: {action}"

        col: int = action[0]
        row: int = Connect4.action_get_row(state, col)
        assert row > -1, f"Invalid Action: {action}"

        turn: int = action[1]
        assert turn == 1 or turn == 2, f"Invalid Turn: {turn}"

        state.board[row][col] = turn
        return state

    @staticmethod
    def get_actions(state: Board) -> list[int]:
        assert Connect4.is_valid_state(state)
        rows: int = state.rows
        cols: int = state.cols

        moves: list[int] = []
        for col in range(cols):
            for row in range(rows):
                if state.board[rows - 1 - row][col] == 0:
                    moves.append(col)
                    break
        return moves

    @staticmethod
    def is_terminal_state(state: Board, action: tuple[int, int]) -> bool:
        return action == None or Connect4.value(state, action) > 0

    @staticmethod
    def value(state: Board, action: tuple[int, int]) -> float:
        result: int = check_result(state, action)
        assert result > 0
        return result / 2

    @staticmethod
    def copy(state: Board) -> Board:
        rows, cols = state.rows, state.cols
        cp: Board = Board(rows, cols)
        for row in range(rows):
            for col in range(cols):
                cp.board[row][col] = state.board[row][col]
        return cp

    @staticmethod
    def init_board(rows: int, cols: int) -> Board:
        return Board(rows, cols)
        
    @staticmethod
    def action_get_row(state: Board, col: int) -> int:
        rows: int = state.rows
        for i in range(rows):
            if state.board[rows - 1 - i][col] == 0:
                return rows - 1 - i
        return -1

    @staticmethod
    def is_out_of_bounds(state: Board, row: int, col: int) -> bool:
        return not (0 <= row < state.rows and 0 <= col < state.cols)

    @staticmethod
    def check_result(state: Board, action: tuple[int, int]) -> int:
        turn: int = action[1]
        col: int  = action[0]
        row: int  = -1

        rows: int = state.rows
        for i in range(rows):
            if i < rows - 1 and state.board[rows - 1 - (i + 1)][col] == 0:
                row = i
                break
        assert state.board[row][col] == turn

        dirs: list[tuple[int, int]] = [
            (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)
            #  E        NE        N       NW        W      SW      S       SE
        ]
        for drow, dcol in dirs:
            if Connect4.count_in_direction(state, row + drow, col + dcol, drow, dcol, turn) >= 3:
                return 2 # win

        cols: int = state.cols
        for col in range(cols):
            if state.board[0][col] == 0:
                return 0 # not terminal state
        return 1 # draw

    @staticmethod
    def count_in_direction(state: Board, row: int, col: int, drow: int, dcol: int, turn: int) -> int:
        count: int = 0
        while not Connect4.is_out_of_bounds(state, row, col) and state.board[row][col] == turn:
            count += 1
            row   += drow
            col   += dcol
        return count
