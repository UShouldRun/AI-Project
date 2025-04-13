from lib.mcts import MCTSInterface, Optional, List

class Connect4Board:
    def __init__(self, rows: int, cols: int) -> None:
        self.player:  int       = 1
        self.rows:    int       = rows
        self.cols:    int       = cols
        self.board1:  int       = 0
        self.board2:  int       = 0
        self.heights: List[int] = cols * [0]

    def place_piece(self, player: int, row: int, col: int) -> None:
        assert 0 <= row < self.rows and 0 <= col < self.cols
        assert player == 1 or player == 2
        if player == 1:
            self.board1 = self.board1 | 1 << (self.cols * row + col)
        else:
            self.board2 = self.board2 | 1 << (self.cols * row + col)

    def get_piece(self, row: int, col: int) -> int:
        assert 0 <= row < self.rows and 0 <= col < self.cols
        piece: int = self.board1 >> (self.cols * row + col) & 0b1
        if piece == 1:
            return 1
        piece = self.board2 >> (self.cols * row + col) & 0b1
        return 2 if piece == 1 else 0

class Connect4(MCTSInterface):
    @staticmethod
    def play(state: Connect4Board, action: int) -> Connect4Board: 
        assert not Connect4.is_out_of_bounds(state, 0, action), f"Invalid Action: {action}"

        col: int = action
        row: int = Connect4.action_get_row(state, col)
        assert row > -1, f"Invalid Action: {action}"

        player: int = state.player
        assert player == 1 or player == 2, f"Invalid player: {player}"

        state.place_piece(player, row, col)
        state.heights[col] += 1
        state.player = player % 2 + 1
        return state

    @staticmethod
    def get_actions(state: Connect4Board) -> List[int]:
        return [
            col 
            for col in range(state.cols) 
            if state.heights[col] < state.rows
        ]

    @staticmethod
    def is_terminal_state(state: Connect4Board, action: int) -> bool:
        return action != None and Connect4.value(state, action, state.player) > -1

    @staticmethod
    def value(state: Connect4Board, action: int, player: Optional[int] = None) -> float:
        result: int = Connect4.check_result(state, action)
        if result == 0:
            return -1
        if player is None:
            return result / 2
        assert player == 1 or player == 2
        return result / 2 if state.player != player else 1 - result / 2

    @staticmethod
    def heuristic(state: Connect4Board, player: int) -> float:
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
    def evaluate_line(line: List[int], player: int) -> int:
        player_count = line.count(player)
        opponent_count = line.count(player % 2 + 1)

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
    def get_current_player(state: Connect4Board) -> int:
        return state.player

    @staticmethod
    def reverse_player(player: int) -> int:
        assert player == 1 or player == 2
        return player % 2 + 1

    @staticmethod
    def copy(state: Connect4Board) -> Connect4Board:
        cp: Connect4Board = Connect4Board(state.rows, state.cols)
        cp.board1  = state.board1
        cp.board2  = state.board2
        cp.player  = state.player
        cp.heights = [height for height in state.heights]
        return cp

    @staticmethod
    def print(state: Connect4Board) -> None:
        for row in range(state.rows):
            print("  [", end = "")
            for col in range(state.cols):
                print(state.get_piece(row, col), " " if col < state.cols - 1 else "", end = "")
            print("]")

    @staticmethod
    def init_board(rows: int, cols: int) -> Connect4Board:
        return Connect4Board(rows, cols)
        
    @staticmethod
    def action_get_row(state: Connect4Board, col: int) -> int:
        return state.rows - 1 - state.heights[col]

    @staticmethod
    def is_out_of_bounds(state: Connect4Board, row: int, col: int) -> bool:
        return not (0 <= row < state.rows and 0 <= col < state.cols)

    @staticmethod
    def check_result(state: Connect4Board, action: int) -> int:
        col: int = action
        row: int = Connect4.action_get_row(state, col) + 1
        player: int = state.player % 2 + 1

        count: int = Connect4.count_in_direction(state, row, col - 1, 0, -1, player)
        if count >= 3:
            return 2 # win
        elif count + Connect4.count_in_direction(state, row, col + 1, 0, 1, player) >= 3:
            return 2

        count = Connect4.count_in_direction(state, row - 1, col - 1, -1, -1, player)
        if count >= 3:
            return 2 # win
        elif count + Connect4.count_in_direction(state, row + 1, col + 1, 1, 1, player) >= 3:
            return 2

        count = Connect4.count_in_direction(state, row - 1, col, -1, 0, player)
        if count >= 3:
            return 2 # win
        elif count + Connect4.count_in_direction(state, row + 1, col, 1, 0, player) >= 3:
            return 2

        count = Connect4.count_in_direction(state, row + 1, col - 1, 1, -1, player)
        if count >= 3:
            return 2 # win
        elif count + Connect4.count_in_direction(state, row - 1, col + 1, -1, 1, player) >= 3:
            return 2

        first_row: int = (1 << state.cols) - 1
        return 0 if (state.board1 | state.board2) & first_row != first_row else 1

    @staticmethod
    def count_in_direction(state: Connect4Board, start_row: int, start_col: int, drow: int, dcol: int, player: int) -> int:
        count: int = 0
        row, col = start_row, start_col

        board: Connect4Board = state.board1 if player == 1 else state.board2

        while 0 <= row < state.rows and 0 <= col < state.cols and (board >> (state.cols * row + col)) & 1 == 1:
            count += 1
            row += drow
            col += dcol

        return count
