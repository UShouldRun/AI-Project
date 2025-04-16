from random import randint, choice
from abc import ABC, abstractmethod
from math import sqrt, log
from typing import TypeVar, List, Optional
from timeit import default_timer

import numpy as np
import asyncio

State  = TypeVar("State")
Action = TypeVar("Action")

class MCTSInterface(ABC):
    @staticmethod
    @abstractmethod
    def play(state: State, action: Action) -> State:
        """Executes the action on the given state and returns the rewarding state."""
        pass

    @staticmethod
    @abstractmethod
    def get_actions(state: State) -> List[Action]:
        """Returns a List of valid actions for the given state."""
        pass

    @staticmethod
    @abstractmethod
    def is_terminal_state(state: State, action: Action) -> bool:
        """Checks if the state is terminal (i.e., no further actions possible)."""
        pass

    @staticmethod
    @abstractmethod
    def value(state: State, action: Action, player: Optional[int] = None) -> float:
        """Returns the value of the given state (e.g., score or utility). Should be in the interval [0,1]."""
        pass

    @staticmethod
    @abstractmethod
    def heuristic(state: State, player: int) -> float:
        """Returns a heuristic value of the given state (e.g., score or utility). Should be in the interval [0,1]."""
        pass

    @staticmethod
    @abstractmethod
    def get_current_player(state: int) -> int:
        pass

    @staticmethod
    @abstractmethod
    def reverse_player(player: int) -> int:
        pass

    @staticmethod
    @abstractmethod
    def copy(state: State) -> State:
        """Creates and returns a copy of the given state."""
        pass

    @staticmethod
    @abstractmethod
    def print(state: State) -> None:
        pass

class MCTSNode:
    __slots__ = (
        'state', 'action', 'parent', 'children', 
        's_children', 'max_children',
        'depth', 'reward', 'visits', 'terminal', 's_t_actions', 't_actions'
    )
    
    def __init__(self, state: State, action: Action, parent: Optional["MCTSNode"], max_children: Optional[int] = None) -> None:
        self.state: State = state
        self.action: Action = action
        self.parent: Optional["MCTSNode"] = parent

        assert max_children is not None or parent is not None
        self.max_children: int = max_children if parent is None else parent.max_children
        self.s_children: int = 0
        self.children: np.ndarray = np.empty(self.max_children, dtype = np.object_)

        self.depth: int = 0 if parent is None else parent.depth + 1
        self.reward: float = 0
        self.visits: int = 0
        self.terminal: float = -1
        self.s_t_actions: int = -1
    
    def is_root(self) -> bool:
        return self.parent is None
    
    def is_leaf(self) -> bool:
        return self.s_children == 0

    def has_undetermined_child(self) -> bool:
        return any(child.terminal == -1 for child in self.get_children())
    
    def add_child(self, child: "MCTSNode") -> None:
        """Add a child node, resizing the array if necessary."""
        assert self.s_children <= self.max_children
        self.children[self.s_children] = child
        self.s_children += 1

    def remove_children(self) -> None:
        self.children     = np.empty(0, dtype = np.object_)
        self.s_children   = 0
        self.max_children = 0
    
    def get_children(self) -> np.ndarray:
        """Get the actual children (not the empty slots)."""
        return self.children[:self.s_children]

    def get_leafs(self) -> np.ndarray["MCTSNode"]:
        capacity: int = 32
        result: np.ndarray["MCTSNode"] = np.empty(capacity, dtype = object)
        stack: np.ndarray["MCTSNode"] = np.empty(capacity, dtype = object)

        result_size: int = 0
        stack_size: int = 1
        stack[0] = self

        while stack_size > 0:
            stack_size -= 1
            current = stack[stack_size]

            if current.is_leaf():
                s_result: int = len(result)
                if result_size >= s_result:
                    result = np.resize(result, 2 * s_result)
                result[result_size] = current
                result_size += 1

            else:
                for i in range(current.s_children):
                    s_stack: int = len(stack)
                    if stack_size >= s_stack:
                        stack = np.resize(stack, 2 * s_stack)
                    stack[stack_size] = current.children[i]
                    stack_size += 1

        return result[:result_size]

class MCTS:
    @staticmethod
    def _encapsulate(state: State, action: Action, max_expansion: int) -> MCTSNode:
        return MCTSNode(state, action, None, max_expansion)

    @staticmethod
    def _is_terminal_state(node: MCTSNode) -> bool:
        return node.action != None and node.terminal > -1

    @staticmethod
    def _inverse_sigmoid(x: float) -> float:
        if x == 1: return float("inf")
        if x == 0: return -float("inf")
        return log(x / (1 - x))

    @staticmethod
    def _convert_eval(value: float) -> float:
        """The sigmoid function maps the real line to the interval [0,1], therefore it's inverse, which exists does the opposite."""
        return MCTS._inverse_sigmoid(value)

    @staticmethod
    def _convert_terminal(value: int) -> float:
        return -float("inf") if value == 0 else (0 if value == 0.5 else float("inf"))

    @staticmethod
    def _sort_children(root: MCTSNode) -> None:
        eval = lambda node: (
            (
                MCTS._convert_eval(node.reward / node.visits)
                if node.visits > 0
                else 1000
            )
            if node.terminal == -1
            else MCTS._convert_terminal(node.terminal)
        )

        for i in range(1, root.s_children):
            key_node: MCTSNode = root.children[i]
            key_value: float = eval(key_node)
            j: int = i - 1

            while j >= 0:
                j_node: MCTSNode = root.children[j]
                j_value: float = eval(j_node) 

                if key_value <= j_value:
                    break

                root.children[j + 1] = root.children[j]
                j -= 1

            root.children[j + 1] = key_node

    @staticmethod
    def _pick_action(root: MCTSNode) -> Action:
        assert not root.is_leaf()
        return root.children[0].action

    @staticmethod
    def _only_action(node: MCTSNode, world: MCTSInterface, c: float) -> Action:
        assert node is not None

        for child in node.get_children():
            MCTS._expand(child, world)

        s_children: int = node.s_children
        s_non_losing: int = 0
        index: int = 0

        for i in range(s_children):
            child: MCTSNode = node.children[i]
            if child.terminal == -1 or child.terminal > 0:
                s_non_losing += 1
                index = i

        if s_non_losing > 1:
            return None
        if s_non_losing == 1:
            return node.children[index].action
        return node.children[randint(0, s_children - 1)].action

    @staticmethod
    def _random_rollout(tree: MCTSNode, world: MCTSInterface, heuristic: tuple[bool, int], n: int = 100) -> None:
        """Performs a random rollout starting from the given tree."""
        assert n > 0, f"Invalid number of random rollouts: n = {n}"

        max_leafs: int = tree.max_children ** 2
        leafs: np.ndarray[MCTSNode] = np.ndarray(max_leafs, dtype = np.object_)
        for _ in range(n):
            s_leafs: int = 0

            for leaf in tree.get_leafs():
                if MCTS._is_terminal_state(leaf):
                    continue
                if s_leafs >= max_leafs:
                    max_leafs *= 2
                    leafs = np.resize(leafs, max_leafs)
                leafs[s_leafs] = leaf
                s_leafs += 1

            if s_leafs == 0:
                break

            node: MCTSNode = leafs[randint(0, s_leafs - 1)]
            if not MCTS._expand(node, world):
                continue

            for child in node.get_children():
                MCTS._rollout(child, world, heuristic)

    @staticmethod
    def _evaluate(node: MCTSNode, c: float) -> float:
        """Evaluates a node using the UCT formula."""
        assert node.parent != None and not MCTS._is_terminal_state(node)
        return node.reward / node.visits + c * sqrt(log(node.parent.visits) / node.visits) \
            if node.visits > 0 and node.parent.visits >= 1\
            else float("inf")

    @staticmethod
    def _select(node: MCTSNode, world: MCTSInterface, c: float) -> Optional[MCTSNode]:
        """Selects the best child node using the UCT formula."""
        while not node.is_leaf():
            best_child_idx: int = 0
            while node.children[best_child_idx].terminal != -1:
                best_child_idx += 1
                if best_child_idx >= node.s_children:
                    return None

            best_score: float = MCTS._evaluate(node.children[best_child_idx], c)
            
            for idx in range(best_child_idx + 1, node.s_children):
                child: MCTSNode = node.children[idx]
                if child.terminal != -1:
                    continue

                score: float = MCTS._evaluate(child, c)
                if score > best_score:
                    best_score = score
                    best_child_idx = idx
            
            node = node.children[best_child_idx]

        return node

    @staticmethod
    def _expand(node: MCTSNode, world: MCTSInterface) -> bool:
        """Expands the node by generating all possible children."""
        assert node.is_leaf() and not MCTS._is_terminal_state(node)
        
        max_terminal: float = -1
        for action in world.get_actions(node.state):
            child: MCTSNode = MCTSNode(
                world.play(world.copy(node.state), action),
                action,
                node
            )
            node.add_child(child)

            child.terminal = world.value(child.state, child.action)
            if child.terminal == 1:
                MCTS._backpropagate_terminal(node, 0)
                return False

            max_terminal = max(max_terminal, child.terminal)

        if max_terminal > -1:
            MCTS._backpropagate_terminal(node, 1 - max_terminal)

        return node.s_children > 0

    @staticmethod
    def _rollout(
        leaf: MCTSNode, world: MCTSInterface, heuristic: tuple[bool, int], timer: bool = False
    ) -> tuple[int, Optional[List[float]]]:
        """Simulates a random rollout from the given leaf node."""
        assert leaf != None and leaf.is_leaf(), f"leaf: {leaf}"
        assert not MCTS._is_terminal_state(leaf)

        # times: Optional[List[float]] = None
        # if timer:
        # start = default_timer()
        # times = 3 * [0]

        state:  State  = world.copy(leaf.state)
        action: Action = None
        value:  float  = -1
        player: int    = world.get_current_player(leaf.parent.state)
        depth:  int    = leaf.depth

        while (not heuristic[0] or depth <= heuristic[1]) and value == -1:
            action = choice(world.get_actions(state))
            state  = world.play(state, action)
            value  = world.value(state, action, player)
            depth += 1

            # if timer:
            # times[1] = (times[0] * times[1] + default_timer() - start) / (times[0] + 1)
            # times[0] += 1
            # start = default_timer()

        if value == -1:
            value = world.heuristic(leaf.state, player)

        MCTS._backpropagate(leaf, value)
        # if timer:
        # times[2] = default_timer() - start

        return depth, None 

    @staticmethod
    def _backpropagate(node: MCTSNode, reward: float) -> None:
        """Backpropagates the reward from the a node up to the root."""
        assert node != None
        assert 0 <= reward <= 1

        while node != None:
            node.visits += 1
            node.reward += reward
            reward = 1 - reward
            node = node.parent

    @staticmethod
    def _backpropagate_terminal(node: MCTSNode, terminal: float) -> None:
        """Backpropagates the terminal value from the a node up to the root."""
        assert node != None
        assert 0 <= terminal <= 1
        s_actions: int = 0

        while node != None and (terminal == 0 or not node.has_undetermined_child()):
            node.terminal    = terminal
            node.s_t_actions = s_actions
            s_actions += 1

            if node.s_children > 0 and node.depth > 0:
                node.remove_children()

            terminal = 1 - terminal
            node = node.parent

    @staticmethod
    def _print_node(node: MCTSNode, world: MCTSInterface, c: float) -> None:
        eval_child = lambda child: (
            (
                MCTS._convert_eval(child.reward / child.visits)
                if child.visits > 0 else
                float("inf")
            )
            if child.terminal == -1 
            else MCTS._convert_terminal(child.terminal)
        )

        print("Node {")
        print(f"  depth = {node.depth},")
        print(f"  visits = {node.visits},")
        print(f"  terminal = {node.terminal},")
        print(f"  s_t_actions = {node.s_t_actions},")

        print("  state =")
        world.print(node.state)

        print("  Children: {")
        for child in sorted(node.get_children(), key = eval_child, reverse = True):
            print(f"    action: {child.action}, visits: {child.visits}, value: {eval_child(child):.3f}, terminal: {child.terminal}")
        print("  }\n}")

    @staticmethod
    def _print_timer(timer_array: List[float], rollout_timer: List[float], max_depth: int) -> None:
        total_execution_time: float = sum(timer_array[0] * timer_array[i] for i in range(1, 4))
        print(f"Total execution time: {total_execution_time:.6f} seconds")

        print(f"Random rollout execution time: {timer_array[8]:.6f} seconds")

        selection_avg_time: float = timer_array[1] * 1e6
        selection_total_time: float = timer_array[0] * timer_array[1]
        print(f"Selection execution time: ")
        print(f"  - Average: {selection_avg_time:.6f} μs")
        print(f"  - Total: {selection_total_time:.3f} seconds")

        expansion_avg_time: float = timer_array[2] * 1e6
        expansion_total_time: float = timer_array[0] * timer_array[2]
        print(f"Expansion execution time: ")
        print(f"  - Average: {expansion_avg_time:.6f} μs")
        print(f"  - Total: {expansion_total_time:.3f} seconds")

        rollout_avg_time: float = timer_array[3] * 1e6
        rollout_total_time: float = timer_array[0] * timer_array[3]
        print(f"Rollout (max_depth = {max_depth}) execution time: ")
        print(f"  - Average: {rollout_avg_time:.6f} μs")
        print(f"  - Total: {rollout_total_time:.3f} seconds")

        if rollout_timer is not None:
            print(f"  - main loop: Average = {(timer_array[4] * 1e6):.6f} μs, Total = {timer_array[0] * timer_array[4]:.3f} s")
            print(f"  - backpropagation: Average = {(timer_array[5] * 1e6):.6f} μs, Total = {timer_array[0] * timer_array[5]:.3f} s")

    @staticmethod
    async def mcts(
        state: State, world: MCTSInterface, s_rollout: int, max_expansion: int = 10,
        s_initial_rollout: int = 100, c: float = round(sqrt(2), 3),
        debug: bool = False, timer: bool = False, heuristic: tuple[bool, Optional[int]] = (False, None)
    ) -> Action:
        """Performs the Monte Carlo Tree Search and returns the best action."""

        if timer:
            start = default_timer()
            # timer_array: List[float] = 9 * [0]
            # rollout_timer: List[float] = 4 * [0]

        tree: MCTSNode = MCTS._encapsulate(state, None, max_expansion)
        MCTS._expand(tree, world)
        MCTS._sort_children(tree)
        MCTS._print_node(tree, world, c)

        if tree.children[0].terminal == 1:
            if debug:
                print("Left winning_actions")
                MCTS._print_node(tree, world, c)

            return tree.children[0].action
        
        if (only_action := MCTS._only_action(tree, world, c)) != None:
            if debug:
                print("Left only_action or just lost")
                MCTS._print_node(tree, world, c)
            return only_action
        
        MCTS._random_rollout(tree, world, heuristic, s_initial_rollout)
        # if timer:
        # timer_array[8] = default_timer() - start

        max_depth: int = 0
        for _ in range(s_rollout):
            # if timer:
            # start = default_timer()

            MCTS._sort_children(tree)
            if tree.children[0].terminal == 1:
                break

            node: MCTSNode = MCTS._select(tree, world, c)
            if node == None:
                break
            if node.terminal == 1:
                break

                # if timer:
                # elapsed_time = default_timer() - start
                # timer_array[1] = (timer_array[0] * timer_array[1] + elapsed_time) / (timer_array[0] + 1)
                # start = default_timer()

            if node.visits > 0:
                expansion: bool = MCTS._expand(node, world)
                # if timer:
                # elapsed_time = default_timer() - start
                # timer_array[2] = (timer_array[0] * timer_array[2] + elapsed_time) / (timer_array[0] + 1)
                # start = default_timer()

                if not expansion:
                    continue
                node = node.children[randint(0, node.s_children - 1)]

            depth, _ = MCTS._rollout(node, world, heuristic, timer = False and timer)
            max_depth = max(max_depth, depth)

            # if timer:
            # elapsed_time = default_timer() - start
            # timer_array[3] = (timer_array[0] * timer_array[3] + elapsed_time) / (timer_array[0] + 1)
            # 
            # if rollout_timer is not None:
            # for i in range(2):
            # n: float = rollout_timer[0] if i == 0 else 1
                # timer_array[4 + i] = (
                # (timer_array[0] * timer_array[4 + i] + n * rollout_timer[i + 1]) / (timer_array[0] + 1)
            # )
            # 
            # timer_array[0] += 1

        if timer:
            print(f"Total execution time: {(default_timer() - start):.3f} seconds")
            # MCTS._print_timer(timer_array, rollout_timer, max_depth)
        if debug:
            print("Left normally")
            MCTS._print_node(tree, world, c)

        return MCTS._pick_action(tree)
