from random import randint, choice
from abc import ABC, abstractmethod
from math import sqrt, log
from typing import TypeVar, List, Optional
from timeit import default_timer
from itertools import repeat

import numpy as np

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
        'action',
        'parent', 'children', 's_children', 'max_children',
        'depth', 'reward', 'visits', 'terminal', 'undet_children'
    )
    
    def __init__(self, action: Action, parent: Optional["MCTSNode"], max_children: Optional[int] = None) -> None:
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
        self.undet_children: int = 0
    
    def is_root(self) -> bool:
        return self.parent is None
    
    def is_leaf(self) -> bool:
        return self.s_children == 0

    def has_undetermined_child(self) -> bool:
        return self.undet_children > 0
    
    def add_child(self, child: "MCTSNode") -> None:
        """Add a child node, resizing the array if necessary."""
        assert self.s_children <= self.max_children
        self.children[self.s_children] = child
        self.s_children += 1
        self.undet_children += 1

    def remove_children(self) -> None:
        self.children       = np.empty(0, dtype = np.object_)
        self.s_children     = 0
        self.max_children   = 0
        self.undet_children = 0
    
    def get_children(self) -> np.ndarray:
        """Get the actual children (not the empty slots)."""
        return self.children[:self.s_children]
    
class MCTS:
    @staticmethod
    def _encapsulate(action: Action, max_expansion: int) -> MCTSNode:
        return MCTSNode(action, None, max_expansion)

    @staticmethod
    def _is_terminal_node(node: MCTSNode) -> bool:
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
    def _only_action(node: MCTSNode, state: State, world: MCTSInterface, c: float) -> Action:
        assert node is not None

        for child in node.get_children():
            if child.terminal != -1:
                continue
            MCTS._expand(
                child,
                world.play(world.copy(state), child.action),
                world
            )

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
    def _random_rollout(root: MCTSNode, root_state: State, world: MCTSInterface, n: int = 100) -> None:
        """Performs a random rollout starting from the given tree."""
        assert n > 0, f"Invalid number of random rollouts: n = {n}"

        for _ in repeat(None, n):
            node: MCTSNode = root
            state: State   = world.copy(root_state)

            while not node.is_leaf():
                possible_children: List[MCTSNode] = [
                    child
                    for child in node.children[:node.s_children]
                    if child.terminal == -1
                ]
                if not possible_children:
                    return

                node = possible_children[randint(0, len(possible_children) - 1)]
                state = world.play(state, node.action)

            if not MCTS._expand(node, state, world):
                continue

            for child in node.get_children():
                if child.terminal != -1:
                    continue
                MCTS._rollout(
                    child,
                    world.play(world.copy(state), child.action),
                    world
                )

        MCTS._sort_children(root)
        if root.children[0].terminal != -1:
            return

    @staticmethod
    def _evaluate(node: MCTSNode, c: float) -> float:
        """Evaluates a node using the UCT formula."""
        assert node.parent != None and not MCTS._is_terminal_node(node)
        return node.reward / node.visits + c * sqrt(log(node.parent.visits) / node.visits) \
            if node.visits > 0 and node.parent.visits >= 1\
            else float("inf")

    @staticmethod
    def _select(root: MCTSNode, root_state: State, world: MCTSInterface, c: float) -> tuple[Optional[MCTSNode], State]:
        """Selects the best child node using the UCT formula."""

        node: MCTSNode = root
        state: State   = world.copy(root_state)

        while not node.is_leaf():
            best_child_idx: int = 0
            while node.children[best_child_idx].terminal != -1:
                best_child_idx += 1
                if best_child_idx >= node.s_children:
                    return None, state
            
            best_score: float = MCTS._evaluate(node.children[best_child_idx], c)

            for idx in range(best_child_idx + 1, node.s_children):
                child: MCTSNode = node.children[idx]
                if child.terminal != -1:
                    continue

                score: float = MCTS._evaluate(child, c)
                if score > best_score:
                    best_score     = score
                    best_child_idx = idx
            
            node  = node.children[best_child_idx]
            state = world.play(state, node.action)

        return node, state

    @staticmethod
    def _expand(node: MCTSNode, state: State, world: MCTSInterface) -> bool:
        """Expands the node by generating all possible children."""
        assert node.is_leaf() and not MCTS._is_terminal_node(node)
        
        max_terminal: float = -1
        for action in world.get_actions(state):
            child: MCTSNode = MCTSNode(action, node)
            node.add_child(child)

            child.terminal = world.value(
                world.play(world.copy(state), action), 
                child.action
            )
            if child.terminal == 1:
                MCTS._backpropagate_terminal(node, 0)
                return False

            max_terminal = max(max_terminal, child.terminal)

        if max_terminal > -1:
            MCTS._backpropagate_terminal(node, 1 - max_terminal)

        return node.s_children > 0

    @staticmethod
    def _rollout(leaf: MCTSNode, state: State, world: MCTSInterface) -> None:
        """Simulates a random rollout from the given leaf node."""
        assert leaf != None and leaf.is_leaf(), f"leaf: {leaf}"
        assert not MCTS._is_terminal_node(leaf), world.print(state)

        action: Action = None
        value:  float  = -1
        player: int    = world.reverse_player(
            world.get_current_player(state)
        )

        while value == -1:
            action = choice(world.get_actions(state))
            state  = world.play(state, action)
            value  = world.value(state, action, player)

        MCTS._backpropagate(leaf, value)

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

        while node != None and (terminal == 0 or not node.has_undetermined_child()):
            node.terminal = terminal

            if node.s_children > 0 and not node.is_root():
                node.remove_children()

            terminal = 1 - terminal
            node = node.parent
            if node is not None:
                node.undet_children -= 1

    @staticmethod
    def _print_node(node: MCTSNode, state: State, world: MCTSInterface, c: float) -> None:
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

        print("  state =")
        world.print(state)

        print("  Children: {")
        for child in sorted(node.get_children(), key = eval_child, reverse = True):
            print(f"    action: {child.action}, visits: {child.visits}, value: {eval_child(child):.3f}, terminal: {child.terminal}")
        print("  }\n}")

    @staticmethod
    def mcts(
        root_state: State, world: MCTSInterface, s_rollout: int, max_expansion: int = 10,
        s_initial_rollout: int = 100, c: float = round(sqrt(2), 3),
        tree: bool = False, debug: bool = False, timer: bool = False
    ) -> tuple[Action, Optional[MCTSNode]]:
        """Performs the Monte Carlo Tree Search and returns the best action."""

        if timer:
            start = default_timer()

        root: MCTSNode = MCTS._encapsulate(None, max_expansion)
        MCTS._expand(root, root_state, world)
        MCTS._sort_children(root)

        if root.children[0].terminal == 1:
            if debug:
                print("Left winning_actions")
                MCTS._print_node(root, root_state, world, c)
            return root.children[0].action, root
        
        if (only_action := MCTS._only_action(root, root_state, world, c)) != None:
            if debug:
                print("Left only_action or just lost")
                MCTS._print_node(root, root_state, world, c)
            return only_action, root
        
        MCTS._random_rollout(root, root_state, world, s_initial_rollout)

        for _ in repeat(None, s_rollout):
            MCTS._sort_children(root)
            if root.children[0].terminal == 1:
                break

            node, state = MCTS._select(root, root_state, world, c)
            if node == None or node.terminal != -1:
                break

            if node.visits > 0:
                if not MCTS._expand(node, state, world):
                    continue

                node  = node.children[randint(0, node.s_children - 1)]
                state = world.play(state, node.action)

                node.terminal = world.value(state, node.action)
                if node.terminal != -1:
                    MCTS._backpropagate_terminal(node.parent, 1 - node.terminal)
                    continue

            MCTS._rollout(node, state, world)

        if timer:
            print(f"Total execution time: {(default_timer() - start):.3f} seconds")
        if debug:
            print("Left normally")
            MCTS._print_node(root, root_state, world, c)

        return MCTS._pick_action(root), None if not tree else root
