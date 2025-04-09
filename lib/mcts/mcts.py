from random import randint
from abc import ABC, abstractmethod
from math import sqrt, log
from typing import TypeVar, List, Optional
from timeit import default_timer

import numpy as np
import asyncio

State = TypeVar("State")
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
    def value(state: State, action: Action, player: int) -> float:
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
        'depth', 'reward', 'visits', 'terminal_value'
    )
    
    def __init__(self, state: State, action: Action, parent: Optional["MCTSNode"]) -> None:
        self.state: State = state
        self.action: Action = action
        self.parent: Optional["MCTSNode"] = parent

        # Initialize with a small array size, will expand as needed
        self.max_children: int = 10  # Initial capacity
        self.s_children: int = 0
        self.children = np.empty(self.max_children, dtype = np.object_)

        self.depth: int = 0 if parent is None else parent.depth + 1
        self.reward: float = 0
        self.visits: int = 0
        self.terminal_value: float = -1
    
    def is_root(self) -> bool:
        return self.parent is None
    
    def is_leaf(self) -> bool:
        return self.s_children == 0
    
    def add_child(self, child: "MCTSNode") -> None:
        """Add a child node, resizing the array if necessary."""
        if self.s_children >= self.max_children:
            # Double the capacity
            new_max = self.max_children * 2
            new_children = np.empty(new_max, dtype = np.object_)
            new_children[:self.s_children] = self.children[:self.s_children]
            self.children = new_children
            self.max_children = new_max
        
        self.children[self.s_children] = child
        self.s_children += 1
    
    def get_children(self) -> np.ndarray:
        """Get the actual children (not the empty slots)."""
        return self.children[:self.s_children]
    
    def get_leafs(self) -> List["MCTSNode"]:
        result = []
        stack = [self]
        
        while stack:
            current = stack.pop()
            if current.is_leaf():
                result.append(current)
            else:
                for i in range(current.s_children):
                    stack.append(current.children[i])
        
        return result

class MCTS:
    @staticmethod
    def _encapsulate(state: State, action: Action) -> MCTSNode:
        return MCTSNode(state, action, None)

    @staticmethod
    def _is_terminal_state(node: MCTSNode) -> bool:
        return node.action != None and node.terminal_value > -1

    @staticmethod
    def _convert_terminal_value(value: int) -> float:
        return -float("inf") if value == 0 else (0.5 if value == 0.5 else float("inf"))

    @staticmethod
    def _pick_action(root: MCTSNode) -> Action:
        assert not root.is_leaf()

        best_child: Optional[MCTSNode] = None
        best_value: float = float("inf")

        for i in range(root.s_children):
            child: MCTSNode = root.children[i]
            value: float = (
                child.reward/child.visits
                if child.terminal_value == -1
                else MCTS._convert_terminal_value(child.terminal_value)
            )
            if value < best_value:
                best_child = child
                best_value = value

        return best_child.action

    @staticmethod
    def _only_action(node: MCTSNode, world: MCTSInterface, c: float) -> Action:
        if node is None:
            return None

        for child in node.get_children():
            MCTS._expand(child, world)

        s_children: int = node.s_children
        non_losing: List[MCTSNode] = []
        for i in range(s_children):
            child: MCTSNode = node.children[i]
            if child.terminal_value == -1 or child.terminal_value > 0:
                non_losing.append(child.action)

        s_non_losing: int = len(non_losing)
        if s_non_losing == 1:
            return non_losing[0]
        elif s_non_losing == 0:
            actions: List[Action] = []
            for i in range(s_children):
                actions.append(node.children[i].action)
            return actions[randint(0, s_children - 1)]

        return None

    @staticmethod
    def _random_rollout(tree: MCTSNode, world: MCTSInterface, heuristic: tuple[bool, int], n: int = 100) -> None:
        """Performs a random rollout starting from the given tree."""
        assert n > 0, f"Invalid number of random rollouts: n = {n}"
        player: int = world.get_current_player(tree.state)

        for _ in range(n):
            leafs: List[MCTSNode] = []
            for leaf in tree.get_leafs():
                if not MCTS._is_terminal_state(leaf):
                    leafs.append(leaf)

            if leafs == []:
                break

            node: MCTSNode = leafs[randint(0, len(leafs) - 1)]
            if not MCTS._expand(node, world):
                continue

            for child in node.get_children():
                MCTS._rollout(child, world, heuristic)

    @staticmethod
    def _evaluate(node: MCTSNode, c: float) -> float:
        """Evaluates a node using the UCT formula."""
        assert node.parent != None
        if node.terminal_value > -1:
            return MCTS._convert_terminal_value(node)
        if node.visits <= 0 or node.parent.visits < 1:
            return float("inf")
        return node.reward / node.visits + c * sqrt(log(node.parent.visits) / node.visits)

    @staticmethod
    def _select(node: MCTSNode, c: float) -> Optional[MCTSNode]:
        """Selects the best child node using the UCT formula."""
        visited: List[MCTSNode] = []

        while node is not None and not node.is_leaf():
            non_terminal: List[int] = []
            for i in range(node.s_children):
                child = node.children[i]
                if child.terminal_value == -1:
                    non_terminal.append(i)

            if not non_terminal:
                max_terminal_value = -float("inf")
                for i in range(node.s_children):
                    max_terminal_value = max(
                        max_terminal_value,
                        node.children[i].terminal_value
                    )
                node.terminal_value = max_terminal_value

                if not visited:
                    return None

                node = visited.pop()
                continue

            visited.append(node)


            best_child_idx: int = non_terminal[0]
            best_score: float = MCTS._evaluate(node.children[best_child_idx], c)
            
            for idx in non_terminal[1:]:
                child: MCTSNode = node.children[idx]
                score: float = MCTS._evaluate(child, c)
                if score > best_score:
                    best_score = score
                    best_child_idx = idx
            
            node = node.children[best_child_idx]

        return node

    @staticmethod
    def _expand(node: MCTSNode, world: MCTSInterface) -> bool:
        """Expands the node by generating all possible children."""
        assert node.is_leaf()
        if MCTS._is_terminal_state(node):
            MCTS._backpropagate(node, node.terminal_value, terminal_value = True)
            return False
        
        key_leaf: bool = False
        max_key_value: float = -float("inf")

        player: int = world.get_current_player(node.state)
        for action in world.get_actions(node.state):
            child: MCTSNode = MCTSNode(
                world.play(world.copy(node.state), action),
                action,
                node
            )
            node.add_child(child)

            child.terminal_value = world.value(
                child.state,
                child.action,
                player
            )

            if child.terminal_value > -1:
                child.terminal_value = 1 - child.terminal_value
                key_leaf = True
                max_key_value = max(max_key_value, child.terminal_value)

        if key_leaf and node.parent:
            MCTS._backpropagate(
                node,
                max_key_value,
                terminal_value = True
            )

        return not key_leaf

    @staticmethod
    def _rollout(
        leaf: MCTSNode, world: MCTSInterface, heuristic: tuple[bool, int], timer: bool = False
    ) -> tuple[int, Optional[List[float]]]:
        """Simulates a random rollout from the given leaf node."""
        assert leaf != None and leaf.is_leaf(), f"leaf: {leaf}"
        assert not MCTS._is_terminal_state(leaf)
        times = None

        if timer:
            start = default_timer()
            times = 4 * [0]

        state: State   = world.copy(leaf.state)
        action: Action = None
        value: float   = None
        player: int    = world.get_current_player(state)
        depth: int     = leaf.depth

        while (not heuristic[0] or depth <= heuristic[1]) and (action is None or value == -1):
            actions: List[Action] = world.get_actions(state)
            assert actions is not None

            action = actions[randint(0, len(actions) - 1)]
            state  = world.play(state, action)

            value  = world.value(state, action, player)
            depth += 1

            if timer:
                times[1] = (times[0] * times[1] + default_timer() - start) / (times[0] + 1)
                times[0] += 1
                start = default_timer()

        if value is None or value == -1:
            value = world.heuristic(leaf.state, player)
            if timer:
                times[2] = default_timer() - start
                start = default_timer()

        MCTS._backpropagate(leaf, value)
        if timer:
            times[3] = default_timer() - start

        return depth, times

    @staticmethod
    def _backpropagate(node: MCTSNode, reward: float, terminal_value: bool = False) -> None:
        """Backpropagates the reward from the a node up to the root."""
        assert node != None
        assert 0 <= reward <= 1

        while node != None:
            if terminal_value:
                if (
                        node.terminal_value != -1
                    and node.terminal_value < reward
                    and (
                           reward == 1
                        or not any(child.terminal_value == -1 for child in node.get_children())
                    )
                ):
                    node.terminal_value = reward

                    if node.terminal_value > -1:
                        # Reset children to save memory
                        node.children     = np.empty(0, dtype = np.object_)
                        node.s_children   = 0
                        node.max_children = 0

            else:
                node.visits += 1
                node.reward += reward

            reward = 1 - reward
            node = node.parent

    @staticmethod
    def _print_node(node: MCTSNode, world: MCTSInterface, c: float) -> None:
        eval = lambda node: (
            (
                node.reward/node.visits
                if node.visits > 0 else
                -1
            )
            if node.terminal_value == -1 else
            (
                MCTS._convert_terminal_value(node)
                if MCTS._convert_terminal_value(node) != 0.5
                else 0.5
            )
        )
        eval_child = lambda child: (
            (
                1 - child.reward/child.visits
                if child.visits > 0 else
                -1
            )
            if child.terminal_value == -1 else
            (
                - MCTS._convert_terminal_value(child)
                if MCTS._convert_terminal_value(child) != 0.5
                else 0.5
            )
        )

        print("Node {")
        print(f"  depth = {node.depth},")
        print(f"  visits = {node.visits},")
        print(f"  eval = {eval(node):.3f},")
        print(f"  s_children = {len(node.children)},")
        print("  state =")
        world.print(node.state)

        print("  Children: {")
        for child in sorted(node.get_children(), key = eval_child, reverse = True):
            print(f"    action: {child.action}, visits: {child.visits}, value: {eval_child(child):.3f}")
        print("  }\n}")

    @staticmethod
    async def mcts(
        state: State, world: MCTSInterface, s_rollout: int,
        s_initial_rollout: int = 100, c: float = round(sqrt(2), 3),
        debug: bool = False, timer: bool = False, heuristic: tuple[bool, int] = (False, None)
    ) -> Action:
        """Performs the Monte Carlo Tree Search and returns the best action."""

        if timer:
            start = default_timer()

        tree: MCTSNode = MCTS._encapsulate(state, None)
        MCTS._expand(tree, world)

        if timer:
            elapsed_time = default_timer() - start
            print(f"Tree creation and expansion execution time: {elapsed_time:.6f} seconds")
            start = default_timer()

        winning_actions: List[Action] = []
        for i in range(tree.s_children):
            child: MCTSNode = tree.children[i]
            if child.terminal_value == 1:
                winning_actions.append(child.action)

        if winning_actions != []:
            if timer:
                elapsed_time = default_timer() - start
                print(f"Winning actions execution time: {elapsed_time:.6f} seconds")
            if debug:
                print("Left winning_actions")
                MCTS._print_node(tree, world, c)

            return winning_actions[0]
        if timer:
            elapsed_time = default_timer() - start
            print(f"Winning actions execution time: {elapsed_time:.6f} seconds")
            start = default_timer()

        if (only_action := MCTS._only_action(tree, world, c)) != None:
            if timer:
                elapsed_time = default_timer() - start
                print(f"Only action execution time: {elapsed_time:.6f} seconds")
            if debug:
                print("Left only_action or just lost")
                MCTS._print_node(tree, world, c)

            return only_action
        if timer:
            elapsed_time = default_timer() - start
            print(f"Only action execution time: {elapsed_time:.6f} seconds")
            start = default_timer()

        MCTS._random_rollout(tree, world, heuristic, s_initial_rollout)
        if timer:
            elapsed_time = default_timer() - start
            print(f"Random rollout execution time: {elapsed_time:.6f} seconds")
            avg_times: List[float] = 8 * [0]

        max_depth: int = 0
        for _ in range(s_rollout):
            if timer:
                start = default_timer()

            node: MCTSNode = MCTS._select(tree, c)
            if node == None:
                break

            # if debug:
                # MCTS._print_node(node, world, c) 
            if timer:
                elapsed_time = default_timer() - start
                avg_times[1] = (avg_times[0] * avg_times[1] + elapsed_time) / (avg_times[0] + 1)
                start = default_timer()

            if node.visits > 0:
                expansion: bool = MCTS._expand(node, world)
                if timer:
                    elapsed_time = default_timer() - start
                    avg_times[2] = (avg_times[0] * avg_times[2] + elapsed_time) / (avg_times[0] + 1)
                    start = default_timer()

                if not expansion:
                    continue
                node = node.children[randint(0, node.s_children - 1)]

            depth, times = MCTS._rollout(node, world, heuristic, timer = True and timer)
            max_depth = max(max_depth, depth)
            if timer:
                elapsed_time = default_timer() - start
                avg_times[3] = (avg_times[0] * avg_times[3] + elapsed_time) / (avg_times[0] + 1)
                avg_times[0] += 1

                if times is not None:
                    for i in range(3):
                        avg_times[4 + i] = (avg_times[0] * avg_times[4 + i] + times[i + 1]) / (avg_times[0] + 1)

        if timer:
            elapsed_time = default_timer() - start
            
            total_execution_time = sum(avg_times[0] * avg_times[i] for i in range(1, 4))
            print(f"Total execution time: {total_execution_time:.6f} seconds")

            selection_avg_time_ms = avg_times[1] * 1e3
            selection_total_time = avg_times[0] * avg_times[1]
            print(f"Selection execution time: ")
            print(f"  - Average: {selection_avg_time_ms:.6f} ms")
            print(f"  - Total: {selection_total_time:.3f} seconds")

            expansion_avg_time_ms = avg_times[2] * 1e3
            expansion_total_time = avg_times[0] * avg_times[2]
            print(f"Expansion execution time: ")
            print(f"  - Average: {expansion_avg_time_ms:.6f} ms")
            print(f"  - Total: {expansion_total_time:.3f} seconds")

            rollout_avg_time_ms = avg_times[3] * 1e3
            rollout_total_time = avg_times[0] * avg_times[3]
            print(f"Rollout (max_depth = {max_depth}) execution time: ")
            print(f"  - Average: {rollout_avg_time_ms:.6f} ms")
            print(f"  - Total: {rollout_total_time:.3f} seconds")

            if times is not None:
                print(f"  - main loop: Average = {avg_times[4]:.6f} ms, Total = {avg_times[0] * avg_times[4]:.3f} s")
                print(f"  - heuristic: Average = {avg_times[5]:.6f} ms, Total = {avg_times[0] * avg_times[5]:.3f} s")
                print(f"  - backpropagation: Average = {avg_times[6]:.6f} ms, Total = {avg_times[0] * avg_times[6]:.3f} s")

        if debug:
            print("Left normally")
            MCTS._print_node(tree, world, c)

        return MCTS._pick_action(tree)
