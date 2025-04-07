from random import choice 
from abc import ABC, abstractmethod
from math import sqrt, log
from typing import TypeVar, List

import asyncio

State = TypeVar("State")
Action = TypeVar("Action")

class MCTSInterface(ABC):
    @staticmethod
    @abstractmethod
    def play(state: State, action: Action) -> State:
        """Executes the action on the given state and returns the resulting state."""
        pass

    @staticmethod
    @abstractmethod
    def get_actions(state: State) -> List[Action]:
        """Returns a list of valid actions for the given state."""
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
    def copy(state: State) -> State:
        """Creates and returns a copy of the given state."""
        pass

    @staticmethod
    @abstractmethod
    def print(state: State) -> None:
        pass

class MCTSNode:
    def __init__(self, state: State, action: Action, parent: "MCTSNode") -> None:
        self.state: State               = state
        self.action: Action             = action
        self.parent: "MCTSNode"         = parent # MCTSNode type is being defined so we need to use ""
        self.children: List["MCTSNode"] = []
        self.depth: int                 = 0 if parent == None else parent.depth + 1
        self.result: float              = 0
        self.visits: int                = 0

    def is_root(self) -> bool:
        return self.parent == None
    def is_leaf(self) -> bool:
        return self.children == []

    def get_leafs(self) -> List["MCTSNode"]:
        if self.is_leaf():
            return [self]

        leafs = []
        for child in self.children:
            leafs.extend(child.get_leafs())
        return leafs

class MCTS:
    @staticmethod
    def _encapsulate(state: State, action: Action) -> MCTSNode:
        return MCTSNode(state, action, None)

    @staticmethod
    def _pick_action(root: MCTSNode, c: float) -> Action:
        assert not root.is_leaf()
        return max(root.children, key = lambda child: MCTS._evaluate(child, c)).action

    @staticmethod
    def _random_rollout(tree: MCTSNode, player: int, world: MCTSInterface, n: int) -> None:
        """Performs a random rollout starting from the given tree."""
        for _ in range(n):
            node: MCTSNode = choice(tree.get_leafs())
            MCTS._expand(node, world)
            for child in node.children:
                MCTS._rollout(child, player, world)

    @staticmethod
    def _evaluate(node: MCTSNode, c: float) -> float:
        """Evaluates a node using the UCT formula."""
        assert node.parent != None
        if node.visits <= 0 or node.parent.visits < 1:
            return float("inf")
        return node.result / node.visits + c * sqrt(log(node.parent.visits)/node.visits)

    @staticmethod
    def _select(node: MCTSNode, c: float) -> MCTSNode:
        """Selects the best child node using the UCT formula."""
        return node if node.is_leaf() else max(node.children, key = lambda child: MCTS._evaluate(child, c))

    @staticmethod
    def _expand(node: MCTSNode, world: MCTSInterface) -> None:
        """Expands the node by generating all possible children."""
        if node.is_leaf() and not world.is_terminal_state(node.state, node.action):
            node.children += [
                MCTSNode(world.play(world.copy(node.state), action), action, node)
                for action in world.get_actions(node.state)
            ]

    @staticmethod
    def _rollout(leaf: MCTSNode, player: int, world: MCTSInterface) -> None:
        """Simulates a random rollout from the given leaf node."""
        state: State   = leaf.state
        action: Action = None
        while not world.is_terminal_state(state, action):
            actions: List[Action] = world.get_actions(state)
            if not actions:
                break
            state, action = choice([
                (world.play(world.copy(state), action), action) for action in actions
            ])
        MCTS._backpropagate(leaf, world.value(state, action, (leaf.depth + (1 if player == 2 else 0)) % 2 + 1))

    @staticmethod
    def _backpropagate(leaf: MCTSNode, result: float) -> None:
        """Backpropagates the result from the leaf node up to the root."""
        assert leaf != None
        assert 0 <= result <= 1

        node: MCTSNode = leaf
        while node != None:
            node.visits += 1
            node.result += result
            result = 1 - result
            node = node.parent

    @staticmethod
    def _print_node(node: MCTSNode, world: MCTSInterface, c: float) -> None:
        print("Node {")
        print(f"depth = {node.depth},\nresult = {node.result},\nvisits = {node.visits}")
        world.print(node.state)
        print("Children: {")
        for child in sorted(node.children, key = lambda child: 1 - MCTS._evaluate(child, c)):
            print(f"action: {child.action}, value: {MCTS._evaluate(child, c)}")
        print("}\n}")

    @staticmethod
    async def mcts(state: State, player: int, world: MCTSInterface, s_initial_rollout: int, s_rollout: int, c: float = round(sqrt(2), 3)) -> Action:
        """Performs the Monte Carlo Tree Search and returns the best action."""
        tree: MCTSNode = MCTS._encapsulate(state, None)
        MCTS._random_rollout(tree, player, world, s_initial_rollout)

        for _ in range(s_rollout):
            node: MCTSNode = MCTS._select(tree, c)
            if node.visits > 0:
                MCTS._expand(node, world)
                node = node.children[0]
            MCTS._rollout(node, player, world)

        MCTS._print_node(tree, world, c)
        return MCTS._pick_action(tree, c)
