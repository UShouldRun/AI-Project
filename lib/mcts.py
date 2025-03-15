from random import choice 
from abc import ABC, abstractmethod
from math import sqrt, log
from typing import TypeVar, List

State = TypeVar("State")
Action = TypeVar("Action")

class MCTSInterface(ABC):
    @staticmethod
    @abstractmethod
    def play(state: State, action: Action) -> State:
        pass

    @staticmethod
    @abstractmethod
    def get_actions(state: State) -> List[Action]:
        pass

    @staticmethod
    @abstractmethod
    def is_terminal_state(state: State) -> bool:
        pass

    @staticmethod
    @abstractmethod
    def value(state: State) -> float:
        pass

    @staticmethod
    @abstractmethod
    def copy(state: State) -> State:
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

    def add_child(self, node: "MCTSNode") -> None:
        self.children.append(node)
        node.parent = self
        node.depth = self.depth + 1

    def get_leafs(self) -> List["MCTSNode"]:
        if self.is_leaf():
            return [self]

        leafs = []
        for child in self.children:
            leafs.extend(child.get_leafs())
        return leafs

class MCTS:
    @staticmethod
    def encapsulate(state: State, action: Action) -> MCTSNode:
        return MCTSNode(state, action, None)

    def pick_action(root: MCTSNode, c: float) -> Action:
        assert not root.is_leaf()
        return max(root.children, key = lambda child: MCTS.evaluate(child, c)).action

    @staticmethod
    def random_rollout(tree: MCTSNode, world: MCTSInterface, n: int) -> None:
        for _ in range(n):
            node: MCTSNode = choice(tree.get_leafs())
            MCTS.expand(node, world)
            for child in node.children:
                MCTS.rollout(child, world)

    @staticmethod
    def evaluate(node: MCTSNode, c: float) -> float:
        assert node.parent != None
        if node.visits <= 0 or node.parent.visits < 1:
            return float("inf")
        return node.result / node.visits + c * sqrt(log(node.parent.visits)/node.visits)

    @staticmethod
    def select(node: MCTSNode, c: float) -> MCTSNode:
        return node if node.is_leaf() else max(node.children, key = lambda child: MCTS.evaluate(child, c))

    @staticmethod
    def expand(node: MCTSNode, world: MCTSInterface) -> None:
        if node.is_leaf() and not world.is_terminal_state(node.state):
            node.children += [
                MCTSNode(world.play(node.state, action), action, node)
                for action in world.get_actions(node.state)
            ]

    @staticmethod
    def rollout(leaf: MCTSNode, world: MCTSInterface) -> None:
        state: State = world.copy(leaf.state)

        while not world.is_terminal_state(state):
            actions: List[Action] = world.get_actions(state)
            if not actions:
                break
            state = choice([world.play(state, action) for action in actions])

        MCTS.backpropagate(leaf, world.value(state))

    @staticmethod
    def backpropagate(leaf: MCTSNode, result: float) -> None:
        assert leaf != None
        assert 0 <= result <= 1

        node: MCTSNode = leaf
        while not node.is_root():
            node.visits += 1
            node.result += result
            result = 1 - result
            node = node.parent

    @staticmethod
    def mcts(tree: MCTSNode, world: MCTSInterface, s_initial_rollout: int, s_rollout: int, c: float = round(sqrt(2), 3)) -> Action:
        MCTS.random_rollout(tree, world, s_initial_rollout)

        for _ in range(s_rollout):
            node: MCTSNode = MCTS.select(tree, c)

            if node.visits > 0:
                MCTS.expand(node, world)
                node = node.children[0]

            MCTS.rollout(node, world)

        return MCTS.pick_action(tree)
