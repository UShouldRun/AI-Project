from random import randint
from abc import ABC, abstractmethod
from math import sqrt, log
from typing import TypeVar

State = TypeVar("State")
Action = TypeVar("Action")

class MCTSInterface(ABC):
    @staticmethod
    @abstractmethod
    def play(state: State, action: Action) -> State:
        pass

    @staticmethod
    @abstractmethod
    def get_actions(state: State) -> [Action]:
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
    def __init__(self, data: any, parent: "MCTSNode") -> None:
        self.data: any              = data
        self.parent: "MCTSNode"     = parent # MCTSNode type is being defined so we need to use ""
        self.children: ["MCTSNode"] = []
        self.depth: int             = 0 if parent == None else parent.depth + 1
        self.result: float          = 0
        self.visits: int            = 0

    def is_root(self) -> bool:
        return self.parent == None
    def is_leaf(self) -> bool:
        return self.children == []

    def add_child(self, node: "MCTSNode") -> None:
        self.children.append(node)
        node.parent = self
        node.depth = self.depth + 1

    def get_leafs(self) -> ["MCTSNode"]:
        if self.is_leaf():
            return [self]

        leafs = []
        for child in self.children:
            leafs.extend(child.get_leafs())
        return leafs

class MCTS:
    @staticmethod
    def encapsulate(data: any) -> MCTSNode:
        return MCTSNode(data, None)

    @staticmethod
    def random_rollout(tree: MCTSNode, world: MCTSInterface, n: int) -> None:
        for _ in range(n):
            leafs: [MCTSNode] = tree.get_leafs()

            node: MCTSNode = leafs[randint(0, len(leafs) - 1)]
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
        if node.is_leaf() and not world.is_terminal_state(node.data):
            node.children.extend(MCTSNode(data, node) for data in world.get_states(node.data))

    @staticmethod
    def rollout(leaf: MCTSNode, world: MCTSInterface) -> None:
        data: any = world.copy(leaf.data)

        while not world.is_terminal_state(data):
            states: any = world.get_states(data)
            if not states:
                break
            data = states[randint(0, len(states) - 1)]

        MCTS.backpropagate(leaf, world.value(data))

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
    def mcts(tree: MCTSNode, world: MCTSInterface, s_initial_rollout: int, s_rollout: int, c: float = round(sqrt(2), 3)) -> MCTSNode:
        MCTS.random_rollout(tree, world, s_initial_rollout)

        for _ in range(s_rollout):
            node: MCTSNode = MCTS.select(tree, c)

            if node.visits > 0:
                MCTS.expand(node, world)
                node = node.children[0]

            MCTS.rollout(node, world)

        return max(tree.children, key = lambda child: MCTS.evaluate(child, c)) if tree.children else tree
