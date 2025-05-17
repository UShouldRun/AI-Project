# cython: language_level=3
# distutils: extra_compile_args = -O3 -ffast-math

import numpy as np
cimport numpy as np
import cython
from libc.math cimport log, sqrt
from random import randint, choice
from timeit import default_timer
from itertools import repeat
from typing import Optional, List, Tuple

# Concrete types for Connect4
ctypedef long State  # Represent board state as long (bitboard)
ctypedef int Action  # Represent moves as integers (column numbers)

cdef class MCTSNode:
    cdef public Action action
    cdef public MCTSNode parent
    cdef public np.ndarray children
    cdef public int s_children, max_children, depth, visits, undet_children
    cdef public double reward, terminal
    
    def __init__(self, Action action, MCTSNode parent, int max_children=-1):
        self.action = action
        self.parent = parent
        self.max_children = max_children if parent is None else parent.max_children
        self.s_children = 0
        self.children = np.empty(self.max_children, dtype=object)
        self.depth = 0 if parent is None else parent.depth + 1
        self.reward = 0
        self.visits = 0
        self.terminal = -1
        self.undet_children = 0
    
    def is_root(self) -> bool:
        return self.parent is None
    
    def is_leaf(self) -> bool:
        return self.s_children == 0

    def has_undetermined_child(self) -> bool:
        return self.undet_children > 0
    
    def add_child(self, MCTSNode child) -> None:
        self.children[self.s_children] = child
        self.s_children += 1
        self.undet_children += 1

    def remove_children(self) -> None:
        self.children = np.empty(0, dtype=object)
        self.s_children = 0
        self.max_children = 0
        self.undet_children = 0
    
    def get_children(self) -> np.ndarray:
        return self.children[:self.s_children]

cdef class MCTS:
    @staticmethod
    def _encapsulate(Action action, int max_expansion) -> MCTSNode:
        return MCTSNode(action, None, max_expansion)

    @staticmethod
    def _is_terminal_node(MCTSNode node) -> bool:
        return node.action is not None and node.terminal > -1

    @staticmethod
    cdef double _inverse_sigmoid(double x):
        if x == 1: return float("inf")
        if x == 0: return -float("inf")
        return log(x / (1 - x))

    @staticmethod
    cdef double _convert_eval(double value):
        return MCTS._inverse_sigmoid(value)

    @staticmethod
    cdef double _convert_terminal(double value):
        return -float("inf") if value == 0 else (0 if value == 0.5 else float("inf"))

    @staticmethod
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _sort_children(MCTSNode root) -> None:
        cdef MCTSNode key_node, j_node
        cdef double key_value, j_value
        cdef int i, j
        
        for i in range(1, root.s_children):
            key_node = root.children[i]
            key_value = (MCTS._convert_eval(key_node.reward / key_node.visits) 
                        if key_node.visits > 0 else 1000) if key_node.terminal == -1 else MCTS._convert_terminal(key_node.terminal)
            j = i - 1

            while j >= 0:
                j_node = root.children[j]
                j_value = (MCTS._convert_eval(j_node.reward / j_node.visits) 
                          if j_node.visits > 0 else 1000) if j_node.terminal == -1 else MCTS._convert_terminal(j_node.terminal)

                if key_value <= j_value:
                    break

                root.children[j + 1] = root.children[j]
                j -= 1

            root.children[j + 1] = key_node

    @staticmethod
    def _pick_action(MCTSNode root) -> Action:
        return root.children[0].action

    @staticmethod
    def _only_action(MCTSNode node, State state, object world, double c) -> Action:
        cdef MCTSNode child
        cdef int i, s_non_losing = 0, index = 0
        
        for child in node.get_children():
            if child.terminal != -1:
                continue
            MCTS._expand(child, world.play(world.copy(state), child.action), world)

        for i in range(node.s_children):
            child = node.children[i]
            if child.terminal == -1 or child.terminal > 0:
                s_non_losing += 1
                index = i

        if s_non_losing > 1:
            return None
        if s_non_losing == 1:
            return node.children[index].action
        return node.children[randint(0, node.s_children - 1)].action

    @staticmethod
    def _random_rollout(MCTSNode root, State root_state, object world, int n=100) -> None:
        cdef int _
        cdef MCTSNode node
        cdef State state
        
        for _ in range(n):
            node = root
            state = world.copy(root_state)

            while not node.is_leaf():
                possible_children = [child for child in node.children[:node.s_children] if child.terminal == -1]
                if not possible_children:
                    return

                node = possible_children[randint(0, len(possible_children) - 1)]
                state = world.play(state, node.action)

            if not MCTS._expand(node, state, world):
                continue

            for child in node.get_children():
                if child.terminal != -1:
                    continue
                MCTS._rollout(child, world.play(world.copy(state), child.action), world)

        MCTS._sort_children(root)
        if root.children[0].terminal != -1:
            return

    @staticmethod
    cdef double _evaluate(MCTSNode node, double c):
        if node.visits > 0 and node.parent.visits >= 1:
            return node.reward / node.visits + c * sqrt(log(node.parent.visits) / node.visits)
        return float("inf")

    @staticmethod
    def _select(MCTSNode root, State root_state, object world, double c) -> Tuple[Optional[MCTSNode], State]:
        cdef MCTSNode node, child
        cdef State state
        cdef int best_child_idx, idx
        cdef double best_score, score
        
        node = root
        state = world.copy(root_state)

        while not node.is_leaf():
            best_child_idx = 0
            while node.children[best_child_idx].terminal != -1:
                best_child_idx += 1
                if best_child_idx >= node.s_children:
                    return None, state
            
            best_score = MCTS._evaluate(node.children[best_child_idx], c)

            for idx in range(best_child_idx + 1, node.s_children):
                child = node.children[idx]
                if child.terminal != -1:
                    continue

                score = MCTS._evaluate(child, c)
                if score > best_score:
                    best_score = score
                    best_child_idx = idx
            
            node = node.children[best_child_idx]
            state = world.play(state, node.action)

        return node, state

    @staticmethod
    def _expand(MCTSNode node, State state, object world) -> bool:
        cdef double max_terminal = -1
        cdef Action action
        cdef MCTSNode child
        
        for action in world.get_actions(state):
            child = MCTSNode(action, node)
            node.add_child(child)
            child.terminal = world.value(world.play(world.copy(state), action), action)
            
            if child.terminal == 1:
                MCTS._backpropagate_terminal(node, 0)
                return False

            max_terminal = max(max_terminal, child.terminal)

        if max_terminal > -1:
            MCTS._backpropagate_terminal(node, 1 - max_terminal)

        return node.s_children > 0

    @staticmethod
    def _rollout(MCTSNode leaf, State state, object world) -> None:
        cdef Action action
        cdef double value = -1
        cdef int player = world.reverse_player(world.get_current_player(state))

        while value == -1:
            action = choice(world.get_actions(state))
            state = world.play(state, action)
            value = world.value(state, action, player)

        MCTS._backpropagate(leaf, value)

    @staticmethod
    def _backpropagate(MCTSNode node, double reward) -> None:
        while node is not None:
            node.visits += 1
            node.reward += reward
            reward = 1 - reward
            node = node.parent

    @staticmethod
    def _backpropagate_terminal(MCTSNode node, double terminal) -> None:
        while node is not None and (terminal == 0 or not node.has_undetermined_child()):
            node.terminal = terminal
            if node.s_children > 0 and not node.is_root():
                node.remove_children()
            terminal = 1 - terminal
            node = node.parent
            if node is not None:
                node.undet_children -= 1

    @staticmethod
    def _print_node(MCTSNode node, State state, object world, double c) -> None:
        def eval_child(child):
            if child.terminal == -1:
                return MCTS._convert_eval(child.reward / child.visits) if child.visits > 0 else float("inf")
            return MCTS._convert_terminal(child.terminal)

        print("Node {")
        print(f"  depth = {node.depth},")
        print(f"  visits = {node.visits},")
        print(f"  terminal = {node.terminal},")
        print("  state =")
        world.print(state)
        print("  Children: {")
        for child in sorted(node.get_children(), key=eval_child, reverse=True):
            print(f"    action: {child.action}, visits: {child.visits}, value: {eval_child(child):.3f}, terminal: {child.terminal}")
        print("  }\n}")

    @staticmethod
    def clear_tree(MCTSNode node) -> None:
        if node is None:
            return
            
        node.parent = None
        for child in node.get_children():
            MCTS.clear_tree(child)

        node.children = None
        node.s_children = 0

    @staticmethod
    def mcts(
        object root_state, 
        object world, 
        int s_rollout, 
        int max_expansion=10,
        int s_initial_rollout=100, 
        double c=1.414,
        bint tree=False, 
        bint debug=False, 
        bint timer=False
    ) -> Tuple[Action, Optional[MCTSNode]]:
        
        cdef double start
        cdef MCTSNode root, node
        cdef State state = root_state
        cdef Action action
        
        if timer:
            start = default_timer()

        root = MCTS._encapsulate(None, max_expansion)
        MCTS._expand(root, root_state, world)
        MCTS._sort_children(root)

        if root.children[0].terminal == 1:
            if debug:
                print("Left winning_actions")
                MCTS._print_node(root, root_state, world, c)
            return root.children[0].action, root
        
        only_action = MCTS._only_action(root, root_state, world, c)
        if only_action is not None:
            if debug:
                print("Left only_action or just lost")
                MCTS._print_node(root, root_state, world, c)
            return only_action, root
        
        MCTS._random_rollout(root, root_state, world, s_initial_rollout)

        for _ in range(s_rollout):
            MCTS._sort_children(root)
            if root.children[0].terminal == 1:
                break

            node, state = MCTS._select(root, root_state, world, c)
            if node is None or node.terminal != -1:
                break

            if node.visits > 0:
                if not MCTS._expand(node, state, world):
                    continue

                node = node.children[randint(0, node.s_children - 1)]
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

        action = MCTS._pick_action(root)
        if not tree:
            MCTS.clear_tree(root)

        return action, None if not tree else root