from game.chess_module import ChessModule
from chess import Board
import torch
import numpy as np

class Node:
    def __init__(self, game: ChessModule, args: dict, state: Board, parent=None, action_taken=None, prior=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children = []

        self.visit_count = 0
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.children) > 0
    
    def select(self):
        pass

    def get_ucb(self, child):
        pass

    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action)
                
                child = Node(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = -value
        if self.parent is not None:
            self.parent.backpropagate(value)

class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, state: Board):
        root = Node(self.args, state)

        for _ in range(self.args['num_searches']):
            node = root

            while node.is_fully_expanded():
                node = node.select()

            is_terminal, value = ChessModule.check_termination_and_get_value(node.state)

            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(ChessModule.get_state_encoding(node.state)).unsqueeze(0) #FIX
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy() #FIX
                valid_moves = ChessModule.get_valid_moves(node.state) #FIX
                policy *= valid_moves
                policy /= np.sum(policy)

                value = value.item()
                node.expand(policy)

            node.backpropagate(value)


        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs

