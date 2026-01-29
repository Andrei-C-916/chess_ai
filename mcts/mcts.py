from game.chess_module import ChessModule
from chess import Board
import torch
import numpy as np
import chess
import math

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

    def is_expanded(self):
        return len(self.children) > 0
    
    def select(self):
        best_child = None
        best_ucb = -np.inf
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
        return best_child

    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior

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
    def __init__(self, game: ChessModule, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, state: Board):
        root = Node(self.game, self.args, state)

        for _ in range(self.args['num_searches']):
            node = root

            while node.is_expanded():
                node = node.select()

            is_terminal, outcome = self.game.check_termination_and_get_value(node.state)
            if is_terminal:
                if node.state.turn == chess.WHITE:   # white to move just lost ⇒ outcome = -1 already
                    value = outcome
                else:                                # black to move just lost ⇒ outcome = +1, need -1
                    value = -outcome

            else:
                policy, value = self.model(
                    torch.tensor(self.game.get_state_encoding(node.state), dtype=torch.float32).unsqueeze(0)
                )
                value = value.item()
                policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                sum_policy = np.sum(policy)
                if sum_policy > 0:
                    policy /= sum_policy
                else:
                    policy = valid_moves / np.sum(valid_moves)

                # dirichlet noise to increase diversity
                if node is root and self.args.get("dirichlet_eps", 0) > 0:
                    eps = self.args["dirichlet_eps"]
                    alpha = self.args["dirichlet_alpha"]

                    valid_idx = np.where(valid_moves > 0)[0]
                    noise = np.random.dirichlet([alpha] * len(valid_idx))

                    policy2 = policy.copy()
                    policy2[valid_idx] = (1 - eps) * policy2[valid_idx] + eps * noise
                    policy = policy2 / np.sum(policy2)

                node.expand(policy)

            node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs

