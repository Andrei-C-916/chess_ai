from mcts.mcts import MCTS
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import trange
from game.chess_module import ChessModule
import chess

class AlphaZero:
    def __init__(self, model, optimizer, game: ChessModule, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)

    def self_play(self):
        memory = []
        state = self.game.get_initial_state()

        while True:
            action_probs = self.mcts.search(state)
            memory.append((state.copy(), action_probs))
            move = np.random.choice(self.game.action_size, p=action_probs)
            state = self.game.get_next_state(state, move) 
            is_terminal, value = self.game.check_termination_and_get_value(state)
            if is_terminal:
                returnMemory = []
                for state, action_probs in memory:
                    if state.turn == chess.WHITE:
                        returnMemory.append((self.game.get_state_encoding(state), action_probs, value))
                    else:
                        returnMemory.append((self.game.get_state_encoding(state), action_probs, -value))
                return returnMemory

    def train(self, memory):
        np.random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory), batchIdx + self.args['batch_size'])]

            state, policy_targets, value_targets = zip(*sample)
            state = np.array(state)
            policy_targets = np.array(policy_targets)
            value_targets = np.array(value_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.device)

            out_policy, out_value = self.model(state)

            log_policy = F.log_softmax(out_policy, dim=1)
            policy_loss = -(policy_targets * log_policy).sum(dim=1).mean()
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []

            self.model.eval()
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations']):
                memory += self.self_play()

            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)

            torch.save(self.model.state_dict(), f"model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")