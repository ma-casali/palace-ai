import numpy as np
import torch
import random
import tqdm
import copy
import matplotlib.pyplot as plt

from CardGame import shuffle_deck, get_valid_mask, DrawPile, DiscardPile, PalaceEnv
from PalacePlayer import PalacePlayer

# inputs: 
# Sequence (N x 13):
# - action history (N x 13)

# Static (27 * (1 + M) + 1 = 82 for 3 players):
# - current hand (1 x 13), current face-up (1 x 13), length face-down (1)
# - other player's hand (M x 13), other player's face-up (M x 13), length other face-down (M)
# - draw pile length (1)

# outputs:
# - softmax action probabilities (6 x 13) + 1
# --> Flattened (1 x 79)

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
print(f"Using device: {device}")

# region Classes
class LivePlot:
    def __init__(self):
        self.generation_history = None
        self.reward_history = None
        self.fig, self.ax = plt.subplots(2, 1, figsize=(15, 10))

    def update(self, generation, reward, total_loss):
        if self.generation_history is None:
            self.generation_history = np.array(generation)
            self.reward_history = np.array([reward])
            self.total_loss_history = np.array([total_loss])
        else:
            self.generation_history = np.append(self.generation_history, generation)
            self.reward_history = np.append(self.reward_history, [reward], axis = 0)
            self.total_loss_history = np.append(self.total_loss_history, [total_loss], axis = 0)

        self.ax[0].clear()
        self.ax[1].clear()
        for i in range(len(reward)):
            if i == 0:
                self.ax[0].plot(self.generation_history, self.reward_history[:,i], color = 'red', alpha = 1.0)
            else:
                self.ax[0].plot(self.generation_history, self.reward_history[:,i], color = 'red', alpha = 0.3)
        self.ax[0].set_xlabel('Generation')
        self.ax[0].set_ylabel('Total Reward')
        self.ax[0].set_title('Training Progress - Total Reward')
        self.ax[0].grid(True)

        for i in range(len(total_loss)):
            if i == 0:
                self.ax[1].plot(self.generation_history, self.total_loss_history[:,i], color = 'blue', alpha = 1.0)
            else:
                self.ax[1].plot(self.generation_history, self.total_loss_history[:,i], color = 'blue', alpha = 0.3)
        self.ax[1].plot(self.generation_history, self.total_loss_history, color = 'blue')
        self.ax[1].set_xlabel('Generation')
        self.ax[1].set_ylabel('Loss')
        self.ax[1].set_title('Training Progress - Loss')
        self.ax[1].grid(True)

        plt.draw()
        plt.pause(0.001)
# endregion

# region Functions
def save_best_player(model, filename = 'Palace_king.pth'):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

def get_sum_value(values):
    sum = [0 for _ in range(1, len(values))]
    if not any(values):
        return sum
    for i in range(1, len(values)):
        sum[i - 1] += np.sum(values[i])

def cards_left(env, player_idx):
    hand_count = np.sum(env.hands[player_idx])
    face_up_count = np.sum(env.face_up_piles[player_idx])
    face_down_count = np.sum(env.face_down_piles[player_idx])
    return int(hand_count + face_up_count + face_down_count)

def update_action_history(action_history, new_action, N = 13):
    # action_history shape: (1, N, 79)
    # new_action: int (0-78)

    # create one-hot encoding for new action
    new_action_one_hot = torch.zeros((1, 1, 79)).to(device)
    if new_action != 1: 
        new_action_one_hot[0, 0, new_action] = 1.0

    # slide the window
    updated_history = torch.cat((action_history[:, 1:, :], new_action_one_hot), dim=1)

    return updated_history  # shape: (1, N, 79)

def create_static_input_vector(env, current_player_idx):
    input_vector = torch.zeros((27 * env.num_players + 1,), dtype = torch.float32, device = device)

    input_vector[:13] = env.hands[current_player_idx]
    input_vector[13:26] = env.face_up_piles[current_player_idx]
    input_vector[26] = torch.sum(env.face_down_piles[current_player_idx])
    offset = 27
    for i in range(env.num_players):
        if i != current_player_idx:
            input_vector[offset:offset + 13] = env.hands[i]
            input_vector[offset + 13:offset + 26] = env.face_up_piles[i]        
            input_vector[offset + 26] = torch.sum(env.face_down_piles[i])
            offset += 27

    input_vector[-1] = len(env.draw_pile.deck)
    return input_vector.unsqueeze(0)  # Shape: (1, 27 * env.num_players + 1)

def finish_batch_update(players, optimizers, b_hist, b_stat, b_mask, b_lp, b_ent, b_rew, ent_coef, king_available=False):
    gamma = 0.98
    start_idx = 1 if king_available else 0

    player_losses = torch.zeros(len(players), device=device)

    for i in range(start_idx, len(players)):
        if not b_lp[i]: continue
        
        flat_ent = torch.cat([torch.stack(ep) for ep in b_ent[i]]).flatten()
        player_eps_lp = [torch.stack(ep) for ep in b_lp[i] if len(ep) > 0]
        if not player_eps_lp: continue
        flat_lp = torch.cat(player_eps_lp).flatten()
        
        # 2. Gather returns for THIS player
        flat_returns = []
        for ep_lp, ep_rew in zip(b_lp[i], b_rew[i]):
            # CRITICAL CHECK: Does the number of rewards match the number of actions?
            if len(ep_lp) != len(ep_rew):
                # If rewards are longer (e.g. terminal reward), trim or merge them
                # This ensures ep_rew matches ep_lp 1:1
                ep_rew = ep_rew[:len(ep_lp)] 
                
            G = 0
            returns = torch.zeros(len(ep_rew), device=device)
            for t in reversed(range(len(ep_rew))):
                G = ep_rew[t] + gamma * G
                returns[t] = G
            flat_returns.append(returns)
        
        returns_tensor = torch.cat(flat_returns)

        # 3. Final Check before multiplication
        if flat_lp.shape[0] != returns_tensor.shape[0]:
            raise RuntimeError(f"Player {i} mismatch: {flat_lp.shape[0]} actions vs {returns_tensor.shape[0]} rewards")

        loss = -(flat_lp * returns_tensor).mean() - ent_coef * flat_ent.mean()

        optimizer = optimizers[i]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(players[i].parameters(), 1.0)
        optimizer.step()
        b_lp[i].clear()
        b_ent[i].clear()

        player_losses[i] = loss.detach() # Stay on device
    
    return player_losses # Return the device tensor
    

# endregion

environment = PalaceEnv(num_players=3) # default 2 players
# plotter = LivePlot()
all_returns = []
all_entropies = []

num_generations = 100
batch_size = 8
num_episodes = batch_size * 10
learning_rate = 1e-3
max_turns = 1000
initial_entropy_coef = 0.05
final_entropy_coef = 0.005
ent_decay = 0.95
current_ent_coef = initial_entropy_coef
total_loss = np.zeros(environment.num_players)
last_king_loss = 0.0
N = 12 # number of actions to track in history

players = [PalacePlayer().to(device) for _ in range(environment.num_players)]
optimizers = [torch.optim.Adam(player.parameters(), lr=learning_rate) for player in players]

best_player_weights = None

for generation in range(num_generations):
    # Prepare players for this generation
    if best_player_weights is not None:
        # P0 is our king, others are learning
        players[0].load_state_dict(best_player_weights)
        players[0].to(device)

    for i in range(1, environment.num_players):
        players[i].train()

    pbar_episode = tqdm.tqdm(range(num_episodes), desc = f"Generation {generation}")
    total_gen_reward = np.zeros(environment.num_players)

    # batch buffers
    batch_histories = [[] for _ in range(environment.num_players)]
    batch_states = [[] for _ in range(environment.num_players)]
    batch_masks = [[] for _ in range(environment.num_players)]

    batch_log_probs = [[] for _ in range(environment.num_players)]
    batch_entropies = [[] for _ in range(environment.num_players)]
    batch_rewards = [[] for _ in range(environment.num_players)]
    episodes_in_batch = 0
    
    # region Episode Training
    for _ in pbar_episode:
        environment.reset(players)

        # per-episode buffers
        episode_histories = [[] for _ in range(environment.num_players)]
        episode_states = [[] for _ in range(environment.num_players)]
        episode_masks = [[] for _ in range(environment.num_players)]

        episode_log_probs = [[] for _ in range(environment.num_players)]
        episode_entropies = [[] for _ in range(environment.num_players)]
        episode_rewards = [[] for _ in range(environment.num_players)]

        action_history = -1*torch.ones((1, N, 79), dtype = torch.float32).to(device)  # Initialize with -1 (no action)
        players_out = [] # track players who have finished their game
        done = False
        turn_count = 0
        current_player_idx = 0

        while not done and turn_count < max_turns:
            if current_player_idx not in players_out:

                # make action decision
                mask = get_valid_mask(
                    environment.hands[current_player_idx],
                    environment.discard_pile,
                    environment.face_up_piles[current_player_idx],
                    environment.face_down_piles[current_player_idx]
                )
                state_input = create_static_input_vector(environment, current_player_idx)
                sequence_input = action_history.detach()
                mask_vec = mask.flatten()

                # determine if this is the king player (static player)
                is_static = (current_player_idx == 0 and best_player_weights is not None)

                # use no_grad for static player
                context = torch.no_grad() if is_static else torch.enable_grad()

                with context:
                    probs = environment.players[current_player_idx](sequence_input, state_input, mask_vec)
                    m = torch.distributions.Categorical(probs)
                    action_idx = m.sample()

                # update action history
                action_history = update_action_history(action_history, action_idx, N = N)

                # record the reward received for that move
                prev_player = current_player_idx
                done, current_player_idx, players_out, s_reward = environment.step(action_idx.cpu(), current_player_idx, players_out)

                player_was_static = (prev_player == 0 and best_player_weights is not None)
                if not player_was_static:
                    # store log prob and entropy
                    episode_log_probs[prev_player].append(m.log_prob(action_idx))
                    episode_entropies[prev_player].append(m.entropy())
                    episode_rewards[prev_player].append(s_reward)
                    # store inputs for this turn
                    episode_histories[prev_player].append(sequence_input.squeeze(0).detach())
                    episode_states[prev_player].append(state_input.squeeze(0).detach())
                    episode_masks[prev_player].append(mask_vec.detach())

            turn_count += 1

        if done:
            all_indices = set(range(environment.num_players))
            loser_idx = list(all_indices - set(players_out))[0]
            # add loser penalty to the very last action
            if loser_idx:
                if episode_rewards[loser_idx]:
                    winner_idx = players_out[0]
                    if episode_rewards[winner_idx]:
                        penalty = episode_rewards[winner_idx][-1]
                        episode_rewards[loser_idx][-1] -= penalty
                else:
                    print(f"Warning: Player {loser_idx} has no recorded rewards in this episode.")
        else:
            # penalty for stalemate
            for i in range(environment.num_players):
                if i not in players_out and len(episode_rewards[i]) > 0:
                    episode_rewards[i][-1] -= 10.0

        # Add this episode's data to the batch buffers
        assert len(episode_rewards[i]) ==  len(episode_log_probs[i]), \
            f"Mismatch! Player {i} has {len(episode_rewards[i])} rewards for {len(episode_log_probs[i])} actions."
        for i in range(environment.num_players):
            batch_histories[i].append(episode_histories[i])
            batch_states[i].append(episode_states[i])
            batch_masks[i].append(episode_masks[i])

            batch_log_probs[i].append(episode_log_probs[i])
            batch_entropies[i].append(episode_entropies[i])
            batch_rewards[i].append(episode_rewards[i])

        episodes_in_batch += 1

        # BATCH UPDATE
        if episodes_in_batch >= batch_size:
            
            # Update the players using the accumulated batch data
            total_loss = finish_batch_update(
                players,
                optimizers, 
                batch_histories, 
                batch_states, 
                batch_masks, 
                batch_log_probs, 
                batch_entropies, 
                batch_rewards, 
                ent_coef=current_ent_coef, 
                king_available=(best_player_weights is not None)
            )            
            total_loss[0] = last_king_loss  # retain king loss for logging

            #decay entropy coefficient
            current_ent_coef = max(final_entropy_coef, current_ent_coef * ent_decay)

            # reset batch buffers
            batch_histories = [[] for _ in range(environment.num_players)]
            batch_states = [[] for _ in range(environment.num_players)]
            batch_masks = [[] for _ in range(environment.num_players)]

            batch_log_probs = [[] for _ in range(environment.num_players)]
            batch_entropies = [[] for _ in range(environment.num_players)]
            batch_rewards = [[] for _ in range(environment.num_players)]
            episodes_in_batch = 0

        for i in range(environment.num_players):
            if episode_rewards[i]:
                total_gen_reward[i] += sum(episode_rewards[i])

    # endregion

    # average values for plotting
    avg_gen_reward = np.array(total_gen_reward / (num_episodes))
    # plotter.update(generation, avg_gen_reward, total_loss)

    # region Player Evaluation
    print(f"Evaluating generation {generation} players against current best...")
    loss_history = [0 for _ in range(len(players))]
    games_played = 0
    num_games = 20 * environment.num_players  
    max_turns = 1000
    N = 12 # Sequence length

    # Ensure all players are in eval mode
    for p in players:
        p.eval()

    with torch.inference_mode():
        while sum(loss_history) < num_games:
            environment.reset(players)
            
            # Initialize evaluation action history (Batch, Seq, Dim)
            action_history = torch.zeros((1, N, 79), device=device)
            players_out = []
            done = False
            turn_count = 0 
            current_player_idx = 0  

            while not done and turn_count < max_turns:
                if current_player_idx not in players_out:
                    # 1. Generate Inputs
                    mask = get_valid_mask(
                        environment.hands[current_player_idx],
                        environment.discard_pile,
                        environment.face_up_piles[current_player_idx],
                        environment.face_down_piles[current_player_idx]
                    )
                    
                    # Prepare Static Vector (1, 82)
                    state_input = create_static_input_vector(environment, current_player_idx)
                    # Use current history (1, 12, 79)
                    sequence_input = action_history 
                    
                    # 2. Get Model Prediction
                    # Model now expects (history, static, mask)
                    probs = players[current_player_idx](sequence_input, state_input, mask)
                    action_idx = torch.argmax(probs).item()

                    # 3. Environment Step
                    # environment.step returns updated state and info
                    prev_player = current_player_idx
                    done, current_player_idx, players_out, _ = environment.step(
                        action_idx, current_player_idx, players_out
                    )

                    # 4. Update History with the action taken
                    # Shift history and add the new action as a one-hot vector
                    new_action_oh = torch.zeros((1, 1, 79), device=device)
                    new_action_oh[0, 0, action_idx] = 1.0
                    action_history = torch.cat((action_history[:, 1:, :], new_action_oh), dim=1)

                else:
                    # If current player is out, rotate to next
                    current_player_idx = (current_player_idx + 1) % environment.num_players

                turn_count += 1

            # Determine "Loser" for this evaluation game
            if done:
                all_indices = set(range(environment.num_players))
                loser_idx = list(all_indices - set(players_out))[0]
                loss_history[loser_idx] += 1
            else:
                # Stalemate: All players lose equally
                loss_history = [loss + 1 for loss in loss_history]  

    # if the king player lost only their equal share of games they stay
    if loss_history[0] <= 20: # king player won at least 20 games
        print(f"King Player lost {loss_history[0]} games and retains the throne.")
        print(f"Loss history: {loss_history}")
        last_king_loss = total_loss[0]
    elif loss_history[0] > 20 and not any(loss < 15 for loss in loss_history[1:]):
        print(f"King Player lost {loss_history[0]} games but no challenger proved superior. King retains the throne.")
        print(f"Loss history: {loss_history}")
        last_king_loss = total_loss[0]
    else:
        print(f"A new king player has emerged and they only lost {loss_history[ np.argmin(loss_history) ]} games!")
        best_player_weights = copy.deepcopy(players[np.argmin(loss_history)].state_dict())
        print(f"Loss history: {loss_history}")
        last_king_loss = total_loss[np.argmin(loss_history)]

    # endregion

save_best_player(players[0], filename='Palace_king.pth')
# Save figure
plt.savefig('training_progress.png')
plt.show()