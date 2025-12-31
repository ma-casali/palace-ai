import numpy as np
import torch
import random
import tqdm
import copy
import matplotlib.pyplot as plt

from CardGame import shuffle_deck, get_valid_mask, DrawPile, DiscardPile, PalaceEnv, PalacePlayer

# inputs: 
# - last 3 actions to discard (3 x 13)
# - current hand (1 x 13)
# - logit mask of valid actions (6 x 13) + 1 this must match output!
# --> Flattened (batch_size x 131)

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

def create_input_vector(last_actions, current_hand, valid_mask):
    last_actions_flat = np.array(last_actions).flatten()  # Shape: (39,)
    current_hand_flat = np.array(current_hand).flatten()  # Shape: (13,)
    valid_mask_flat = np.array(valid_mask).flatten()      # Shape: (79,)

    input_vector = np.concatenate([last_actions_flat, current_hand_flat, valid_mask_flat])
    return input_vector  # Shape: (131,)

def finish_batch_update(players, optimizers, log_probs, entropies, rewards, ent_coef, king_available=False):
    gamma = 0.98
    start_idx = 1 if king_available else 0
    player_loss = np.zeros(len(players))

    for i in range(start_idx, len(players)):
        if not batch_log_probs[i]: 
            total_loss = 0.0
            continue
        
        player_policy_losses = []

        # Iterate through each episode in the batch
        for episode_lp, episode_e, episode_r in zip(batch_log_probs[i], batch_entropies[i], batch_rewards[i]):
            # discounted returns
            returns = []
            G = 0
            for r in reversed(episode_r):
                G = r + gamma * G
                returns.insert(0, G)
            
            returns = torch.tensor(returns, dtype = torch.float32).to(device)

            # standardize returns
            if returns.numel() > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            
            # calculate loss for each move
            for log_p, ent, G_t in zip(episode_lp, episode_e, returns):
                loss = -log_p * G_t - ent_coef * ent
                player_policy_losses.append(loss)

        if player_policy_losses:
            optimizer = optimizers[i]
            optimizer.zero_grad()
            total_loss = torch.stack(player_policy_losses).mean() # mean loss over batch
            player_loss[i] = total_loss.detach().cpu().item()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(players[i].parameters(), 1.0) # prevent exploding gradients
            optimizer.step()
    
    return player_loss
    

# endregion

environment = PalaceEnv(num_players=3) # default 2 players
plotter = LivePlot()
all_returns = []
all_entropies = []

num_generations = 100
batch_size = 16
num_episodes = batch_size * 10
learning_rate = 1e-3
max_turns = 1000
initial_entropy_coef = 0.01
final_entropy_coef = 0.001
ent_decay = 0.95
current_ent_coef = initial_entropy_coef
total_loss = np.zeros(environment.num_players)
last_king_loss = 0.0

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
    batch_log_probs = [[] for _ in range(environment.num_players)]
    batch_entropies = [[] for _ in range(environment.num_players)]
    batch_rewards = [[] for _ in range(environment.num_players)]
    episodes_in_batch = 0
    
    # region Episode Training
    for _ in pbar_episode:
        environment.reset(players)

        episode_log_probs = [[] for _ in range(environment.num_players)]
        episode_entropies = [[] for _ in range(environment.num_players)]
        episode_rewards = [[] for _ in range(environment.num_players)]

        action_history = [np.zeros(13) for _ in range(3)]
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
                input_vec = torch.tensor(create_input_vector(action_history, environment.hands[current_player_idx], mask), dtype = torch.float32).to(device)
                mask_vec = torch.tensor(mask.flatten(), dtype = torch.bool).to(device)

                # determine if this is the king player (static player)
                is_static = (current_player_idx == 0 and best_player_weights is not None)

                # use no_grad for static player
                context = torch.no_grad() if is_static else torch.enable_grad()

                with context:
                    probs = environment.players[current_player_idx](input_vec, mask_vec)
                    m = torch.distributions.Categorical(probs)
                    action_idx = m.sample()

                # record the move (only for learning players)
                if not is_static:
                    episode_log_probs[current_player_idx].append(m.log_prob(action_idx))
                    episode_entropies[current_player_idx].append(m.entropy())

                # record the reward received for that move
                prev_player = current_player_idx
                done, action_history, current_player_idx, players_out, s_reward = environment.step(action_idx.cpu().item(), action_history, current_player_idx, players_out)

                # record rewards for all players (helpful for loser penalty)
                episode_rewards[prev_player].append(s_reward)

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
        for i in range(environment.num_players):
            batch_log_probs[i].append(episode_log_probs[i])
            batch_entropies[i].append(episode_entropies[i])
            batch_rewards[i].append(episode_rewards[i])

        episodes_in_batch += 1

        # BATCH UPDATE
        if episodes_in_batch >= batch_size:
            assert len(episode_rewards[i]) ==  len(episode_log_probs[i]), \
                f"Mismatch! Player {i} has {len(episode_rewards[i])} rewards for {len(episode_log_probs[i])} actions."
            # Update the players using the accumulated batch data
            # move batch data to device
            batch_log_probs = [[ [lp.to(device) for lp in ep] for ep in player_eps] for player_eps in batch_log_probs]
            batch_entropies = [[ [e.to(device) for e in ep] for ep in player_eps] for player_eps in batch_entropies]
            total_loss = finish_batch_update(players, optimizers, batch_log_probs, batch_entropies, batch_rewards, ent_coef=current_ent_coef, king_available=(best_player_weights is not None))
            total_loss[0] = last_king_loss  # retain king loss for logging

            #decay entropy coefficient
            current_ent_coef = max(final_entropy_coef, current_ent_coef * ent_decay)

            # Reset buffers
            batch_log_probs = [[] for _ in range(environment.num_players)]
            batch_entropies = [[] for _ in range(environment.num_players)]
            batch_rewards = [[] for _ in range(environment.num_players)]
            episodes_in_batch = 0

        pbar_episode.set_postfix({"Batch: ": f"{episodes_in_batch}/{batch_size}", "Loss": total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss})

        for i in range(environment.num_players):
            if episode_rewards[i]:
                total_gen_reward[i] += sum(episode_rewards[i])

    # endregion

    # average values for plotting
    avg_gen_reward = np.array(total_gen_reward / (num_episodes))
    plotter.update(generation, avg_gen_reward, total_loss)

    # region Player Evaluation
    print("Evaluating players against current best...")
    loss_history = [0 for _ in range(len(players))]
    games_played = 0
    num_games = 20 * environment.num_players  
    max_turns = 1000
    with torch.inference_mode():
        while games_played < num_games:
            environment.reset(players)

            players_out = []
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
                    input_vec = torch.tensor(create_input_vector(action_history, environment.hands[current_player_idx], mask), dtype = torch.float32)
                    mask_vec = torch.tensor(mask.flatten(), dtype = torch.bool)

                    probs = environment.players[current_player_idx](input_vec, mask_vec)
                    action_idx = torch.argmax(probs).item()

                    # record the move
                    prev_player = current_player_idx
                    done, action_history, current_player_idx, players_out, s_reward = environment.step(action_idx, action_history, current_player_idx, players_out)

                turn_count += 1

            if done:
                # Normal completion
                all_indices = set(range(environment.num_players))
                loser_idx = list(all_indices - set(players_out))[0]
                loss_history[loser_idx] += 1
            else:
                # Max turns reached, determine winner by least cards
                cards_counts = [cards_left(environment, idx) for idx in range(environment.num_players)]
                loss_history[np.argmax(cards_counts)] += 1
                
            games_played += 1

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