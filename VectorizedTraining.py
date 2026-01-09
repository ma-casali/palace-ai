import numpy as np
import torch
import random
import tqdm
import copy
import matplotlib.pyplot as plt
import cProfile
import pstats
import time

from VectorizedCardGame import PalaceEnv
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

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")

# region Classes
class LivePlot:
    def __init__(self):
        # Initialize as empty lists for easier appending, convert to arrays for plotting
        self.generation_history = []
        self.reward_history = []
        self.lr_history = []
        self.total_loss_history = []
        self.entropy_history = []
        self.stalemate_history = []
        self.smooth_win = 25
        
        # We'll calculate smoothed values on the fly to avoid list/array mismatches
        self.fig, self.ax = plt.subplots(3, 1, figsize=(10, 6))
        # plt.ion() # Turn on interactive mode

    def update(self, generation, reward, total_loss, lr, entropy, stalemate_rate):
        # Append new data to lists
        self.generation_history.append(generation)
        self.reward_history.append(reward)  # Assuming reward is a list/tensor of 3 players
        self.total_loss_history.append(total_loss)
        self.lr_history.append(lr)
        self.entropy_history.append(entropy)
        self.stalemate_history.append(stalemate_rate)

        # Convert to numpy arrays for indexing/math
        gen_arr = np.array(self.generation_history)
        rew_arr = np.array(self.reward_history) # Shape: (Gen, 3)
        loss_arr = np.array(self.total_loss_history)

        # Clear axes
        for a in self.ax: a.clear()

        # --- 1. REWARD PLOT ---
        # Plot individual player rewards (Red, transparent)
        if rew_arr.ndim > 1:
            for p in range(rew_arr.shape[1]):
                self.ax[0].plot(gen_arr, rew_arr[:, p], color='red', alpha=0.15)
            # Bold Line: Mean of all players
            mean_rew = np.mean(rew_arr, axis=1)
        else:
            mean_rew = rew_arr
        
        # Smoothed line (Window of 10)
        if len(mean_rew) > self.smooth_win:
            smoothed = np.convolve(mean_rew, np.ones(self.smooth_win)/self.smooth_win, mode='valid')
            self.ax[0].plot(gen_arr[self.smooth_win-1:], smoothed, color='red', linewidth=2, label='Smoothed')
            
        self.ax[0].set_title('Total Reward')
        self.ax[0].legend(loc='upper left')

        # --- 2. LOSS PLOT ---
        self.ax[1].plot(gen_arr, loss_arr, color='blue', alpha=0.3)
        if len(loss_arr) > self.smooth_win:
            smoothed_l = np.convolve(loss_arr, np.ones(self.smooth_win)/self.smooth_win, mode='valid')
            self.ax[1].plot(gen_arr[self.smooth_win-1:], smoothed_l, color='blue', linewidth=2)
        
        self.ax[1].set_title('Total Loss')

        # --- 3. LOG SCALE METRICS ---
        self.ax[2].plot(gen_arr, self.lr_history, label='LR', color='green')
        self.ax[2].plot(gen_arr, self.entropy_history, label='Entropy Coeff', color='orange')
        self.ax[2].plot(gen_arr, self.stalemate_history, label='Stalemate Rate', color='purple')
        
        self.ax[2].set_yscale('log')
        self.ax[2].set_title('Training Hyperparameters (Log Scale)')
        self.ax[2].legend(loc='upper right')

        # Formatting
        for a in self.ax: a.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

        return np.abs(np.mean(np.diff(smoothed_l[-self.smooth_win:])) if len(loss_arr) > self.smooth_win else np.abs(np.mean(np.diff(loss_arr))))
# endregion

# region Functions
def get_graph_depth(tensor):
    depth = 0
    curr = tensor.grad_fn
    while curr is not None:
        depth += 1
        curr = curr.next_functions[0][0] if curr.next_functions else None
    return depth

def plot_batch_status(env, fig, ax):
    # Calculate status
    status = np.float32((env.hands.detach() + env.face_up_piles.detach() + env.face_down_piles.detach()).sum(dim=2).cpu().numpy())
    status[status == 0] = np.nan

    # Check if we have already drawn the image
    # We use a custom attribute 'my_im' on the axes to store the reference
    if not hasattr(ax, 'my_im'):
        ax.my_im = ax.imshow(np.transpose(status), aspect='auto', cmap='jet', vmin=0, vmax=20)
        ax.set_xlabel('Game Index')
        ax.set_ylabel('Player Index')
        
        # Create the colorbar ONLY once
        cb = fig.colorbar(ax.my_im, ax=ax)
        cb.set_label('Total Cards Held')
    else:
        # Just update the pixel data
        ax.my_im.set_data(np.transpose(status))

    plt.draw()
    plt.pause(0.001)

def save_best_player(model, filename = 'temporary_king.pth'):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

def create_static_input_vector(env):
    batch_ids = torch.arange(env.batch_size, device=device)
    current_players = env.active_players

    #1. current player states
    hands = env.hands[batch_ids, current_players, :]  # (B, 13)
    faceup= env.face_up_piles[batch_ids, current_players, :]  # (B, 13)
    facedown_len = env.face_down_piles[batch_ids, current_players, :].sum(dim=1, keepdim=True)  # (B, 1)
    parts = [hands, faceup, facedown_len]

    #2. other players' states
    for offset in range(1, env.num_players):
        other_players = (current_players + offset) % env.num_players
        other_hands = env.hands[batch_ids, other_players, :].sum(dim=1, keepdim=True)  # (B, 1)
        other_faceup = env.face_up_piles[batch_ids, other_players, :]  # (B, 13)
        other_facedown_len = env.face_down_piles[batch_ids, other_players, :].sum(dim=1, keepdim=True)  # (B, 1)
        parts.extend([other_hands, other_faceup, other_facedown_len])

    #3. Draw pile length
    discard_counts = env.discard_counts.float()  # (B, 13)
    parts.append(discard_counts)

    #4. Global States
    top_cards = env.top_cards.unsqueeze(1).float()  # (B, 1)
    run_counts = env.run_count.unsqueeze(1).float()  # (B, 1)
    drawpile_len = env.drawpile_counts.sum(dim=1, keepdim=True).float() # (B, 1)
    parts.extend([top_cards, run_counts, drawpile_len])

    return torch.cat(parts, dim = 1).float()  # (B, 82)

def finish_batch_update(optimizer, b_lp, b_ent, b_rew, b_counters, ent_coef, king_available=False):

    def compute_returns(player_rew_buffer, mask, gamma):
        batch_size, max_turns = player_rew_buffer.shape
        returns = torch.zeros_like(player_rew_buffer, device=device)
        running_return = torch.zeros(batch_size, device=device)

        for turn in reversed(range(max_turns)):
            running_return = player_rew_buffer[:, turn] + gamma * running_return * mask[:, turn]
            returns[:, turn] = running_return
        
        return returns[mask]
    
    gamma = 0.98
    start_idx = 1 if king_available else 0
    total_loss = 0.0
    for p_idx in range(start_idx, len(players)):

        max_t = b_lp.shape[2]
        t_indices = torch.arange(max_t, device=device).unsqueeze(0)
        mask = t_indices < b_counters[p_idx].unsqueeze(1)

        if not mask.any():
            continue
        
        flat_lp = b_lp[p_idx][mask]
        flat_ent = b_ent[p_idx][mask]
        
        # Gather returns for the player
        flat_ret = compute_returns(b_rew[p_idx], mask, gamma)
        if flat_ret.numel() > 1:
            flat_ret = (flat_ret - flat_ret.mean()) / (flat_ret.std() + 1e-8)

        # Calculate loss
        policy_loss = -(flat_lp * flat_ret).mean()
        entropy_loss = -ent_coef * flat_ent.mean()
        
        total_loss += (policy_loss + entropy_loss)

    if isinstance(total_loss, torch.Tensor):
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(players[1].parameters(), max_norm=0.5)
        optimizer.step()
        return total_loss.item()

    return 0.0 # Return the device tensor

def training_step(players, best_player_weights, batch_size, sequence_length, level_wins, level_counts, current_level_probs):

    env = PalaceEnv(batch_size = batch_size, num_players = 3, device = device)
    if best_player_weights is not None:
        # P0 is our king, others are learning
        players[0].load_state_dict(best_player_weights)
    for i in range(1, env.num_players):
        players[i].train()

    current_levels = env.reset(players, levels = current_level_probs)
    active_envs = torch.ones(batch_size, dtype=torch.bool, device=device)

    shape = (env.num_players, batch_size, env.max_turns)
    buffer_lp = torch.zeros(shape, device=device)
    buffer_ent = torch.zeros(shape, device=device)
    buffer_rew = torch.zeros(shape, device=device)
    player_step_counters = torch.zeros((env.num_players, batch_size), dtype=torch.long, device=device)

    action_history = -1 * torch.ones((batch_size, sequence_length, 79), device=device)
    hidden_states = [p.init_hidden(batch_size) for p in players]

    pbar_batch = tqdm.tqdm(total = env.max_turns, desc="Training Batch Progress", leave=False)
    last_active_count = batch_size
    turn_count = 0

    while active_envs.any():
        #1. Prepare inputs for the whole batch
        masks = env.get_valid_mask() # (B, 79)
        static_obs = create_static_input_vector(env) # (B, 82)
        current_actors = env.active_players.clone()

        # define environments that are active
        combined_probs = torch.zeros((batch_size, 79), device=device)
        if turn_count == 0:
            hidden_states = [p.init_hidden(batch_size) for p in players]
        for p_idx in range(env.num_players):
            turn_mask = (env.active_players == p_idx) & active_envs  # (B,)
            if turn_mask.any():
                is_king = (p_idx == 0) and (best_player_weights is not None)
                h, c = hidden_states[p_idx]
                h, c = h.detach(), c.detach()

                with torch.no_grad() if is_king else torch.enable_grad():
                    # BATCH INFERENCE
                    h, c = hidden_states[p_idx]
                    h, c = h.detach(), c.detach()
                    probs, (new_h, new_c) = env.players[p_idx](
                        action_history[turn_mask], 
                        static_obs[turn_mask], 
                        masks[turn_mask], 
                        (h[:, turn_mask, :].detach(), c[:, turn_mask, :].detach())
                        )
                    combined_probs[turn_mask] = probs
                    hidden_states[p_idx][0][:, turn_mask, :] = new_h
                    hidden_states[p_idx][1][:, turn_mask, :] = new_c

                player_step_counters[p_idx, turn_mask] += 1

        combined_probs[~active_envs, 0] = 1.0 
        m = torch.distributions.Categorical(combined_probs)
        action_indices = m.sample()  # (B,)

        # Environment step
        rewards, dones = env.step(action_indices)

        # vectorized writing
        log_probs = m.log_prob(action_indices)
        entropies = m.entropy()

        for player_index in range(env.num_players):
            acting_mask = (current_actors == player_index) & active_envs
            if acting_mask.any():
                indices = player_step_counters[player_index, acting_mask]

                buffer_lp[player_index, acting_mask, indices] = log_probs[acting_mask]
                buffer_ent[player_index, acting_mask, indices] = entropies[acting_mask]

        buffer_rew[:, :, turn_count] = rewards.transpose(0, 1)

        #6. Update history (one-hot) and active envs
        new_action = torch.zeros((batch_size, 1, 79), device=device)
        new_action[torch.arange(batch_size), 0, action_indices] = 1.0
        action_history = torch.cat((action_history[:, 1:, :].detach(), new_action.detach()), dim=1)

        active_envs &= ~dones.detach()

        # PBar logic
        turn_count += 1
        if turn_count % 25 == 0:
            cur_count = active_envs.sum().item()
            pbar_batch.update(25)
            pbar_batch.set_postfix({"Active Envs": cur_count})

    total_stalemates = env.stalemates.sum().item()
    stalemate_rate = total_stalemates / batch_size

    pbar_batch.close()

    for lvl in range(4):
        lvl_mask = (current_levels == lvl)
        level_counts[lvl] += lvl_mask.sum().item()
        level_wins[lvl] += (lvl_mask & (env.done & ~env.stalemates)).sum().item()

    return buffer_lp, buffer_rew, buffer_ent, player_step_counters, stalemate_rate, level_wins, level_counts

def update_level_distribution(wins, counts, current_probs):
    win_rates = wins / (counts + 1e-8)
    new_probs = list(current_probs)
    threshold = 0.70 # if win rate is greater than this, then
    shift_amount = 0.05 # shift this much towards next hardest level

    if win_rates[3] > threshold and new_probs[3] > 0.05:
        new_probs[3] -= shift_amount
        new_probs[2] += shift_amount
    if win_rates[2] > threshold and new_probs[2] > 0.05:
        new_probs[2] -= shift_amount
        new_probs[1] += shift_amount
    if win_rates[1] > threshold and new_probs[1] > 0.05:
        new_probs[1] -= shift_amount
        new_probs[0] += shift_amount

    # Normalize
    total = sum(new_probs)
    return [p / total for p in new_probs]

def evaluate_players(players, num_games = 100, sequence_length = 12, max_turns = 1000):
    # num_games acts as our batch_size for evaluation
    env = PalaceEnv(batch_size=num_games, num_players=3, device=device)
    env.reset(players)
    
    # Track losses: in Palace, usually the last player remaining is the loser
    # Or you can track wins (first player out)
    losses = torch.zeros(env.num_players, device=device)
    active_envs = torch.ones(num_games, dtype=torch.bool, device=device)
    action_history = torch.zeros((num_games, sequence_length, 79), device=device)

    for p in players:
        p.eval()

    with torch.inference_mode():
        while active_envs.any():
            masks = env.get_valid_mask()
            static_obs = create_static_input_vector(env)
            actions = torch.zeros(num_games, dtype=torch.long, device=device)

            for p_idx in range(env.num_players):
                turn_mask = (env.active_players == p_idx) & active_envs
                if turn_mask.any():
                    # Greedy selection for evaluation
                    probs, _ = players[p_idx](action_history[turn_mask], static_obs[turn_mask], masks[turn_mask])
                    actions[turn_mask] = torch.argmax(probs, dim=1)

            # Step the environment
            _, dones = env.step(actions)

            # Update history
            new_action = torch.zeros((num_games, 1, 79), device=device)
            new_action[torch.arange(num_games), 0, actions] = 1.0
            action_history = torch.cat((action_history[:, 1:, :], new_action), dim=1)

            # Check which envs just finished
            just_finished = active_envs & dones.detach()
            if just_finished.any():
                # The "loser" is the player who did NOT make it into players_out
                # Your env.players_out likely stores the order of finishers
                # Here we calculate the loser for the finished games
                for env_idx in torch.where(just_finished)[0]:
                    # Find which player ID is not in env.players_out[env_idx]
                    finished_players = env.players_out[env_idx]
                    all_players = torch.arange(env.num_players, device=device)
                    # A player is a loser if they are still 'in' when the game ends
                    # masks for players not in the finished_players list (excluding -1)
                    loser_mask = ~torch.isin(all_players, finished_players)
                    losses[loser_mask] += 1

            active_envs &= ~dones.detach()

    return losses.cpu().numpy()
# endregion

plotter = LivePlot()
all_returns = []
all_entropies = []

num_generations = 3000
batch_size = 256

initial_lr = 1e-2
lr_decay = 0.999
final_lr = 1e-5
current_lr = initial_lr

initial_entropy_coef = 0.1
final_entropy_coef = 0.005
ent_decay = 0.99
current_ent_coef = initial_entropy_coef

last_king_loss = 0.0
sequence_length = 6 # number of actions to track in history

shared_player = PalacePlayer().to(device)
optimizer = torch.optim.Adam(shared_player.parameters(), lr=current_lr)
players = [shared_player for _ in range(3)]

best_player_weights = None

level_wins = torch.zeros(4, device = device)
level_counts = torch.zeros(4, device = device)
current_level_probs = [0.0, 0.0, 0.0, 1.0] # start all at level 3

delta_smoothed_loss = None

# profiler = cProfile.Profile()
# profiler.enable()

try:
    
    for generation in range(num_generations):
        start_time = time.perf_counter()

        # Prepare players for this generation
        b_lp, b_rew, b_ent, player_counters, stalemate_rate, level_wins, level_counts = training_step(players, best_player_weights, batch_size, sequence_length, level_wins, level_counts, current_level_probs)

        # Update level distribution based on performance
        current_level_probs = update_level_distribution(level_wins.cpu().numpy(), level_counts.cpu().numpy(), current_level_probs)

        # Update using the accumulated batch data
        print("Calculating and backpropagating loss...")
        total_loss = finish_batch_update(
            optimizer, 
            b_lp, b_ent, b_rew, player_counters,
            ent_coef=current_ent_coef, 
            king_available=(best_player_weights is not None)
        )           

        if stalemate_rate > 0.3:
            current_lr = min(initial_lr, current_lr * 1.5)
            current_ent_coef = min(initial_entropy_coef, current_ent_coef * 1.5)
        else:
            current_ent_coef = max(final_entropy_coef, current_ent_coef * ent_decay)
            current_lr = max(final_lr, current_lr * lr_decay)

        if (generation % 500 == 0 and generation > 0) or (delta_smoothed_loss is not None and delta_smoothed_loss < 1e-4):
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr

            current_ent_coef = min(initial_entropy_coef, current_ent_coef * 10)

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        print("\n" + "="*50)
        print(f"Generation {generation} completed.")
        print(f"Loss = {total_loss:.4f}, \nLR={current_lr:.6f}, \nEntropy Coef={current_ent_coef:.4f}, \nStalemate Rate={stalemate_rate:.4f}")
        print(f"Mean Turns Taken: {player_counters.sum(dim=0).float().mean().item():.2f}")
        print(f"Level Distribution Probabilities: {[f'{prob:.2f}' for prob in current_level_probs]}")
        print("="*50 + "\n")
        
        # b_rew shape: (NumPlayers, BatchSize, MaxTurns)
        avg_rewards = []
        for p_idx in range(b_rew.shape[0]):
            total_rewards_per_game = b_rew[p_idx].sum(dim=1) 
            
            # Calculate the mean reward across the batch
            avg_rewards.append(total_rewards_per_game.mean().item())

        # Ensure total_loss is a scalar for the plotter
        loss_val = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss

        delta_smoothed_loss = plotter.update(
            generation, 
            avg_rewards, 
            loss_val, 
            lr = current_lr,
            entropy = current_ent_coef,
            stalemate_rate = stalemate_rate,
        )

        if generation % 20 == 0 and generation > 0:
            # region Player Evaluation
            print(f"Evaluating generation {generation} players against current best...")
            
            # Run 60 games at once (20 games per player equivalent)
            loss_history = evaluate_players(players, num_games=60, sequence_length=sequence_length)
            
            # if the king player lost only their equal share of games they stay
            if loss_history[0] <= 20: 
                print(f"King Player lost {loss_history[0]} games and retains the throne.")
            elif loss_history[0] > 20 and not any(loss < 15 for loss in loss_history[1:]):
                print(f"King Player lost {loss_history[0]} games but no challenger proved superior.")
            else:
                new_king_idx = np.argmin(loss_history)
                print(f"A new king player has emerged (Player {new_king_idx}) with only {loss_history[new_king_idx]} losses!")
                best_player_weights = copy.deepcopy(players[new_king_idx].state_dict())
            
            print(f"Loss history: {loss_history}")
            # endregion

        end_time = time.perf_counter()
        print(f"Generation {generation} took {end_time - start_time:.2f} seconds.\n")

    save_best_player(players[0], filename='temporary_king.pth')
    # Save figure
    

except KeyboardInterrupt:
    print("\n[INTERRUPT] Training paused by user. Saving current models...")
    
    # Save the Learner (current weights being optimized)
    torch.save(players[1].state_dict(), "learner_interrupted.pt")
    
    # Save the King (the current best performer)
    save_best_player(players[0], filename='temporary_king.pth')
    print("Models saved successfully. Exiting.")

    plt.savefig('training_progress.png')

# profiler.disable()
# stats = pstats.Stats(profiler).sort_stats('cumtime')
# stats.print_stats(20)

plt.savefig('training_progress.png')