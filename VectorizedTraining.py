import numpy as np
import torch
import wandb
import tqdm
import copy
import matplotlib.pyplot as plt
import cProfile
import pstats
import time
import os
import multiprocessing

from VectorizedCardGame import PalaceEnv
from PalacePlayer import PalacePlayer


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
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
        self.game_length_history = []
        self.smooth_win = 25
        
        # We'll calculate smoothed values on the fly to avoid list/array mismatches
        self.fig, self.ax = plt.subplots(4, 1, figsize=(10, 8))
        # plt.ion() # Turn on interactive mode

    def update(self, generation, reward, total_loss, lr, entropy, stalemate_rate, game_length):
        # Append new data to lists
        self.generation_history.append(generation)
        self.reward_history.append(reward)  # Assuming reward is a list/tensor of 3 players
        self.total_loss_history.append(total_loss)
        self.lr_history.append(lr)
        self.entropy_history.append(entropy)
        self.stalemate_history.append(stalemate_rate)
        self.game_length_history.append(game_length)  # Placeholder if needed later

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
        
        # Smoothed line 
        if len(mean_rew) > self.smooth_win:
            smoothed = np.convolve(mean_rew, np.ones(self.smooth_win)/self.smooth_win, mode='valid')
            self.ax[0].plot(gen_arr[self.smooth_win-1:], smoothed, color='red', linewidth=2, label='Smoothed')
            
        self.ax[0].set_title('Total Reward')

        # --- 2. LOSS PLOT ---
        self.ax[1].plot(gen_arr, loss_arr, color='blue', alpha=0.3)
        if len(loss_arr) > self.smooth_win:
            smoothed_l = np.convolve(loss_arr, np.ones(self.smooth_win)/self.smooth_win, mode='valid')
            self.ax[1].plot(gen_arr[self.smooth_win-1:], smoothed_l, color='blue', linewidth=2)
        self.ax[1].set_title('Total Loss')

        # --- 3. METRICS ---
        self.ax[2].plot(gen_arr, self.stalemate_history, label='Stalemate Rate (%)', color='purple')
        self.ax[2].plot(gen_arr, self.game_length_history, label='Mean Game Length (turns)', color='brown')
        self.ax[2].set_title('Training Metrics')
        self.ax[2].legend(loc='best')

        # --- 4. TRAINING PARAMETERS ---
        self.ax[3].plot(gen_arr, self.lr_history, label='LR', color='green')
        self.ax[3].plot(gen_arr, self.entropy_history, label='Entropy Coeff', color='orange')
        
        self.ax[3].set_yscale('log')
        self.ax[3].set_title('Training Hyperparameters (Log Scale)')
        self.ax[3].legend(loc='best')

        # Formatting
        for a in self.ax: a.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

        # Calculate delta of smoothed loss for convergence monitoring
        delta_smoothed_loss = np.abs(np.mean(np.diff(smoothed_l[-self.smooth_win:])) if len(loss_arr) > self.smooth_win else np.abs(np.mean(np.diff(loss_arr))))
        delta_smoothed_reward = np.abs(np.mean(np.diff(smoothed[-self.smooth_win:])) if len(mean_rew) > self.smooth_win else np.abs(np.mean(np.diff(mean_rew))))
        
        return delta_smoothed_loss, delta_smoothed_reward
# endregion

# region Functions

def save_best_player(model, filename = 'temporary_king.pth'):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

def create_static_input_vector(env):
    batch_ids = torch.arange(env.batch_size, device=device)
    current_players = env.active_players

    #1. Current player states
    hands = env.hands[batch_ids, current_players, :]  # (B, 13)
    faceup= env.face_up_piles[batch_ids, current_players, :]  # (B, 13)
    facedown_len = env.face_down_piles[batch_ids, current_players, :].sum(dim=1, keepdim=True)  # (B, 1)
    parts = [hands, faceup, facedown_len]

    #2. Other players' states
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
    drawpile_len = 52 - env.drawpile_ptrs.view(-1, 1).float() # (B, 1)
    parts.extend([top_cards, run_counts, drawpile_len])

    return torch.cat(parts, dim = 1).float()  # (B, 73)

def finish_batch_update(optimizer, players, b_lp, b_ent, b_rew, b_counters, ent_coef, king_available=False):

    """
    Docstring for finish_batch_update
    
    :param optimizer: optimizer object used for training
    :param b_lp: batch log probabilities tensor
    :param b_ent: batch entropies tensor
    :param b_rew: batch rewards tensor
    :param b_counters: batch counters tensor -- number of steps taken per player
    :param ent_coef: entropy coefficient for loss calculation
    :param king_available: boolean indicating if a king player is present
    :return: total loss value after backpropagation
    """

    def compute_returns(player_rew_buffer, mask, gamma):
        batch_size, max_turns = player_rew_buffer.shape
        returns = torch.zeros_like(player_rew_buffer, device=device)
        running_return = torch.zeros(batch_size, device=device)

        # reduce the impact of rewards from earlier turns
        for turn in reversed(range(max_turns)):
            running_return = player_rew_buffer[:, turn] + gamma * running_return * mask[:, turn]
            returns[:, turn] = running_return
        
        return returns[mask]
    
    gamma = 0.98 # reduction factor for returns
    start_idx = 1 if king_available else 0
    total_loss = 0.0
    for p_idx in range(start_idx, len(players)):

        max_t = b_lp.shape[2]
        t_indices = torch.arange(max_t, device=device).unsqueeze(0)
        mask = t_indices < b_counters[p_idx].unsqueeze(1)

        if not mask.any(): continue
        
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

    """
    Docstring for training_step
    
    :param players: list of player models (typically PalacePlayer instances)
    :param best_player_weights: the current best player weights (state_dict) or None if there is no current king
    :param batch_size: The number of parallel game environments to run
    :param sequence_length: Length of action history to maintain for LSTM input
    :param level_wins: The number of batch games won per level
    :param level_counts: The number of batch games played per level
    :param current_level_probs: The probability distribution over levels to sample from
    :return: tuple of (buffer_lp, buffer_rew, buffer_ent, player_step_counters, stalemate_rate, level_wins, level_counts)
    """

    env = PalaceEnv(batch_size = batch_size, num_players = 3, device = device)
    if best_player_weights is not None:
        # P0 is our king, others are learning
        players[0].load_state_dict(best_player_weights)
    for i in range(1, env.num_players):
        players[i].train()

    # reset the environment so that all batch games start fresh
    current_levels = env.reset(players, levels = current_level_probs)
    active_envs = torch.ones(batch_size, dtype=torch.bool, device=device)

    # buffers to store log probs, entropies, rewards
    shape = (env.num_players, batch_size, env.max_turns)
    buffer_lp = torch.zeros(shape, device=device)
    buffer_ent = torch.zeros(shape, device=device)
    buffer_rew = torch.zeros(shape, device=device)

    # determining step counters per player for masking in loss calculation
    player_step_counters = torch.zeros((env.num_players, batch_size), dtype=torch.long, device=device)

    action_history = -1 * torch.ones((batch_size, sequence_length, 79), device=device)
    ptr = 0

    # pbar_batch = tqdm.tqdm(total = env.max_turns, desc="Training Batch Progress", leave=False)

    # initialize hidden states for all players (num_players, num_layers, batch_size, hidden_size)
    h_all = torch.zeros((env.num_players, players[0].num_rnn_layers, batch_size, players[0].hidden_dim), device=device)
    c_all = torch.zeros((env.num_players, players[0].num_rnn_layers, batch_size, players[0].hidden_dim), device=device)
   
    turn_count = 0
    while active_envs.any():
        #1. Prepare inputs for the whole batch
        masks = env.get_valid_mask() # (B, 79)
        static_obs = create_static_input_vector(env) # (B, 73)
        current_actors = env.active_players.clone()

        # define environments that are active
        combined_probs = torch.zeros((batch_size, 79), device=device)
        if turn_count == 0:
            hidden_states = [p.init_hidden(batch_size) for p in players]
        for p_idx in range(env.num_players):
            turn_mask = (env.active_players == p_idx) & active_envs  # (B,)
            if turn_mask.any():
                is_king = (p_idx == 0) and (best_player_weights is not None)
                h_in = h_all[p_idx, :, turn_mask, :].detach()
                c_in = c_all[p_idx, :, turn_mask, :].detach()

                with torch.no_grad() if is_king else torch.enable_grad():
                    # BATCH INFERENCE
                    probs, (new_h, new_c) = env.players[p_idx](
                        action_history[turn_mask], 
                        static_obs[turn_mask], 
                        masks[turn_mask], 
                        (h_in, c_in)
                        )
                    combined_probs[turn_mask] = probs
                    h_all[p_idx, :, turn_mask, :] = new_h
                    c_all[p_idx, :, turn_mask, :] = new_c

                player_step_counters[p_idx, turn_mask] += 1

        combined_probs[~active_envs, 0] = 1.0 
        m = torch.distributions.Categorical(combined_probs)
        action_indices = m.sample()  # (B,)

        # Environment step -- game rules applied here
        rewards, dones = env.step(action_indices)

        # vectorized writing
        log_probs = m.log_prob(action_indices)
        entropies = m.entropy()

        buffer_lp[current_actors, torch.arange(batch_size, device=device), turn_count] = log_probs
        buffer_ent[current_actors, torch.arange(batch_size, device=device), turn_count] = entropies

        buffer_rew[:, :, turn_count] = rewards.transpose(0, 1)

        #6. Update history (one-hot) and active envs
        new_action = torch.zeros((batch_size, 1, 79), device=device)
        new_action[torch.arange(batch_size), 0, action_indices] = 1.0
        action_history[:, ptr, :] = 0 # clear out old action
        action_history[torch.arange(batch_size), ptr, action_indices] = 1.0
        ptr = (ptr + 1) % sequence_length

        active_envs &= ~dones.detach()

        # PBar logic
        turn_count += 1
        # if turn_count % 25 == 0:
        #     cur_count = active_envs.sum().item()
            # pbar_batch.update(25)
            # pbar_batch.set_postfix({"Active Envs": cur_count})

    total_stalemates = env.stalemates.sum().item()
    stalemate_rate = total_stalemates / batch_size

    # pbar_batch.close()

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
    secondary_threshold = 0.30 # if win rate is less than this, shift towards easier level

    if win_rates[3] > threshold and new_probs[3] > 0.05:
        new_probs[3] -= shift_amount
        new_probs[2] += shift_amount

    if win_rates[2] > threshold and new_probs[2] > 0.05:
        new_probs[2] -= shift_amount
        new_probs[1] += shift_amount
    elif win_rates[2] < secondary_threshold and new_probs[2] > 0.05:
        new_probs[3] += shift_amount 
        new_probs[2] -= shift_amount

    if win_rates[1] > threshold and new_probs[1] > 0.05:
        new_probs[1] -= shift_amount
        new_probs[0] += shift_amount
    elif win_rates[1] < secondary_threshold and new_probs[1] > 0.05:
        new_probs[2] += shift_amount 
        new_probs[1] -= shift_amount

    if win_rates[0] < secondary_threshold and new_probs[0] > 0.05:
        new_probs[1] += shift_amount
        new_probs[0] -= shift_amount

    # Normalize
    total = sum(new_probs)
    return [p / total for p in new_probs]

def evaluate_players(players, num_games=100, sequence_length=6, max_turns=500):
    # Initialize environment
    env = PalaceEnv(batch_size=num_games, num_players=3, device=device)
    env.reset(players)
    
    # Initialize tracking
    losses = torch.zeros(env.num_players, device=device)
    active_envs = torch.ones(num_games, dtype=torch.bool, device=device)
    
    # Pre-allocate hidden states for all players
    hidden_states = [p.init_hidden(num_games) for p in players]
    
    # Initialize History
    action_history = torch.zeros((num_games, sequence_length, 79), device=device)

    for p in players:
        p.eval()

    turn_count = 0
    with torch.inference_mode():
        while active_envs.any() and turn_count < max_turns:
            masks = env.get_valid_mask()
            static_obs = create_static_input_vector(env)
            actions = torch.zeros(num_games, dtype=torch.long, device=device)

            for p_idx in range(env.num_players):
                # Identify which games in the batch are currently this player's turn
                turn_mask = (env.active_players == p_idx) & active_envs
                
                if turn_mask.any():
                    # pass the hidden states to the model
                    h, c = hidden_states[p_idx]
                    probs, (new_h, new_c) = players[p_idx](
                        action_history[turn_mask], 
                        static_obs[turn_mask], 
                        masks[turn_mask],
                        (h[:, turn_mask, :], c[:, turn_mask, :])
                    )
                    
                    actions[turn_mask] = torch.argmax(probs, dim=1)
                    
                    # Update hidden states for the next time this player acts
                    hidden_states[p_idx][0][:, turn_mask, :] = new_h
                    hidden_states[p_idx][1][:, turn_mask, :] = new_c

            # Step the environment
            _, dones = env.step(actions)

            # update history for active environments
            if active_envs.any():
                new_action_slice = torch.zeros((num_games, 79), device=device)
                new_action_slice[torch.arange(num_games), actions] = 1.0
                
                updated_history = torch.cat((action_history[:, 1:, :], new_action_slice.unsqueeze(1)), dim=1)
                action_history = torch.where(active_envs.view(-1, 1, 1), updated_history, action_history)

            # track losses for each player
            just_finished = active_envs & dones
            if just_finished.any():
                # The loser is the player who never finished (finish_time remains 0)
                loser_mask = (env.finish_times == 0) & just_finished.unsqueeze(1)
                losses += loser_mask.sum(dim=0).float()

            active_envs &= ~dones
            turn_count += 1

        # Handle Stalemates: If turns exceed max, whoever is still playing is a loser
        if active_envs.any():
            remaining_losers = (env.finish_times == 0) & active_envs.unsqueeze(1)
            losses += remaining_losers.sum(dim=0).float()

    return losses.cpu().numpy()

def get_dynamic_entropy(generation, cycle_len = 250, start_ent = 0.05, end_ent = 0.005):
    cycle_count = generation // cycle_len
    current_start = start_ent * (0.5 ** cycle_count)
    progress = (generation % cycle_len) / cycle_len
    ent_coef = current_start - (current_start - end_ent) * progress

    return max(ent_coef, end_ent)

def sweep_train():
    run = wandb.init()
    
    try:
        config = wandb.config

        # training length
        num_generations = 500
        batch_size = 256

        # learning rate
        initial_lr = config.initial_lr
        final_lr = config.final_lr

        # player parameters
        sequence_length = 6 # number of actions to track in history
        shared_player = PalacePlayer().to(device)
        players = [shared_player for _ in range(3)]

        # optimizer
        optimizer = torch.optim.Adam(shared_player.parameters(), lr=initial_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=150, T_mult=2, eta_min=final_lr)

        # level tracking
        level_wins = torch.zeros(4, device = device)
        level_counts = torch.zeros(4, device = device)
        current_level_probs = [0.0, 0.0, 0.0, 1.0] # start all at level 3

        # misc. parameter initializations
        best_player_weights = None
        patience = 50
        best_patience_criterion = 1.0 
        patience_counter = 0
        min_delta = 0.001
            
        for generation in range(num_generations):

            # Take a training step
            b_lp, b_rew, b_ent, player_counters, stalemate_rate, level_wins, level_counts = training_step(players, best_player_weights, batch_size, sequence_length, level_wins, level_counts, current_level_probs)
            mean_turns = player_counters.sum(dim=0).float().mean().item()
            
            # Update level distribution based on performance
            current_level_probs = update_level_distribution(level_wins.cpu().numpy(), level_counts.cpu().numpy(), current_level_probs)

            # adapt entropy coefficient according to learning rate schedule
            current_ent_coef = get_dynamic_entropy(generation, start_ent=config.initial_ent, end_ent=config.final_ent)

            # Update using the accumulated batch data
            total_loss = finish_batch_update(
                optimizer, 
                players,
                b_lp, b_ent, b_rew, player_counters,
                ent_coef=current_ent_coef, 
                king_available=(best_player_weights is not None)
            )           

            # 1. Prepare level distribution for logging
            # W&B can log dictionaries directly, making it easy to compare level behaviors
            level_dist_dict = {f"level_{i}": prob for i, prob in enumerate(current_level_probs)}

            # 2. Log to Weights & Biases
            wandb.log({
                "generation": generation,
                
                # Training Dynamics
                "train/loss": total_loss,
                "train/learning_rate": scheduler.get_last_lr()[0],
                "train/entropy_coefficient": current_ent_coef,
                
                # Game Performance
                "game/stalemate_rate": stalemate_rate,
                "game/mean_turns_taken": mean_turns,
                
                # Level Strategy
                **{f"level_probs/{k}": v for k, v in level_dist_dict.items()},
                
                # Optional: Log the relationship (Global exploration health)
                "meta/ent_lr_ratio": current_ent_coef / (scheduler.get_last_lr()[0] + 1e-8)
            })
            
            # early exit logic
            patience_criterion = stalemate_rate * mean_turns / 300.0
            if patience_criterion < (best_patience_criterion - min_delta):
                best_patience_criterion = patience_criterion
                patience_counter = 0
            else:
                patience_counter += 1
            

            # Player Evaluation
            if generation % 20 == 0 and generation > 0:

                num_games = 256
                loss_history = evaluate_players(players, num_games=num_games, sequence_length=sequence_length)
                
                # if the king player lost only their equal share of games they stay
                if loss_history[0] <= num_games / 3: 
                    if generation == 20:
                        best_player_weights = copy.deepcopy(players[0].state_dict())
                        save_best_player(players[0], filename='temporary_king.pth')
                # if no challenger proves significantly better, king stays
                elif loss_history[0] > num_games / 3 and not any(loss_history[0] - loss > 0.05 * num_games for loss in loss_history[1:]):
                    pass
                # if none of the above conditions are met, we have a new king
                else:
                    new_king_idx = np.argmin(loss_history)
                    best_player_weights = copy.deepcopy(players[new_king_idx].state_dict())
                    save_best_player(players[new_king_idx], filename='temporary_king.pth')

            if generation > 20 and patience_counter >= patience:
                wandb.log({"meta/early_stopped": True})
                break
            
            # step the scheduler
            scheduler.step()

    except Exception as e:
        print(f"An error occurred during sweep training: {e}")
        raise e
    
    finally:
        save_best_player(players[0], filename='temporary_king.pth')
        wandb.finish()
# endregion

# region WANDB Initialization
wandb.init(
    project="palace-ai",
    name="vectorized-training-run",
    config={
        "num_generations": 1000,
        "batch_size": 256,
        "initial_lr": 1e-4,
        "final_lr": 1e-6,
        "entropy_start": 0.05,
        "entropy_end": 0.005,
        "sequence_length": 6,
    }
)
# endregion

# region Training Parameters
# plotter = LivePlot()

# sweep config for weights & biases
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'train/loss',
        'goal': 'minimize'   
    },
    'parameters': {
        'initial_lr': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-3
        },
        'final_lr': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-5
        },
        'initial_ent': {
            'values': [0.01, 0.05, 0.1, 0.2]
        },
        'final_ent': {
            'values': [0.001, 0.005, 0.01, 0.02]
        }
    }
}

if __name__ == "__main__":
    os.environ["WANDB_START_METHOD"] = "thread"
    
    sweep_id = wandb.sweep(sweep_config, project="palace-ai-sweeps")
    num_agents = multiprocessing.cpu_count()
    print(f"Starting sweep with {num_agents} parallel agents...")
    processes = []
    for _ in range(num_agents):
        p = multiprocessing.Process(target=wandb.agent, args=(sweep_id,), kwargs={'function': sweep_train, 'count': 4})
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


# # training length
# num_generations = 1000
# batch_size = 256

# # learning rate
# initial_lr = 1e-4
# final_lr = 1e-6

# last_king_loss = 0.0

# # player parameters
# sequence_length = 6 # number of actions to track in history
# shared_player = PalacePlayer().to(device)
# players = [shared_player for _ in range(3)]

# # optimizer
# optimizer = torch.optim.Adam(shared_player.parameters(), lr=initial_lr)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=150, T_mult=2, eta_min=final_lr)

# # level tracking
# level_wins = torch.zeros(4, device = device)
# level_counts = torch.zeros(4, device = device)
# current_level_probs = [0.0, 0.0, 0.0, 1.0] # start all at level 3

# # misc. parameter initializations
# best_player_weights = None
# delta_smoothed_loss = None

# # endregion

# # profiler = cProfile.Profile()
# # profiler.enable()

# try:
    
#     for generation in range(num_generations):
#         start_time = time.perf_counter()

#         # Prepare players for this generation
#         b_lp, b_rew, b_ent, player_counters, stalemate_rate, level_wins, level_counts = training_step(players, best_player_weights, batch_size, sequence_length, level_wins, level_counts, current_level_probs)

#         # Update level distribution based on performance
#         current_level_probs = update_level_distribution(level_wins.cpu().numpy(), level_counts.cpu().numpy(), current_level_probs)

#         # adapt entropy coefficient according to learning rate schedule
#         current_ent_coef = get_dynamic_entropy(generation)

#         # Update using the accumulated batch data
#         print("Calculating and backpropagating loss...")
#         total_loss = finish_batch_update(
#             optimizer, 
#             b_lp, b_ent, b_rew, player_counters,
#             ent_coef=current_ent_coef, 
#             king_available=(best_player_weights is not None)
#         )           

#         # 1. Prepare level distribution for logging
#         # W&B can log dictionaries directly, making it easy to compare level behaviors
#         level_dist_dict = {f"level_{i}": prob for i, prob in enumerate(current_level_probs)}

#         # 2. Log to Weights & Biases
#         wandb.log({
#             "generation": generation,
            
#             # Training Dynamics
#             "train/loss": total_loss,
#             "train/learning_rate": scheduler.get_last_lr()[0],
#             "train/entropy_coefficient": current_ent_coef,
            
#             # Game Performance
#             "game/stalemate_rate": stalemate_rate,
#             "game/mean_turns_taken": player_counters.sum(dim=0).float().mean().item(),
            
#             # Level Strategy
#             **{f"level_probs/{k}": v for k, v in level_dist_dict.items()},
            
#             # Optional: Log the relationship (Global exploration health)
#             "meta/ent_lr_ratio": current_ent_coef / (scheduler.get_last_lr()[0] + 1e-8)
# })

#         print("\n" + "="*50)
#         print(f"Generation {generation} completed.")
#         print(f"Loss = {total_loss:.4f}, \nLR={scheduler.get_last_lr()[0]:.6f}, \nEntropy Coef={current_ent_coef:.4f}, \nStalemate Rate={stalemate_rate:.4f}")
#         print(f"Mean Turns Taken: {player_counters.sum(dim=0).float().mean().item():.2f}")
#         print(f"Level Distribution Probabilities: {[f'{prob:.2f}' for prob in current_level_probs]}")
#         print("="*50 + "\n")
        
#         avg_rewards = []
#         for p_idx in range(b_rew.shape[0]): # b_rew shape: (NumPlayers, BatchSize, MaxTurns)
#             total_rewards_per_game = b_rew[p_idx].sum(dim=1) 
#             avg_rewards.append(total_rewards_per_game.mean().item())

#         loss_val = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss

#         # delta_smoothed_loss, delta_smoothed_reward = plotter.update(
#         #     generation, 
#         #     avg_rewards, 
#         #     loss_val, 
#         #     lr = scheduler.get_last_lr()[0],
#         #     entropy = current_ent_coef,
#         #     stalemate_rate = stalemate_rate*100.0,
#         #     game_length = player_counters.sum(dim=0).float().mean().item()
#         # )

#         # region Player Evaluation
#         if generation % 20 == 0 and generation > 0:
            
#             print(f"Evaluating generation {generation} players against current best...")
            
#             num_games = 256
#             loss_history = evaluate_players(players, num_games=num_games, sequence_length=sequence_length)
            
#             # if the king player lost only their equal share of games they stay
#             if loss_history[0] <= num_games / 3: 
#                 print(f"King Player lost {loss_history[0]} games and retains the throne.")
#                 if generation == 20:
#                     best_player_weights = copy.deepcopy(players[0].state_dict())
#                     save_best_player(players[0], filename='temporary_king.pth')
#             # if no challenger proves significantly better, king stays
#             elif loss_history[0] > num_games / 3 and not any(loss_history[0] - loss > 0.05 * num_games for loss in loss_history[1:]):
#                 print(f"King Player lost {loss_history[0]} games but no challenger proved superior.")
#             # if none of the above conditions are met, we have a new king
#             else:
#                 new_king_idx = np.argmin(loss_history)
#                 print(f"A new king player has emerged (Player {new_king_idx}) with only {loss_history[new_king_idx]} losses!")
#                 best_player_weights = copy.deepcopy(players[new_king_idx].state_dict())
#                 save_best_player(players[new_king_idx], filename='temporary_king.pth')
            
#             print(f"Loss history: {loss_history}")
            
#             # endregion

#         end_time = time.perf_counter()
#         print(f"Generation {generation} took {end_time - start_time:.2f} seconds.\n")

#         # step the scheduler
#         scheduler.step()

#     save_best_player(players[0], filename='temporary_king.pth')

# except KeyboardInterrupt:
#     print("\n[INTERRUPT] Training paused by user. Saving current models...")
    
#     # Save the Learner (current weights being optimized)
#     torch.save(players[1].state_dict(), "learner_interrupted.pt")
    
#     # Save the King (the current best performer)
#     save_best_player(players[0], filename='temporary_king.pth')
#     print("Models saved successfully. Exiting.")

#     # plt.savefig('training_progress.png')

# # profiler.disable()
# # stats = pstats.Stats(profiler).sort_stats('cumtime')
# # stats.print_stats(20)

# # plt.savefig('training_progress.png')