from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import SymLogNorm
from CardGame import get_valid_mask, PalaceEnv, PalacePlayer
import torch
import sys
import tqdm

import shap
import lime
import lime.lime_tabular

""" The following script is meant to illuminate how the player is making its decisions."""

np.set_printoptions(threshold=sys.maxsize)

def create_input_vector(last_actions, current_hand, valid_mask):
    last_actions_flat = np.array(last_actions).flatten()  # Shape: (39,)
    current_hand_flat = np.array(current_hand).flatten()  # Shape: (13,)
    valid_mask_flat = np.array(valid_mask).flatten()      # Shape: (79,)

    input_vector = np.concatenate([last_actions_flat, current_hand_flat, valid_mask_flat])
    return input_vector  # Shape: (131,)

def play_game(king_path):

    # hand history shape: (num_turns+1, 13)
    # action history shape: (num_turns, )

    env = PalaceEnv(num_players = 3)

    # Load in the kings
    king_model = PalacePlayer()
    king_model.load_state_dict(torch.load(king_path, map_location=torch.device('cpu')))
    king_model.eval() 

    players = [king_model, king_model, king_model]
    action_history = [np.zeros(13) for _ in range(3)]
    env.reset(players)

    players_out = [] # track players who have finished their game
    done = False
    current_player_idx = 0

    full_hand_history = np.zeros((1, 13))
    full_action_history = np.zeros((1, 2), dtype=int)  # (player_idx, action_idx)
    input_history = []
    num_turns = 0
    max_turns = 1000
    end_of_deck = None

    with torch.inference_mode():
        while not done and num_turns < max_turns:
            if current_player_idx in players_out:
                current_player_idx = (current_player_idx + 1) % env.num_players
                continue

            mask = get_valid_mask(env.hands[current_player_idx],
                                    env.discard_pile,
                                    env.face_up_piles[current_player_idx],
                                    env.face_down_piles[current_player_idx])
            input_vec = torch.tensor(create_input_vector(action_history, env.hands[current_player_idx], mask), dtype=torch.float32)
            mask_vec = torch.tensor(mask.flatten(), dtype=torch.bool)

            probs = king_model(input_vec, mask_vec)
            action_idx = torch.argmax(probs).item()

            full_hand_history = np.concatenate([full_hand_history, [env.hands[current_player_idx]]])
            full_action_history = np.concatenate([full_action_history, [[current_player_idx, action_idx]]])
            input_history.append(input_vec.numpy())
            if len(env.draw_pile.deck) == 0 and end_of_deck is None:
                end_of_deck = num_turns

            # Execute step
            done, action_history, current_player_idx, players_out, _ = env.step(action_idx, action_history, current_player_idx, players_out)
            num_turns += 1

    return input_history, full_hand_history, full_action_history, end_of_deck

def analyze_wild_card_lifetimes(king_path):

    num_games = 100
    life_times_dict = {f'{rank}': [] for rank in [0, 1, 5, 8]}
    picked_up_times_dict = {f'{rank}': [] for rank in [0, 1, 5, 8]}
    pbar = tqdm.tqdm(range(num_games), desc="Analyzing Wild Card Lifetimes")
    for game in pbar:
        # play a game and collect data
        _, hand_history, action_history, end_of_deck = play_game(king_path="Palace_king.pth")

        # Make dict for wild-cards in and out of hand
        wild_card_ranks = [0, 1, 5, 8]
        dict_wild_card_events = {f'{rank}': {'picked_up': [], 'played': []} for rank in wild_card_ranks}

        # Determine when each wild card has been picked-up
        turn_number_pickup, card_rank_pickup = np.where(np.diff(hand_history, axis = 0) > 0)

        # Determine when each wild card has been played
        turn_number_played, card_rank_played = np.where(np.diff(hand_history, axis = 0) < 0)
        
        for rank in wild_card_ranks:
            picked_up_times = turn_number_pickup[card_rank_pickup == rank]
            played_times = turn_number_played[card_rank_played == rank]

            for pick_time in picked_up_times:
                # find the first played time after pick_time
                played_after_pick = played_times[played_times > pick_time]
                if len(played_after_pick) > 0:
                    lifetime = played_after_pick[0] - pick_time
                    life_times_dict[str(rank)].append(lifetime)
                    picked_up_times_dict[str(rank)].append(pick_time/end_of_deck if end_of_deck is not None else 0)

    fig, ax = plt.subplots(2,2, figsize=(12,8))
    ax[0,0].set_title('2 Lifetimes')
    ax[0,0].hist2d(picked_up_times_dict['0'], life_times_dict['0'], bins=20, cmap='Blues', norm = LogNorm())
    ax[0,0].set_xlabel('Picked-up Time (turns/end of deck)')
    ax[0,0].set_ylabel('Lifetime (turns)')
    cb = plt.colorbar(ax[0,0].collections[0], ax=ax[0,0])
    cb.set_label('Number of Occurrences (Log Scale)')

    ax[0,1].set_title('3 Lifetimes')
    ax[0,1].hist2d(picked_up_times_dict['1'], life_times_dict['1'], bins=20, cmap='Blues', norm = LogNorm())
    ax[0,1].set_xlabel('Picked-up Time (turns/end of deck)')
    ax[0,1].set_ylabel('Lifetime (turns)')
    cb = plt.colorbar(ax[0,1].collections[0], ax=ax[0,1])
    cb.set_label('Number of Occurrences (Log Scale)')

    ax[1,0].set_title('7 Lifetimes')
    ax[1,0].hist2d(picked_up_times_dict['5'], life_times_dict['5'], bins=20, cmap='Blues', norm = LogNorm())
    ax[1,0].set_xlabel('Picked-up Time (turns/end of deck)')
    ax[1,0].set_ylabel('Lifetime (turns)')
    cb = plt.colorbar(ax[1, 0].collections[0], ax=ax[1, 0])
    cb.set_label('Number of Occurrences (Log Scale)')

    ax[1,1].set_title('10 Lifetimes')
    ax[1,1].hist2d(picked_up_times_dict['8'], life_times_dict['8'], bins=20, cmap='Blues', norm = LogNorm())
    ax[1,1].set_xlabel('Picked-up Time (turns/end of deck)')
    ax[1,1].set_ylabel('Lifetime (turns)')
    cb = plt.colorbar(ax[1, 1].collections[0], ax=ax[1, 1])
    cb.set_label('Number of Occurrences (Log Scale)')

    plt.show()

import numpy as np
import shap

def analyze_game_dominance(king_model, input_history, action_history, N=3):
    """
    N: The number of 'runner-up' features to compare against.
    """
    # 1. Wrapper as defined previously
    def model_predict(data):
        input_tensor = torch.tensor(data, dtype=torch.float32)
        mask_slice = input_tensor[:, 52:] > 0.5 
        with torch.no_grad():
            return king_model(input_tensor, mask_slice).numpy()

    # Use a generic background (e.g., first 100 turns or a subset)
    background = np.array(input_history[:min(100, len(input_history))])
    explainer = shap.Explainer(model_predict, background)
    
    game_insights = []

    # Skip turn 0 initialization if necessary (match your history offset)
    pbar = tqdm.tqdm(range(len(input_history)), desc="Analyzing Game Dominance")
    for turn in pbar:
        sample = np.array([input_history[turn]])
        actual_action = action_history[turn, 1] 
        
        # Get SHAP values for this turn's chosen action
        # shap_values.values is shape (1, 131, 79)
        shap_values = explainer(sample)
        
        # Extract impacts for the specific action chosen
        impacts = np.abs(shap_values.values[0, :, actual_action])
        
        # Sort indices by impact descending
        sorted_indices = np.argsort(impacts)[::-1]
        
        top_idx = sorted_indices[0]
        top_val = impacts[top_idx]
        
        # Calculate mean of next N features
        runner_up_indices = sorted_indices[1:1+N]
        runner_up_mean = np.mean(impacts[runner_up_indices]) + 1e-9 # avoid div by zero
        
        dominance_scalar = top_val / runner_up_mean
        
        game_insights.append({
            'turn': turn,
            'feature_impacts': impacts,
            'action': actual_action
        })
        
    return game_insights


king_path = "Palace_king.pth"
king_model = PalacePlayer()
king_model.load_state_dict(torch.load(king_path, map_location=torch.device('cpu')))
king_model.eval()

full_input_history = []
full_action_history = []
num_games = 1
pbar = tqdm.tqdm(range(num_games), desc="Playing Games for Dominance Analysis")
for _ in pbar:
    input_history, hand_history, action_history, end_of_deck = play_game(king_path)
    full_input_history.extend(input_history)
    full_action_history.extend(action_history[1:])  # skip initial zero action

full_input_history = np.array(full_input_history)
full_action_history = np.array(full_action_history)

game_insights = analyze_game_dominance(king_model, full_input_history, full_action_history)

# 1. Convert list of impacts to a 2D Matrix (Features x Turns)
impact_matrix = np.array([insight['feature_impacts'] for insight in game_insights]).T # Shape: (131, num_turns)

print("Impact matrix shape:", impact_matrix.shape)
print("Sample impact values (Feature 0 over first 10 turns):", impact_matrix[0, :10])

num_turns = impact_matrix.shape[1]

fig, ax = plt.subplots(figsize=(14, 8))

# 2. Plot using pcolormesh
# We use SymLogNorm because SHAP values can be positive or negative 
# and often vary by orders of magnitude.
im = ax.pcolormesh(np.arange(num_turns), np.arange(131), impact_matrix, 
                    cmap='RdBu',
                    shading='auto')

# 3. Add Category Delineation Lines
boundaries = [38.5, 51.5]
for b in boundaries:
    ax.axhline(y=b, color='black', lw=2, alpha=0.7)

# 4. Right-side Category Labels
ax_right = ax.twinx()
ax_right.set_ylim(0, 130)
ax_right.set_yticks([19, 45, 91])
ax_right.set_yticklabels(["HISTORY", "HAND", "MASK"], fontweight='bold')

# 5. Formatting
ax.set_xlabel("Turn Number")
ax.set_ylabel("Feature Index")
ax.set_title("Feature Impact Evolution (SHAP Values) Over Time")

# # Colorbar logic
# cb = fig.colorbar(im, ax=ax, pad=0.1)
# cb.set_label('Impact on Chosen Action (Positive/Negative)')

plt.tight_layout()
plt.show()