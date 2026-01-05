import numpy as np
import matplotlib.pyplot as plt
from CardGame import get_valid_mask, PalaceEnv, PalacePlayer
import torch
import sys

""" The following script is meant to illuminate how the player is making its decisions."""

np.set_printoptions(threshold=sys.maxsize)

def create_input_vector(last_actions, current_hand, valid_mask):
    last_actions_flat = np.array(last_actions).flatten()  # Shape: (39,)
    current_hand_flat = np.array(current_hand).flatten()  # Shape: (13,)
    valid_mask_flat = np.array(valid_mask).flatten()      # Shape: (79,)

    input_vector = np.concatenate([last_actions_flat, current_hand_flat, valid_mask_flat])
    return input_vector  # Shape: (131,)

def play_game(king_path):

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
    full_action_history = np.array([])
    num_turns = 0
    max_turns = 1000

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
            full_action_history = np.append(full_action_history, {f"{current_player_idx}": action_idx})

            # Execute step
            done, action_history, current_player_idx, players_out, _ = env.step(action_idx, action_history, current_player_idx, players_out)
            num_turns += 1

    return full_hand_history[1:,:], full_action_history

# print(play_game(king_path="Palace_king.pth"))
full_hand_history, _ = play_game(king_path="Palace_king.pth")
print(full_hand_history)
# Play a couple of games, determine when a player makes the next player pick up
