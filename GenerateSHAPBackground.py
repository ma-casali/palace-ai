from GetPlayerStats import play_game
import numpy as np
import shap

# Run this once in your analysis environment
input_rec, action_rec, _, _ = play_game("Palace_king.pth")
print("Input record shape:", input_rec.shape)
print("Action record shape:", action_rec.shape)
num_turns = input_rec.shape[0]
action_dim, seq_len = 79, 12

# Flatten and combine
flattened_action_history = action_rec.reshape(num_turns, seq_len * action_dim)
full_states = np.hstack((flattened_action_history, input_rec))

# Sample 100 representative states and save
background_summary = shap.sample(full_states, 100)
np.save("shap_background.npy", background_summary)