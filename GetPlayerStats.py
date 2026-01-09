import torch
import numpy as np
import tqdm
import shap
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# Ensure these match your actual file names
from VectorizedCardGame import PalaceEnv 
from PalacePlayer import PalacePlayer

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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

    return torch.cat(parts, dim = 1).float()  # (B, 73)

def play_game(king_path, device="mps"):
    batch_size = 1
    env = PalaceEnv(batch_size=batch_size, num_players=3, device=device)
    
    king_model = PalacePlayer().to(device)
    king_model.load_state_dict(torch.load(king_path, map_location=device))
    king_model.eval()

    action_history = torch.zeros((batch_size, 6, 79), device=device)
    hidden_states = {
        p: (torch.zeros(2, batch_size, king_model.hidden_dim, device=device),
            torch.zeros(2, batch_size, king_model.hidden_dim, device=device))
        for p in range(3)
    }

    env.reset([king_model]*3, levels=[1.0, 0.0, 0.0, 0.0])

    # Use lists for efficient gathering
    input_record = []
    action_record = []
    action_idx_record = []
    num_turns = 0
    
    with torch.no_grad():
        while not env.done.all() and num_turns < 500:
            current_p = env.active_players[0].item()
            
            # get the current static observation
            static_obs = create_static_input_vector(env) # Shape: (1, 82)
            mask = env.get_valid_mask()
            
            # infer the action 
            h, c = hidden_states[current_p]
            probs, (h_new, c_new) = king_model(
                action_history, 
                static_obs, 
                mask.unsqueeze(0), 
                (h, c)
            )
            hidden_states[current_p] = (h_new, c_new)
            
            action = torch.argmax(probs[0]).item()

            # Record data only for player 0 that led to action
            if current_p == 0:
                action_idx_record.append(action)
                input_record.append(static_obs[0].cpu().numpy())
                action_record.append(action_history[0].cpu().numpy())

            # step the environment
            env.step(torch.tensor([action], device=device))
            
            # Update action history tensor
            new_act = torch.zeros((batch_size, 1, 79), device=device)
            new_act[0, 0, action] = 1.0
            action_history = torch.cat((action_history[:, 1:], new_act), dim=1)

            num_turns += 1

    # Convert lists to final structured arrays
    # np.stack adds a new dimension at the beginning (axis 0)
    final_input_record = np.stack(input_record) if input_record else np.array([])
    final_action_record = np.stack(action_record) if action_record else np.array([])

    return final_input_record, final_action_record, action_idx_record

def analyze_game_dominance(king_path, input_record, action_record):

    king_model = PalacePlayer()
    king_model.load_state_dict(torch.load(king_path, map_location="cpu"))
    king_model.eval()
    
    def model_predict(data):
        # expand flattened data back to original shape
        device = next(king_model.parameters()).device
        X = torch.tensor(data, dtype=torch.float32, device = device)
        batch_size = X.shape[0]
        
        # first 6 * 79 = 474 are action history
        action_dim = 79
        seq_len = 6
        action_history_flat = X[:, :seq_len * action_dim] # (B, 474)
        static_obs = X[:, seq_len * action_dim:] # (B, 82)
        action_history = action_history_flat.view(batch_size, seq_len, action_dim) # (B, 6, 79)

        # initialize hidden states to zero
        h0 = torch.zeros(king_model.num_rnn_layers, batch_size, king_model.hidden_dim, device=device)
        c0 = torch.zeros(king_model.num_rnn_layers, batch_size, king_model.hidden_dim, device=device)

        # the model will see all actions as valid for SHAP analysis
        dummy_mask = torch.ones((batch_size, 79), device=device)
        with torch.no_grad():
            logits, _ = king_model(action_history, static_obs, dummy_mask, (h0, c0))
            return logits.numpy()

    # Flatten input_record and action_record for SHAP
    num_turns = input_record.shape[0]
    action_dim = 79
    seq_len = 6
    flattened_action_history = action_record.reshape(num_turns, seq_len * action_dim)
    flattened_input = np.hstack((flattened_action_history, input_record))  # (num_turns, 474 + 82)

    # define a background to compare the model's predictions against
    unique_states = np.unique(flattened_input, axis=0)
    background_summary = shap.sample(unique_states, np.minimum(100, unique_states.shape[0]))
    explainer = shap.Explainer(model_predict, background_summary)
    
    # Analyze the chosen actions
    shap_values = explainer(flattened_input, max_evals = 2000, batch_size = 64)
    
    return shap_values

def plot_decision_component(shap_values, action_idx_record, action_record):

    num_turns = shap_values.values.shape[0]
    action_dim = 79
    seq_len = 6
    num_features = shap_values.values.shape[1]
    shap_impact_on_chosen = np.zeros((num_turns, num_features))
    for t in range(num_turns):
        action_idx = action_idx_record[t]
        shap_impact_on_chosen[t] = shap_values.values[t, :, action_idx] 
    static_impact = shap_impact_on_chosen[:, seq_len * action_dim:]  # (num_turns, 82)
    history_impact = shap_impact_on_chosen[:, :seq_len * action_dim]  # (num_turns, 474)

    impacts = {}
    
    # break down the impacts into categories

    hand_impact = static_impact[:, :13].sum(axis=1)  # (num_turns,)
    impacts["Player_Hand"] = hand_impact
    faceup_impact = static_impact[:, 13:26].sum(axis=1)  # (num_turns,)
    impacts["Player_Faceup"] = faceup_impact
    facedown_impact = static_impact[:, 26]  # (num_turns,)
    impacts["Player_Facedown"] = facedown_impact
    
    opponent1_hand_impact = static_impact[:, 27] # (num_turns,)
    impacts["Opponent1_Hand"] = opponent1_hand_impact
    opponent1_faceup_impact = static_impact[:, 28:41].sum(axis=1)  # (num_turns,)
    impacts["Opponent1_Faceup"] = opponent1_faceup_impact
    opponent1_facedown_impact = static_impact[:, 41]  # (num_turns,)
    impacts["Opponent1_Facedown"] = opponent1_facedown_impact
    
    opponent2_hand_impact = static_impact[:, 42]  # (num_turns,)
    impacts["Opponent2_Hand"] = opponent2_hand_impact
    opponent2_faceup_impact = static_impact[:, 43:56].sum(axis=1)  # (num_turns,)
    impacts["Opponent2_Faceup"] = opponent2_faceup_impact
    opponent2_facedown_impact = static_impact[:, 56]  # (num_turns,)
    impacts["Opponent2_Facedown"] = opponent2_facedown_impact

    discard_impact = static_impact[:, 57:70].sum(axis=1)  # (num_turns,)
    impacts["Discard_Cards"] = discard_impact  # (num_turns,)
    top_card_impact = static_impact[:, 70]  # (num_turns,)
    impacts["Top_Card"] = top_card_impact
    run_count_impact = static_impact[:, 71]  # (num_turns,)
    impacts["Run_Count"] = run_count_impact
    drawpile_impact = static_impact[:, 72]  # (num_turns,)
    impacts["Drawpile_Length"] = drawpile_impact

    history_impact = history_impact.reshape(num_turns, seq_len, action_dim)  # (num_turns, 6, 79)
    history_impact_sum = history_impact.sum(axis=2)  # (num_turns, 6)
    for t in range(seq_len):
        impacts[f"History_T-{seq_len - t}"] = history_impact_sum[:, t]  # (num_turns,)

    # Labels for action indices
    labels = []
    card_names = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
    for action_idx in range(79):
        if action_idx == 78:
            labels.append("Pickup")
        else:
            card_rank = action_idx % 13
            action_category = action_idx // 13
            if action_category == 0:
                labels.append(f"Played a {card_names[card_rank]}")
            elif action_category > 0 and action_category < 4:
                labels.append(f"Played {action_category + 1} {card_names[card_rank]}s")
            elif action_category == 4:
                labels.append(f"Played F-U {card_names[card_rank]}")
            else:
                labels.append(f"Played F-D")

    # colors to use
    color_hex = [
        # Player hands
        "#00ffff",
        "#008b8b",
        "#265d5f",

        # Opponent 1 hands
        "#ffff00",
        "#b4a21a",
        "#9f8d24",

        # Opponent 2 hands
        "#ff00ff",
        "#af2eaf",
        "#8e268b",

        # Table cards
        "#ff0000",
        "#b22222",
        "#bb7c76",
        "#6A3928",

        # Action history
        "#dcdcdc",
        "#d3d3d3",
        "#c0c0c0",
        "#a9a9a9",
        "#808080",
        "#696969"
    ]

    # Plotting
    fig, [ax, ax2] = plt.subplots(2, 1, figsize=(12, 10), layout = "constrained")
    turn_indices = np.arange(num_turns)
    pos_bottom = np.zeros(num_turns)
    neg_bottom = np.zeros(num_turns)
    net_impact = np.zeros(num_turns)
    turn = 0

    normalized_impacts = {}
    total_abs_impact = np.zeros(num_turns)
    for values in impacts.values():
        total_abs_impact += np.abs(values)
    total_abs_impact[total_abs_impact == 0] = 1e-9  # prevent division by zero

    bottom = 0
    for key, values in impacts.items():
        net_impact += values 
        normalized_impacts[key] = np.abs(values) / total_abs_impact

        ax.bar(turn_indices*3, normalized_impacts[key], bottom=bottom, label=key, color=color_hex[turn], width = 2.8)

        bottom += normalized_impacts[key]
        
        turn += 1

    for turn in range(num_turns):
        # Place text above the highest positive bar
        y_pos = -0.05
        ax.text(turn*3, 0, "|", ha='center', va='center', fontsize=12, color='black')
        ax.text(turn*3, y_pos, labels[action_idx_record[turn]], ha='center', va='top', fontsize=8, rotation=90, fontweight='bold')
        ax.text(turn*3 + 1, y_pos, "Opp. 1 " + labels[np.argmax(action_record[np.minimum(turn+1, num_turns - 1), -2])], ha='center', va='top', fontsize=8, rotation=90, color = "#6a5418")
        ax.text(turn*3 + 2, y_pos, "Opp. 2 " + labels[np.argmax(action_record[np.minimum(turn+1, num_turns - 1), -1])], ha='center', va='top', fontsize=8, rotation=90, color = "#540d54")

    ax.set_ylim([-0.5, 1.02])
    ax.set_xlabel("Turn")
    ax.set_ylabel("SHAP Value Impact")
    ax.set_title("SHAP Value Decomposition of Player's Decisions Over Game Turns")
    handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(handles, legend_labels, loc = 'upper left', bbox_to_anchor=(1.05, 1))

    ax2.plot(turn_indices*3, net_impact, color='black', marker = 'o', linewidth=2, label='Net SHAP Impact')
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.grid(True)
    ax2.set_xlabel("Turn")
    ax2.set_ylabel("Net SHAP Impact")

    plt.show()


king_path = "Palace_king.pth"
print("Playing game...")
input_record, action_record, action_idx_record = play_game(king_path, device = device)

print("Analyzing game performance...")
shap_results = analyze_game_dominance(king_path, input_record, action_record)

print("Plotting decision components...")
plot_decision_component(shap_results, action_idx_record, action_record)