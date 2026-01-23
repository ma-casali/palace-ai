import torch
import numpy as np
import tqdm
import shap
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
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
    drawpile_len = 52 - env.drawpile_ptrs.float().unsqueeze(1)  # (B, 1)
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
    acting_player = []
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
            action_idx_record.append(action)
            input_record.append(static_obs[0].cpu().numpy())
            action_record.append(action_history[0].cpu().numpy())
            acting_player.append(current_p)

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
    final_acting_player = np.array(acting_player) if acting_player else np.array([])

    return final_input_record, final_action_record, final_acting_player, action_idx_record

def analyze_game_dominance(king_path, input_record, action_record, acting_player):

    king_model = PalacePlayer()
    king_model.load_state_dict(torch.load(king_path, map_location="cpu"))
    king_model.eval()

    player0_mask = (acting_player == 0)
    input_record = input_record[player0_mask]
    action_record = action_record[player0_mask]
    
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

def plot_decision_component(shap_values, action_idx_record, action_record, acting_player):

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
        impacts[f"History: Turn -{seq_len - t}"] = history_impact_sum[:, t]  # (num_turns,)

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
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), layout = "constrained")
    net_impact = np.zeros(num_turns)

    normalized_impacts = {}
    total_abs_impact = np.zeros(num_turns)
    for values in impacts.values():
        total_abs_impact += np.abs(values)
    total_abs_impact[total_abs_impact == 0] = 1e-9  # prevent division by zero

    turn_idx = 0
    bottom = 0
    player0_turns = np.where(acting_player == 0)[0]
    all_turns = np.concatenate((np.array([-1]), player0_turns, np.array([len(acting_player)]) if num_turns > player0_turns[-1] else np.array([len(acting_player) + 1])))
    dist_left = np.diff(all_turns[:-1])
    dist_right = np.diff(all_turns[1:])
    widths = np.minimum(dist_left, dist_right) * 0.8
    
    for key, values in impacts.items():
        net_impact += values 
        normalized_impacts[key] = np.abs(values) / total_abs_impact
        ax.bar(player0_turns, normalized_impacts[key], bottom=bottom, label=key, color=color_hex[turn_idx], width = widths)
        bottom += normalized_impacts[key]
        turn_idx += 1

    y_pos = -0.05
    player0_turn = 1
    for turn in range(acting_player.shape[0]):
        if acting_player[turn] == 0:
            ax.text(turn, 0, "|", ha='center', va='center', fontsize=12, color='black')
            ax.text(turn, y_pos, f"{player0_turn}. {labels[action_idx_record[turn]]}", ha='center', va='top', fontsize=8, rotation=90, fontweight='bold')
            player0_turn += 1
        elif acting_player[turn] == 1:
            ax.text(turn, y_pos, "Opp. 1 " + labels[action_idx_record[turn]], ha='center', va='top', fontsize=8, rotation=90, color = "#6a5418")
        elif acting_player[turn] == 2:
            ax.text(turn, y_pos, "Opp. 2 " + labels[action_idx_record[turn]], ha='center', va='top', fontsize=8, rotation=90, color = "#540d54")

    ax.set_ylim([-0.5, 1.02])
    ax.set_xlabel("Turn")
    ax.set_ylabel("SHAP Value Impact")
    ax.set_title("Portion of Impact on Chosen Action by Decision Component")
    ax.set_xlim([-1.5, num_turns*3])
    ax.axis('off')

    flat_labels = [
        "Player Hand", "Player Faceup", "Player Facedown", "", "", "",
        "Opp. 1 Hand", "Opp. 1 Faceup", "Opp. 1 Facedown", "", "", "",
        "Opp. 2 Hand", "Opp. 2 Faceup", "Opp. 2 Facedown", "", "", "",
        "Discard Cards", "Top Card", "Run Count", "Drawpile Length", "", "",
        "History Turn -6", "History Turn -5", "History Turn -4", "History Turn -3", "History Turn -2", "History Turn -1"
    ]

    all_handles_in_columns = [[] for _ in range(5)]
    color_idx = 0
    col_idx = 0
    for item in flat_labels:

        this_hex = color_hex[color_idx] if item != "" else "#ffffff"
        patch = Patch(color=this_hex, label=item, visible=(item != ""))
        color_idx = color_idx + 1 if item != "" else color_idx
        
        all_handles_in_columns[col_idx].append(patch)
        col_idx = (col_idx + 1) % 5


    row_ordered_handles = []
    for row in zip(*all_handles_in_columns):
        row_ordered_handles.extend(row)

    ax.legend(
        handles=row_ordered_handles,
        ncol=5,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        columnspacing=1.0
    )

    plt.show()

def save_explanation_file(shap_values, input_record, action_idx_record, acting_player, filename):

    # mask input_record to only player 0 turns
    player0_mask = (acting_player == 0)
    input_record = input_record[player0_mask]
    action_idx_record = np.array(action_idx_record)[player0_mask]

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
    
    document_string = ""
    num_turns = shap_values.values.shape[0]
    action_dim = 79
    seq_len = 6
    for t in range(num_turns):
        action_idx = np.argmax(shap_values.values[t].sum(axis=0))
        document_string += f"{t + 1}. {labels[action_idx_record[t]]} because: "
        
        # get top three feature impacts on chosen action
        feature_impacts = shap_values.values[t, :, action_idx]  # (num_features,)
        top_feature_indices = np.argsort(-np.abs(feature_impacts))[:3]
        for reason_idx, feature_idx in enumerate(top_feature_indices):
            if reason_idx == 0:
                prefix, suffix = "", ", "
            elif reason_idx == 1:
                prefix, suffix = "", ", and "
            else:
                prefix, suffix = "", ".\n"
            if feature_idx >= seq_len * action_dim: # static features
                idx = np.int32(feature_idx - seq_len * action_dim)
                if idx < 13:
                    mult = np.int32(input_record[t, idx])  # number of that card in hand
                    document_string += f"{prefix}player had {mult} {card_names[idx]}{suffix}"
                elif idx >= 13 and idx < 26:
                    mult = np.int32(input_record[t, idx])
                    document_string += f"{prefix}player had {mult} faceup {card_names[idx - 13]}{suffix}"
                elif idx == 26:
                    document_string += f"{prefix}player had {np.int32(input_record[t, idx])} facedown cards{suffix}"
                elif idx == 27:
                    document_string += f"{prefix}opponent 1 had {np.int32(input_record[t, idx])} cards in their hand{suffix}"
                elif idx >= 28 and idx < 41:
                    mult = np.int32(input_record[t, idx])
                    document_string += f"{prefix}opponent 1 had {mult} faceup {card_names[idx - 28]}{suffix}"
                elif idx == 41:
                    document_string += f"{prefix}opponent 1 had {np.int32(input_record[t, idx])} facedown cards{suffix}"
                elif idx == 42:
                    document_string += f"{prefix}opponent 2 had {np.int32(input_record[t, idx])} cards in their hand{suffix}"
                elif idx >= 43 and idx < 56:
                    mult = np.int32(input_record[t, idx])
                    document_string += f"{prefix}opponent 2 had {mult} faceup {card_names[idx - 43]}{suffix}"
                elif idx == 56:
                    document_string += f"{prefix}opponent 2 had {np.int32(input_record[t, idx])} facedown cards{suffix}"
                elif idx >= 57 and idx < 70:
                    # say which card in discard pile
                    mult = np.int32(input_record[t, idx])
                    document_string += f"{prefix}there was {mult} {card_names[idx - 57]} in the discard pile{suffix}"
                elif idx == 70:
                    document_string += f"{prefix}the top card on the pile was a {card_names[np.int32(input_record[t, idx])]}{suffix}"
                elif idx == 71:
                    document_string += f"{prefix}the run count was {np.int32(input_record[t, idx])}{suffix}"
                elif idx == 72:
                    document_string += f"{prefix}the drawpile had {np.int32(input_record[t, idx])} cards left{suffix}"
            else:
                history_turn = (feature_idx // action_dim) + 1
                action_idx = (feature_idx - 73) % action_dim
                document_string += f"{prefix}on turn -{seq_len - history_turn}, someone {labels[action_idx]}{suffix}"
            
    with open(filename, "w") as f:
        f.write(document_string)

if __name__ == "__main__":
    king_path = "Palace_king.pth"
    print("Playing game...")
    input_record, action_record, acting_player, action_idx_record = play_game(king_path, device = device)

    print("Analyzing game performance...")
    shap_results = analyze_game_dominance(king_path, input_record, action_record, acting_player)

    print("Saving explanation file...")
    save_explanation_file(shap_results, input_record, action_idx_record, acting_player, "game_explanation.txt")

    print("Plotting decision components...")
    plot_decision_component(shap_results, action_idx_record, action_record, acting_player)