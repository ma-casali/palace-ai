import torch
import numpy as np
from VectorizedCardGame import PalaceEnv
from PalacePlayer import PalacePlayer
from colorama import Fore, Style, init
import tqdm

# Initialize colorama
init(autoreset=True)

def create_static_input_vector(env):
    batch_size = env.batch_size
    batch_indices = torch.arange(batch_size, device=env.device)
    
    # Get the ID of the current player for every env in the batch
    p = env.active_players # Shape (batch_size,)
    
    # Calculate opponent indices for the whole batch
    op1 = (p + 1) % 3
    op2 = (p + 2) % 3

    # 1. Self States
    # We use batch_indices and p to pick the correct hand for each env
    self_hand = env.hands[batch_indices, p]                # (batch_size, 13)
    self_faceup = env.face_up_piles[batch_indices, p]      # (batch_size, 13)
    self_facedown = env.face_down_piles[batch_indices, p].sum(dim=1, keepdim=True) # (batch_size, 1)

    # 2. Opponent 1
    opp1_hand_total = env.hands[batch_indices, op1].sum(dim=1, keepdim=True)       # (batch_size, 1)
    opp1_faceup = env.face_up_piles[batch_indices, op1]                           # (batch_size, 13)
    opp1_facedown = env.face_down_piles[batch_indices, op1].sum(dim=1, keepdim=True) # (batch_size, 1)

    # 3. Opponent 2
    opp2_hand_total = env.hands[batch_indices, op2].sum(dim=1, keepdim=True)       # (batch_size, 1)
    opp2_faceup = env.face_up_piles[batch_indices, op2]                           # (batch_size, 13)
    opp2_facedown = env.face_down_piles[batch_indices, op2].sum(dim=1, keepdim=True) # (batch_size, 1)

    # 4. Table State
    table_discard = env.discard_counts.float()                                     # (batch_size, 13)
    table_top = env.top_cards.view(-1, 1).float()                                  # (batch_size, 1)
    table_run = env.run_count.view(-1, 1).float()                                  # (batch_size, 1)
    table_draw = 52 - env.drawpile_ptrs.view(-1, 1).float()                        # (batch_size, 1)

    static_obs = torch.cat([
        self_hand, self_faceup, self_facedown,         # 27
        opp1_hand_total, opp1_faceup, opp1_facedown,   # 15
        opp2_hand_total, opp2_faceup, opp2_facedown,   # 15
        table_discard, table_top, table_run, table_draw # 16
    ], dim=1)

    return static_obs # Final shape: (batch_size, 73)

class MultiGameLogger:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        # Create 256 individual history lists
        self.histories = [[] for _ in range(batch_size)]

    def log_step(self, states, actions, rewards, active_players, dones):
        # states: (batch_size, 73)
        # actions: (batch_size,)
        # rewards: (batch_size, 3)
        # active_players: (batch_size,)
        # dones: (batch_size,)
        
        for i in range(self.batch_size):
            # Only continue logging if this specific game isn't finished
            if not dones[i]:
                p_idx = active_players[i].item()
                self.histories[i].append({
                    'state': states[i].detach().cpu().numpy(),
                    'action': actions[i].item(),
                    'reward': rewards[i, p_idx].item(),
                })

    def save_stalemate_log(self, dones, filename):
        # Find indices where done is False (stalemates)
        stalemate_indices = torch.where(~dones)[0]
        
        if len(stalemate_indices) == 0:
            print("No stalemates found. Saving first completed game instead.")
            target_idx = 0
        else:
            target_idx = stalemate_indices[0].item()
            print(f"Logging stalemate from env index: {target_idx}")

        selected_history = self.histories[target_idx]
        
        with open(filename, "w") as f:
            f.write(f"--- LOG FOR ENV {target_idx} (STALEMATE: {not dones[target_idx].item()}) ---\n")
            for i, step in enumerate(selected_history):
                f.write(f"Step {i+1}:\n")
                f.write(self.format_state_table(torch.tensor(step['state'])))
                f.write(f"Action Taken: {step['action']} | Reward: {step['reward']:.4f}\n")

    def format_state_table(self, static_obs):
        # ... (Keep your existing format_state_table logic here) ...
        obs = static_obs.squeeze().cpu().numpy()
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        
        def get_rank_string(data_slice):
            active = []
            for i, count in enumerate(data_slice):
                if count > 0: active.append(f"{ranks[i]}:{int(count)}")
            return ", ".join(active) if active else "Empty"

        lines = ["\n" + "="*70]
        lines.append(f"{'CATEGORY':<15} | {'METRIC':<20} | {'DATA / COUNTS'}")
        lines.append("-" * 70)
        lines.append(f"{'SELF':<15} | {'Hand':<20} | {get_rank_string(obs[0:13])}")
        lines.append(f"{'':<15} | {'Face-Up':<20} | {get_rank_string(obs[13:26])}")
        lines.append(f"{'':<15} | {'Face-Down Count':<20} | {int(obs[26])}")
        lines.append("-" * 70)
        lines.append(f"{'OPP 1':<15} | {'Hand Total':<20} | {int(obs[27])} cards")
        lines.append(f"{'':<15} | {'Face-Up':<20} | {get_rank_string(obs[28:41])}")
        lines.append(f"{'':<15} | {'Face-Down Count':<20} | {int(obs[41])}")
        lines.append("-" * 70)
        lines.append(f"{'OPP 2':<15} | {'Hand Total':<20} | {int(obs[42])} cards")
        lines.append(f"{'':<15} | {'Face-Up':<20} | {get_rank_string(obs[43:56])}")
        lines.append(f"{'':<15} | {'Face-Down Count':<20} | {int(obs[56])}")
        lines.append("-" * 70)
        top_card_idx = int(obs[70])
        top_card_val = ranks[top_card_idx] if top_card_idx != -1 else "None"
        lines.append(f"{'TABLE':<15} | {'Discard Pile':<20} | {get_rank_string(obs[57:70])}")
        lines.append(f"{'':<15} | {'Top Card':<20} | {top_card_val}")
        lines.append(f"{'':<15} | {'Run Count':<20} | {int(obs[71])}")
        lines.append(f"{'':<15} | {'Draw Pile Left':<20} | {int(obs[72])}")
        lines.append("="*70 + "\n")
        return "\n".join(lines)

def play_kings_against_kings(king_path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    batch_size = 256
    max_steps = 300 # Define your stalemate cutoff
    
    env = PalaceEnv(batch_size=batch_size, num_players=3, device=device)
    king_model = PalacePlayer().to(device)
    king_model.load_state_dict(torch.load(king_path, map_location=device))
    king_model.eval()

    # Shared logger for the batch
    logger = MultiGameLogger(batch_size)

    # Initialize LSTM states and History
    action_history = torch.zeros((batch_size, 6, 79), device=device)
    hidden_states = {
        p: (torch.zeros(2, batch_size, king_model.hidden_dim, device=device),
            torch.zeros(2, batch_size, king_model.hidden_dim, device=device))
        for p in range(3)
    }

    players = [king_model for _ in range(3)]
    env.reset(players)

    pbar_batch = tqdm.tqdm(total=max_steps, desc="Turns taken", leave=False)

    with torch.no_grad():
        for step_idx in range(max_steps):
            if env.done.all():
                break

            # In Palace, active_players varies per env in the batch
            # We must get the hidden state corresponding to the player whose turn it is in EACH env
            current_players = env.active_players # Shape (batch_size,)
            
            # Since LSTM states are stored per player, we perform inference
            # Note: For efficiency, in a true vectorized loop, you'd group envs by current_player
            # But here we'll assume the model can handle the batch.
            
            masks = env.get_valid_mask()
            static_obs = create_static_input_vector(env) # Ensure this handles full batch

            # simplified: getting hidden states for the batch is tricky if current_player differs
            # Assuming for this log we just use the first player's hidden state context 
            # or you've unified the LSTM to handle the batch
            h, c = hidden_states[0] # Example: needs to be mapped per active_player
            
            probs, (h_new, c_new) = king_model(
                action_history,
                static_obs,
                masks.unsqueeze(0) if masks.dim()==1 else masks,
                hidden_state=(h, c)
            )
            hidden_states[0] = (h_new, c_new)

            actions = torch.argmax(probs, dim=1) # (batch_size,)
            
            # Step environments
            rewards, dones = env.step(actions)

            # Update Action History
            new_action_onehot = torch.zeros((batch_size, 1, 79), device=device)
            new_action_onehot[torch.arange(batch_size), 0, actions] = 1.0
            action_history = torch.cat((action_history[:, 1:, :], new_action_onehot), dim=1)

            # Log everything
            logger.log_step(static_obs, actions, rewards, current_players, env.done)

            pbar_batch.set_postfix({"Active Games": int((~env.done).sum().item())})
            pbar_batch.update(1)
    
    pbar_batch.close()

    # Finally, save a log from a game that didn't finish
    logger.save_stalemate_log(env.done, "stalemate_analysis.txt")

def play_against_kings(king_path):
    # ... [Keep environment and model setup as you have it] ...
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
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

    players = [king_model for _ in range(3)]
    env.reset(players, levels=[1.0, 0.0, 0.0, 0.0])

    rank_names = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    category_names = ["Single", "Pair", "Three", "Four", "Face-Up", "Face-Down"]

    print(f"\n{Fore.GREEN}{'='*50}\n   WELCOME TO PALACE: HUMAN VS KINGS\n{'='*50}{Style.RESET_ALL}")

    with torch.no_grad():
        while not env.done.all():
            current_player = env.active_players[0].item()

            if current_player == 0:
                print(f"\n{Fore.WHITE}{Style.BRIGHT}--- BOARD STATE ---{Style.RESET_ALL}")
                
                # Table Info
                top_card_idx = env.top_cards[0].item()
                top_card_name = rank_names[top_card_idx % 13] if top_card_idx >= 0 else "Empty"
                pile_count = int(env.discard_counts[0].sum().item())
                draw_count = int(52 - env.drawpile_ptrs[0].item())
                
                print(f"Draw Pile: [{Fore.YELLOW}{draw_count}{Style.RESET_ALL}] cards left")
                print(f"Current Pile: {Fore.CYAN}{top_card_name}{Style.RESET_ALL} (x{env.run_count[0].item()}) | Pile size: {pile_count}")
                
                # Opponent Status (Structured)
                print(f"\n{Fore.WHITE}OPPONENTS:{Style.RESET_ALL}")
                for opp in [1, 2]:
                    h_count = int(env.hands[0, opp].sum())
                    fu_cards = [rank_names[i] for i in range(13) if env.face_up_piles[0, opp, i] > 0]
                    fd_count = int(env.face_down_piles[0, opp].sum())
                    print(f" P{opp}: Hand[{h_count}] | Face-Up{fu_cards} | Face-Down[{fd_count}]")

                # Your Hand
                print(f"\n{Fore.GREEN}{Style.BRIGHT}YOUR CARDS:{Style.RESET_ALL}")
                my_hand = [f"{rank_names[i]}(x{int(env.hands[0,0,i])})" for i in range(13) if env.hands[0,0,i] > 0]
                my_fu = [f"{rank_names[i]}(x{int(env.face_up_piles[0,0,i])})" for i in range(13) if env.face_up_piles[0,0,i] > 0]
                my_fd_count = int(env.face_down_piles[0, 0].sum())
                
                print(f" Hand:    {', '.join(my_hand) if my_hand else 'None'}")
                print(f" Face-Up: {', '.join(my_fu) if my_fu else 'None'}")
                print(f" Face-Down: [{'x ' * my_fd_count}]")
                print("-" * 30)

            # Inference
            masks = env.get_valid_mask()
            static_obs = create_static_input_vector(env)
            h, c = hidden_states[current_player]
            probs, (h_new, c_new) = king_model(action_history, static_obs, masks.unsqueeze(0), (h, c))
            hidden_states[current_player] = (h_new, c_new)

            if current_player == 0:
                suggested_action = torch.argmax(probs, dim=-1).item()
                valid_actions = torch.where(masks[0])[0].cpu().numpy().tolist()

                print(f"{Fore.WHITE}Suggested: {Fore.YELLOW}{suggested_action}{Style.RESET_ALL}")
                print(f"Options:")
                
                # Group and print actions cleanly
                for i, action in enumerate(valid_actions):
                    if action == 78:
                        desc = f"{Fore.RED}PICK UP PILE{Style.RESET_ALL}"
                    elif action <= 5 * 13:
                        cat, rank = action // 13, action % 13
                        desc = f"{category_names[cat]} {rank_names[rank]}"
                    else: 
                        cat, rank = action // 13, action % 13
                        desc = f"{category_names[cat]}"
                    
                    print(f"  [{Fore.YELLOW}{action:2}{Style.RESET_ALL}] {desc}", end="\t")
                    if (i + 1) % 3 == 0: print() # New line every 3 actions
                
                choice = -1
                while choice not in valid_actions:
                    try:
                        choice = int(input(f"\n\nChoose Action: "))
                    except ValueError: pass
                action = choice
            else:
                # AI turn summary
                action = torch.argmax(probs[0]).item()
                color = Fore.BLUE if current_player == 1 else Fore.MAGENTA
                if action == 78:
                    print(f"{color}Player {current_player} picks up the pile.{Style.RESET_ALL}")
                else:
                    cat, rank = action // 13, action % 13
                    print(f"{color}Player {current_player} plays {category_names[cat]} {rank_names[rank]}.{Style.RESET_ALL}")

            # Step and Update History
            action_tensor = torch.tensor([action], device=device)
            rewards, done = env.step(action_tensor)
            
            new_action_onehot = torch.zeros((batch_size, 1, 79), device=device)
            new_action_onehot[0, 0, action] = 1.0
            action_history = torch.cat((action_history[:, 1:, :], new_action_onehot), dim=1)

    results = np.argsort(env.finish_times.cpu().numpy())
    rankings = [("You" if p == 0 else f"Opponent {p}") for p in results if p != -1]
    print(f"\n{Fore.YELLOW}{Style.BRIGHT}🏆 FINAL RANKINGS: {' > '.join(rankings)}{Style.RESET_ALL}")

if __name__ == "__main__":
    # play_kings_against_kings('Palace_king.pth')
    play_against_kings('Palace_king.pth')




    
