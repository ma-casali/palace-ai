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
    table_draw = env.drawpile_counts.sum(dim=1, keepdim=True).float()              # (batch_size, 1)

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

    # SET UP ENVIRONMENT AND MODEL
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    batch_size = 1
    env = PalaceEnv(batch_size=batch_size, num_players = 3, device=device)

    # load in the king player
    king_model = PalacePlayer().to(device)
    king_model.load_state_dict(torch.load(king_path, map_location=device))
    king_model.eval()

    # initialize action history (one-hot encoded)
    action_history = torch.zeros((batch_size, 6, 79), device=device)

    # initialize LSTM hidden states for all 3
    hidden_states = {
        p: (torch.zeros(2, batch_size, king_model.hidden_dim, device=device),
            torch.zeros(2, batch_size, king_model.hidden_dim, device=device))
        for p in range(3)
    }

    players = [king_model for _ in range(3)]
    env.reset(players)

    rank_names = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

    # PLAY THE GAME
    print(f"{Fore.GREEN}Game start! You are Player 0.{Style.RESET_ALL}")

    total_rewards = torch.zeros((batch_size, 3), device=device)
    with torch.no_grad(): # static weights
        while not env.done.all(): # play until the game is done
            current_player = env.active_players[0].item()

            if current_player == 0: # the user
                print(f"\n{Fore.CYAN}" + "="*30)
                print("Draw pile cards remaining:", env.drawpile_counts[0].sum().item())
                top_card_idx = env.top_cards[0].item()
                top_card_name = rank_names[top_card_idx % 13] if top_card_idx >= 0 else "None"
                card_amount = env.discard_counts[0].sum().item() if env.discard_counts[0].sum().item() > 0 else 0
                print(f"Top card on {card_amount}-card pile: {top_card_name} with run count {env.run_count[0].item()}")

                print(f"Your Hand: {[f'{rank_names[i]} (x{int(env.hands[0,0,i])})' for i in range(13) if env.hands[0,0,i] > 0]}")
                print(f"Face-Up:   {[f'{rank_names[i]} (x{int(env.face_up_piles[0,0,i])})' for i in range(13) if env.face_up_piles[0,0,i] > 0]}")
                print(f"Face-Down: {'x ' * int(env.face_down_piles[0, 0].sum())}")

            # get AI inference for suggestions or opponent moves
            masks = env.get_valid_mask()
            static_obs = create_static_input_vector(env)

            h, c = hidden_states[current_player]
            probs, (h_new, c_new) = king_model(
                action_history,
                static_obs,
                masks.unsqueeze(0),
                hidden_state=(h, c)
            ) 
            hidden_states[current_player] = (h_new, c_new)

            # Handle action selection
            category_names = ["Hand", "Hand 2 x", "Hand 3 x", "Hand 4 x","Face-Up", "Face-Down"]
            if current_player == 0: # user
                suggested_action = torch.argmax(probs, dim=-1).item()
                valid_actions = torch.where(masks[0])[0].cpu().numpy().tolist()

                print(f"AI suggests action: [{Fore.YELLOW}{suggested_action}{Style.RESET_ALL}] ")
                print(f"Valid actions: ")
                for action in valid_actions:
                    if action == 78:
                        num_cards = env.discard_counts[0].sum().item()
                        print(f" - [{Fore.YELLOW}{action}{Style.RESET_ALL}]: Pick up {num_cards}-card pile")
                    else:
                        category, rank = action // 13, action % 13
                        print(f" - [{Fore.YELLOW}{action}{Style.RESET_ALL}]: Play from {category_names[category]} {rank_names[rank]} ")

                choice = -1 
                while choice not in valid_actions:
                    choice = int(input(f"Enter your action choice: "))
                    if choice not in valid_actions:
                        print(f"{Fore.RED}Invalid action. Please choose again.{Style.RESET_ALL}")
                action = choice
            else: # AI players
                action = torch.argmax(probs[0]).item()
                color = Fore.BLUE if current_player == 1 else Fore.MAGENTA
                if action == 78:
                    num_cards = env.discard_counts[0].sum().item()
                    print(f"{color}Player {current_player} picked up the {num_cards}-card pile. [{action}]{Style.RESET_ALL}")
                else:
                    category, rank = action // 13, action % 13
                    print(f"{color}Player {current_player} plays action: Play from {category_names[category]} {rank_names[rank]} ([{action}]){Style.RESET_ALL}")

            # Step the environment
            action_tensor = torch.tensor([action], device=device)
            rewards, done = env.step(action_tensor)
            total_rewards += rewards
            print(f"Player {current_player}'s hand: {env.hands[0, current_player].cpu().numpy()}")

            for player in range(3):
                if rewards[0, player].item() <= 0: 
                    color = Fore.RED
                else:
                    color = Fore.GREEN
                print(f"Player {player} rewards for this turn : {color}{rewards[0, player].cpu().numpy():.2f}{Style.RESET_ALL} ({total_rewards[0, player].cpu().numpy():.2f})")

            # update the action history
            new_action_onehot = torch.zeros((batch_size, 1, 79), device=device)
            new_action_onehot[0, 0, action] = 1.0
            action_history = torch.cat((action_history[:, 1:, :], new_action_onehot), dim=1)

    # Print final results
    results = env.players_out[0].cpu().numpy().tolist()
    rankings = [("You" if p == 0 else f"Opponent {p}") for p in results if p != -1]
    print(f"\n{Fore.GREEN}Game Over! Final Rankings: {rankings}{Style.RESET_ALL}")

if __name__ == "__main__":
    play_kings_against_kings('Palace_king.pth')




    
