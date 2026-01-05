import numpy as np
from CardGame import get_valid_mask, PalaceEnv, PalacePlayer
import torch
from colorama import Fore, Style, init

def create_input_vector(last_actions, current_hand, valid_mask):
    last_actions_flat = np.array(last_actions).flatten()  # Shape: (39,)
    current_hand_flat = np.array(current_hand).flatten()  # Shape: (13,)
    valid_mask_flat = np.array(valid_mask).flatten()      # Shape: (79,)

    input_vector = np.concatenate([last_actions_flat, current_hand_flat, valid_mask_flat])
    return input_vector  # Shape: (131,)

def play_against_kings(king_path):

    env = PalaceEnv(num_players = 3)

    # Load in the kings
    king_model = PalacePlayer()
    king_model.load_state_dict(torch.load(king_path, map_location=torch.device('cpu')))
    king_model.eval() 

    # First index is the user
    players = [None, king_model, king_model]
    action_history = [np.zeros(13) for _ in range(3)]
    env.reset(players)
    print(env.hands)

    players_out = [] # track players who have finished their game
    done = False
    current_player_idx = 0

    # map ranks to human readable
    rank_names = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]

    print("Game Start! You are Player 0.")

    with torch.inference_mode():
        while not done:
            if current_player_idx in players_out:
                current_player_idx = (current_player_idx + 1) % env.num_players
                continue

            if current_player_idx == 0: # user turn
                print(f"{Fore.BLACK}\n" + "="*30)
                print(f"TOP CARD: {rank_names[env.discard_pile.cards[-1]]+' with '+str(len(env.discard_pile.cards))+' cards in discard pile' if env.discard_pile.cards else 'Empty Discard Pile'}\n")
                print(f"Your Hand: {[f'{rank_names[i]} (x{int(env.hands[0][i])})' for i in range(13) if env.hands[0][i] > 0]}")
                print(f"Your Face-Up Pile: {[f'{rank_names[i]} (x{int(env.face_up_piles[0][i])})' for i in range(13) if env.face_up_piles[0][i] > 0]}")
                print(f"Your Face-Down Pile: {'x '*sum(env.face_down_piles[0]) if any(env.face_down_piles[0][i] > 0 for i in range(13)) else ''}\n")
                
                print(f"Opponent 1 has: {int(sum(env.hands[1]))} card(s) in hand.")
                print(f"Opponent 1's Face-Up Cards: {[f'{rank_names[i]} (x{int(env.face_up_piles[1][i])})' for i in range(13) if env.face_up_piles[1][i] > 0]}")
                print(f"Opponent 1's Face-Down Cards: {'x '*sum(env.face_down_piles[1]) if any(env.face_down_piles[1][i] > 0 for i in range(13)) else ''}\n")
                
                print(f"Opponent 2 has: {int(sum(env.hands[2]))} card(s) in hand.")
                print(f"Opponent 2's Face-Up Cards: {[f'{rank_names[i]} (x{int(env.face_up_piles[2][i])})' for i in range(13) if env.face_up_piles[2][i] > 0]}")
                print(f"Opponent 2's Face-Down Cards: {'x '*sum(env.face_down_piles[2]) if any(env.face_down_piles[2][i] > 0 for i in range(13)) else ''}\n")

                mask = get_valid_mask(env.hands[current_player_idx],
                                        env.discard_pile,
                                        env.face_up_piles[current_player_idx],
                                        env.face_down_piles[current_player_idx])
                input_vec = torch.tensor(create_input_vector(action_history, env.hands[current_player_idx], mask), dtype=torch.float32)
                mask_vec = torch.tensor(mask.flatten(), dtype=torch.bool)

                probs = king_model(input_vec, mask_vec)
                suggested_idx = torch.argmax(probs).item()
                valid_indices = np.where(mask)[0]

                print(f"\nValid Actions: AI suggests action [{suggested_idx}]")
                for idx in valid_indices:
                    if idx == 78:
                        print(f"[{idx}] - Pick up the deck")
                    else:
                        cat = idx // 13
                        rank = idx % 13
                        move_type = ["Hand", "Hand x 2", "Hand x 3", "Hand x 4", "Face-Up", "Face-Down"][cat]
                        if cat <= 4:
                            print(f"[{idx}] - Play from {move_type} {rank_names[rank]}")
                        else:
                            print(f"[{idx}] - Play from {move_type} (unknown card)")

                choice = -1
                while choice not in valid_indices:
                    try:
                        choice = int(input("\nEnter the number of your chosen action: "))
                    except ValueError:
                        pass

                action_idx = choice

            else: # AI turn
                mask = get_valid_mask(env.hands[current_player_idx],
                                        env.discard_pile,
                                        env.face_up_piles[current_player_idx],
                                        env.face_down_piles[current_player_idx])
                input_vec = torch.tensor(create_input_vector(action_history, env.hands[current_player_idx], mask), dtype=torch.float32)
                mask_vec = torch.tensor(mask.flatten(), dtype=torch.bool)

                probs = king_model(input_vec, mask_vec)
                action_idx = torch.argmax(probs).item()

                if action_idx == 78:
                    if current_player_idx == 1:
                        print(f"{Fore.RED}\nOpponent {current_player_idx} picks up the deck.")
                    else: 
                        print(f"{Fore.BLUE}\nOpponent {current_player_idx} picks up the deck.")
                else:
                    cat = action_idx // 13
                    rank = action_idx % 13
                    move_type = ["Hand", "Hand x 2", "Hand x 3", "Hand x 4", "Face-Up", "Face-Down"][cat]
                    if current_player_idx == 1:
                        print(f"{Fore.RED}\nOpponent {current_player_idx} plays from {move_type} {rank_names[rank]}")
                    else:
                        print(f"{Fore.BLUE}\nOpponent {current_player_idx} plays from {move_type} {rank_names[rank]}")

            # Execute step
            done, action_history, current_player_idx, players_out, _ = env.step(action_idx, action_history, current_player_idx, players_out)

    print("\nGAME OVER!")
    print(f"Final Rankings: {['You' if p == 0 else f'Opponent {p}' for p in players_out]}")

if __name__ == "__main__":
    play_against_kings("Palace_king.pth")