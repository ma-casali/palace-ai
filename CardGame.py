import numpy as np
import random

import torch
from torch import nn

def shuffle_deck():
    ranks = np.arange(13) # (2 to Ace)
    deck = np.tile(ranks, 4).tolist() # 4 suits
    random.shuffle(deck)
    return deck

def get_valid_mask(hand, discard_pile, face_up_pile, face_down_pile):
    mask = np.zeros((6, 13), dtype=bool)
    
    # 1. POSSESSION CHECK (Only mask True if the player OWNS the card)
    for rank in range(13):
        # Hand play (1-4 cards)
        for num_cards in range(1, 5):
            if hand[rank] >= num_cards:
                mask[num_cards - 1, rank] = True

        # Face-up play (Only if hand is empty)
        if np.sum(hand) == 0 and face_up_pile[rank] > 0:
            mask[4, rank] = True

        # Face-down play (Only if hand and face-up are empty)
        if np.sum(hand) == 0 and np.sum(face_up_pile) == 0 and face_down_pile[rank] > 0:
            mask[5, rank] = True

    mask_pickup = len(discard_pile.cards) > 0

    # 2. DISCARD PILE RESTRICTIONS
    if mask_pickup:
        top_card = discard_pile.cards[-1]

        # check for three first
        if top_card == 1:
            for rank in range(13):
                if rank != 1:
                    mask[:, rank] = False

        else: # normal restrictions
            for rank in range(13):
                is_wild_card = (rank == 0 or rank == 1 or rank == 5 or rank == 8)
                if not is_wild_card:
                    if top_card == 0: # 2 is on top
                        break # Everything already True from possession check stays True
                    elif top_card == 5: # 7 is on top
                        if rank >= 6: 
                            mask[:, rank] = False
                    elif rank < top_card:
                        mask[:, rank] = False

    # Flatten and add the pickup action (Index 78)
    return np.append(mask.flatten(), mask_pickup).astype(bool)

class DrawPile:
    def __init__(self, deck):
        self.deck = deck

    def draw(self, num_cards):
        drawn_cards = self.deck[:num_cards] # draws first num_cards
        self.deck = self.deck[num_cards:]
        return drawn_cards
    
class DiscardPile:
    def __init__(self):
        self.cards = []

    def add(self, cards):
        self.cards.append(cards)

class PalaceTurn:
    def __init__(self):
        return
    
class PalaceEnv:
    def __init__(self, num_players=2):
        self.num_players = num_players
        self.players = []
        self.hands = []
        self.face_up_piles = []
        self.face_down_piles = []
        self.rewards = [0 for _ in range(num_players)]

    def reset(self, players=None):
        self.rewards = [0 for _ in range(self.num_players)]
        self.players = players
        self.hands = []
        self.face_up_piles = []
        self.face_down_piles = []
        deck = shuffle_deck()
        self.draw_pile = DrawPile(deck)
        self.discard_pile = DiscardPile()

        for player in range(self.num_players):
            if len(players) < self.num_players:
                self.players.append(PalacePlayer())
            self.hands.append(np.zeros(13, dtype=int)) # 13 ranks
            self.face_up_piles.append(np.zeros(13, dtype=int)) # 13 ranks
            self.face_down_piles.append(np.zeros(13, dtype=int)) # 13 ranks
        
        self.deal()

    def add_card(self, player_index, pile, cards):
        for card in cards:
            pile[player_index][card] += 1

    def deal(self):
        cards_per_player = 9
        for i in range(self.num_players):
            deal_cards = self.draw_pile.deck[i * cards_per_player:(i + 1) * cards_per_player]
            self.add_card(i, self.face_down_piles, deal_cards[:3])
            self.add_card(i, self.face_up_piles, deal_cards[3:6])
            self.add_card(i, self.hands, deal_cards[6:9])
        self.draw_pile.deck = self.draw_pile.deck[self.num_players * cards_per_player:]

    def step(self, action_idx, action_history, current_player_idx, players_out):
        done = False
        card_rank = None
        action_category = action_idx // 13
        step_reward = 0.0

        # 1. HANDLE CARD PLAYING (Categories 0-5)
        if action_category < 4:  # From Hand
            card_rank = action_idx % 13
            num_cards = action_category + 1
            self.hands[current_player_idx][card_rank] -= num_cards

            if card_rank == 8: # 10 card
                self.discard_pile.cards = []
            else:
                for _ in range(num_cards):
                    self.discard_pile.add(card_rank)

            step_reward += float((num_cards / 4.0) if num_cards < 4 else 2.0)

        elif action_category == 4: # From Face-Up
            card_rank = action_idx % 13
            self.face_up_piles[current_player_idx][card_rank] -= 1
            self.discard_pile.add(card_rank)
            
            step_reward += 5.0

        elif action_category == 5: # From Face-Down
            available_ranks = np.where(self.face_down_piles[current_player_idx] > 0)[0]

            if len(available_ranks) > 0:
                chosen_rank = np.random.choice(available_ranks)
                card_rank = chosen_rank

                self.face_down_piles[current_player_idx][chosen_rank] -= 1
                top_card = self.discard_pile.cards[-1] if len(self.discard_pile.cards) > 0 else -1

                # determine if the player is able to place down this card.
                is_fail = False
                if top_card != -1:
                    if top_card == 0:
                        is_fail = False
                    elif top_card == 5:
                        if chosen_rank >= 6:
                            is_fail = False
                        else:
                            is_fail = True
                    elif chosen_rank < top_card:
                        is_fail = True
                
                if is_fail: # blind pick failed
                    self.discard_pile.add(chosen_rank)
                    picked_cards = list(self.discard_pile.cards)
                    for card in picked_cards:
                        self.hands[current_player_idx][card] += 1
                    self.discard_pile.cards = []
                    card_rank = None
                    step_reward -= 5.0
                else: # blind pick succeeded
                    if chosen_rank == 8: # 10 card
                        self.discard_pile.cards = []
                    else:
                        self.discard_pile.add(chosen_rank)
                    step_reward += 10.0

        # 2. HANDLE PICKUP (Action 78)
        elif action_idx == 78:
            picked_cards = list(self.discard_pile.cards) # Copy the list
            
            # Special logic for the '3' card (Rank 1)
            if len(picked_cards) > 0 and picked_cards[-1] == 1:
                picked_cards.pop()

            for card in picked_cards:
                self.hands[current_player_idx][card] += 1
            self.discard_pile.cards = []

            # Calculate Penalty
            pc_arr = np.array(picked_cards)
            if pc_arr.size > 0:
                mask = (pc_arr != 1) & (pc_arr != 5)
                bad_cards = pc_arr[mask]
                if bad_cards.size > 0:
                    penalty = len(picked_cards) / 13 * np.mean(12 - bad_cards + 1)
                    step_reward -= float(penalty)
                else:
                    step_reward -= 0.5
        
        # Update Action History for State Tracking
        action_history.pop(0)
        action_vec = np.zeros(13)
        if card_rank is not None:
            action_vec[card_rank] = (action_category + 1) if action_category < 4 else 1
        else:
            action_vec -= 1 # -1 signifies a pickup
        action_history.append(action_vec)

        # 3. ENVIRONMENT UPDATES (Burning 4-of-a-kind & Drawing)
        if any(np.sum(action_history, axis=0) == 4):
            self.discard_pile.cards = []

        while np.sum(self.hands[current_player_idx]) < 3 and len(self.draw_pile.deck) > 0:
            drawn = self.draw_pile.draw(1)[0]
            self.hands[current_player_idx][drawn] += 1

        # 4. WIN/LOSS CONDITION
        if np.sum(self.hands[current_player_idx]) == 0 and \
        np.sum(self.face_up_piles[current_player_idx]) == 0 and \
        np.sum(self.face_down_piles[current_player_idx]) == 0:
            
            if len(players_out) == 0:
                step_reward += 20.0 # First Place
            else: 
                step_reward += 10.0 # Second Place
                
            players_out.append(current_player_idx)

            if len(players_out) == self.num_players - 1:
                done = True
                return done, action_history, current_player_idx, players_out, step_reward

        # 5. TURN ROTATION
        if card_rank is not None:
            if card_rank == 1: # Play again (or skip) logic
                current_player_idx = (current_player_idx + 2) % self.num_players
            elif card_rank == 8: # 10 goes again
                pass 
            else:
                current_player_idx = (current_player_idx + 1) % self.num_players
        else:
            current_player_idx = (current_player_idx + 1) % self.num_players

        return done, action_history, current_player_idx, players_out, step_reward

class PalacePlayer(nn.Module):
    def __init__(self, input_size = 131, output_size = 79):
        super(PalacePlayer, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size) # outputs raw logits
        )

    def forward(self, x, mask):
        logits = self.network(x)
        masked_logits = logits.masked_fill(mask == 0, -1e9) # mask is provided externally
        return torch.softmax(masked_logits, dim=-1)