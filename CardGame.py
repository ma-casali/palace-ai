import torch
from torch import nn
import random
from PalacePlayer import PalacePlayer

# Use global device or pass it to classes
# device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

def shuffle_deck():
    # Keep as list/cpu for initial randomization
    ranks = list(range(13)) * 4
    random.shuffle(ranks)
    return ranks

def get_valid_mask(hand, discard_pile_list, face_up_pile, face_down_pile):
    """
    All inputs except discard_pile_list should be torch tensors on 'device'.
    """
    mask = torch.zeros((6, 13), dtype=torch.bool, device=device)
    
    hand_sum = hand.sum()
    face_up_sum = face_up_pile.sum()
    face_down_sum = face_down_pile.sum()

    # possession logic
    for rank in range(13):
        if hand_sum == 0 and face_up_sum == 0 and face_down_sum > 0:
            mask[5, rank] = True # Face-down
        elif hand_sum == 0 and face_up_pile[rank] > 0:
            mask[4, rank] = True # Face-up
        else:
            # Hand play (1-4 cards)
            # Efficiently check counts
            for num_cards in range(1, 5):
                if hand[rank] >= num_cards:
                    mask[num_cards - 1, rank] = True

    mask_pickup = len(discard_pile_list.cards) > 0
    
    if mask_pickup:
        top_card = discard_pile_list.cards[-1]
        # Check for three (Rank 1)
        if top_card == 1:
            mask[:, torch.arange(13) != 1] = False
        else:
            # Restrictions
            ranks = torch.arange(13, device=device)
            # Wildcards: 2 (0), 3 (1), 7 (5), 10 (8)
            is_wild = (ranks == 0) | (ranks == 1) | (ranks == 5) | (ranks == 8)
            
            if top_card == 5: # 7 is on top (must play <= 7)
                mask[:, ranks >= 6] = False
            elif top_card != 0: # 2 is on top, no restrictions, otherwise:
                mask[:, (ranks < top_card) & (~is_wild)] = False

    # Flatten and append pickup bit
    pickup_tensor = torch.tensor([mask_pickup], dtype=torch.bool, device=device)
    return torch.cat([mask.flatten(), pickup_tensor])

class DrawPile:
    def __init__(self, deck):
        self.deck = deck # List of ints

    def draw(self, num_cards):
        drawn_cards = self.deck[:num_cards]
        self.deck = self.deck[num_cards:]
        return drawn_cards
    
class DiscardPile:
    def __init__(self):
        self.cards = [] # Simple list for easy sequential access

    def add(self, card_rank):
        self.cards.append(card_rank)

class PalaceEnv:
    def __init__(self, num_players=2):
        self.num_players = num_players
        self.reset()

    def reset(self, players=None):
        self.players = players if players else [PalacePlayer().to(device) for _ in range(self.num_players)]
        # Use Tensors on Device for all piles
        self.hands = torch.zeros((self.num_players, 13), dtype=torch.long, device=device)
        self.face_up_piles = torch.zeros((self.num_players, 13), dtype=torch.long, device=device)
        self.face_down_piles = torch.zeros((self.num_players, 13), dtype=torch.long, device=device)
        
        deck = shuffle_deck()
        self.draw_pile = DrawPile(deck)
        self.discard_pile = DiscardPile()
        self.deal()

    def deal(self):
        for i in range(self.num_players):
            # 3 Face-down, 3 Face-up, 3 Hand
            cards = self.draw_pile.draw(9)
            for j, card in enumerate(cards):
                if j < 3: self.face_down_piles[i, card] += 1
                elif j < 6: self.face_up_piles[i, card] += 1
                else: self.hands[i, card] += 1

    def step(self, action_idx, current_player_idx, players_out):
        done = False
        card_rank = None
        action_category = action_idx // 13
        step_reward = 0.0

        # Logic for Playing Cards
        if action_category < 4: # Hand
            card_rank = action_idx % 13
            num_cards = action_category + 1
            self.hands[current_player_idx, card_rank] -= num_cards
            if card_rank == 8: self.discard_pile.cards = [] # 10 burns
            else:
                for _ in range(num_cards): self.discard_pile.add(card_rank)
            step_reward += (num_cards / 4.0) if num_cards < 4 else 2.0

        elif action_category == 4: # Face-Up
            card_rank = action_idx % 13
            self.face_up_piles[current_player_idx, card_rank] -= 1
            if card_rank == 8: self.discard_pile.cards = []
            else: self.discard_pile.add(card_rank)
            step_reward += 5.0

        elif action_category == 5: # Face-Down (Blind)
            available = torch.where(self.face_down_piles[current_player_idx] > 0)[0]
            if len(available) > 0:
                chosen_rank = available[torch.randint(0, len(available), (1,))].item()
                self.face_down_piles[current_player_idx, chosen_rank] -= 1
                
                # Check failure
                top_card = self.discard_pile.cards[-1] if self.discard_pile.cards else -1
                fail = False
                if top_card != -1:
                    if top_card == 5 and chosen_rank < 6 and chosen_rank not in [0,1,5,8]: fail = True
                    elif chosen_rank < top_card and chosen_rank not in [0,1,5,8]: fail = True
                
                if fail:
                    self.hands[current_player_idx, chosen_rank] += 1
                    for c in self.discard_pile.cards: self.hands[current_player_idx, c] += 1
                    self.discard_pile.cards = []
                    step_reward -= 5.0
                else:
                    if chosen_rank == 8: self.discard_pile.cards = []
                    else: self.discard_pile.add(chosen_rank)
                    step_reward += 10.0
                    card_rank = chosen_rank

        elif action_idx == 78: # Pickup
            for c in self.discard_pile.cards: self.hands[current_player_idx, c] += 1
            penalty = len(self.discard_pile.cards) / 13.0
            self.discard_pile.cards = []
            step_reward -= float(penalty)

        # Draw cards
        while self.hands[current_player_idx].sum() < 3 and self.draw_pile.deck:
            drawn = self.draw_pile.draw(1)[0]
            self.hands[current_player_idx, drawn] += 1

        # Check Win
        if self.hands[current_player_idx].sum() == 0 and \
           self.face_up_piles[current_player_idx].sum() == 0 and \
           self.face_down_piles[current_player_idx].sum() == 0:
            players_out.append(current_player_idx)
            step_reward += 20.0 if len(players_out) == 1 else 10.0
            if len(players_out) == self.num_players - 1: done = True

        # Rotation
        if card_rank != 8:
            current_player_idx = (current_player_idx + 1) % self.num_players

        return done, current_player_idx, players_out, step_reward