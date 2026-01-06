import torch
from torch import nn
import random
from PalacePlayer import PalacePlayer

REWARD_CONFIG = {
    # Frequency/Urgency
    'step_penalty': -0.05,
    'pickup_penalty_per_card': -0.1,
    'stalemate_penalty': -20.0,

    # Hand management
    'card_played_base': 0.1,
    'per_card_bonus': 0.02,
    'hand_size_penalty': -0.005,

    # Game Milestones
    'burn_pile_bonus': 1.0,
    'pickup_base_penalty': -0.2,
    'pickup_per_card_penalty': -0.02,
    'facedown_milestone_bonus': 1.0,
    'faceup_milestone_bonus': 1.0,

    # Winning/Losing
    'win_reward': 5.0,
    'lose_penalty': -5.0,
    'card_difference_bonus_rate': 0.01,
    'card_difference_penalty_rate': -0.05,

    # Wildcard bonuses
    'played_two': 0.1,
    'played_three': 0.5,
    'played_seven': 0.3,
    'played_ten': 0.7
}

class PalaceEnv:
    def __init__(self, batch_size = 32, num_players=3, device = 'cpu'):
        self.batch_size = batch_size
        self.num_players = num_players
        self.device = device

        self.turn_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.max_turns = 300

        # each state within the environment isof shape (batch, players, ranks)
        self.hands = torch.zeros((batch_size, num_players, 13), dtype=torch.long, device=device)
        self.face_up_piles = torch.zeros((batch_size, num_players, 13), dtype=torch.long, device=device)
        self.face_down_piles = torch.zeros((batch_size, num_players, 13), dtype=torch.long, device=device)

        # other piles (1, batch_size)
        self.discard_counts = torch.zeros((batch_size, 13), dtype=torch.long, device=device)
        self.top_cards = -torch.ones(batch_size, dtype=torch.long, device=device) # -1 means empty discard pile
        self.drawpile_counts = 4*torch.ones((batch_size, 13), dtype=torch.long, device=device)
        self.run_ranks = -1*torch.ones((batch_size,), dtype=torch.long, device=device)
        self.run_count = torch.zeros((batch_size,), dtype=torch.long, device=device)

        # meta-states
        self.active_players = torch.zeros(batch_size, dtype = torch.long, device = device)
        self.players_out = -1*torch.ones((batch_size, num_players), dtype = torch.long, device = device) # -1 means still in game
        self.done = torch.zeros(batch_size, dtype = torch.bool, device = device)
        self.stalemates = torch.zeros(batch_size, dtype = torch.bool, device = device)

    def reset(self, players):
        self.turn_counts.fill_(0)
        self.hands.fill_(0)
        self.face_up_piles.fill_(0)
        self.face_down_piles.fill_(0)
        self.discard_counts.fill_(0)
        self.top_cards.fill_(-1)
        self.drawpile_counts.fill_(4)
        self.run_ranks.fill_(-1)
        self.run_count.fill_(0)
        self.active_players.fill_(0)
        self.players_out.fill_(-1)
        self.done.fill_(False)
        self.stalemates.fill_(False)

        self.players = players

        base_deck = torch.arange(13, device=self.device).repeat(4)
        all_decks = base_deck.unsqueeze(0).repeat(self.batch_size, 1)
        for i in range(self.batch_size):
            all_decks[i] = all_decks[i, torch.randperm(52)]
        
        for p in range(self.num_players):
            start_idx = p * 9
            facedown_ranks = all_decks[:, start_idx:start_idx + 3]
            faceup_ranks = all_decks[:, start_idx + 3:start_idx + 6]
            hand_ranks = all_decks[:, start_idx + 6:start_idx + 9]

            self.face_down_piles[:, p].scatter_add_(1, facedown_ranks, torch.ones_like(facedown_ranks, dtype=torch.long))
            self.face_up_piles[:, p].scatter_add_(1, faceup_ranks, torch.ones_like(faceup_ranks, dtype=torch.long))
            self.hands[:, p].scatter_add_(1, hand_ranks, torch.ones_like(hand_ranks, dtype=torch.long))

        # store the remaining cards in the draw pile
        remaining_deck = all_decks[:, self.num_players * 9:]
        self.drawpile_counts.scatter_add_(1, remaining_deck, torch.ones_like(remaining_deck, dtype=torch.long))

    def step(self, actions):
        
        # actions: Tensor (batch_size, ) where each values is 0-78
        cfg = REWARD_CONFIG
        rewards = torch.zeros((self.batch_size, self.num_players), device = self.device)
        rewards += cfg['step_penalty']
        batch_ids = torch.arange(self.batch_size, device=self.device)
        active_hand_sizes = (
            self.hands[batch_ids, self.active_players].sum(dim=1) + 
            self.face_up_piles[batch_ids, self.active_players].sum(dim=1) + 
            self.face_down_piles[batch_ids, self.active_players].sum(dim=1)
        )
        rewards[batch_ids, self.active_players] += cfg['hand_size_penalty'] * active_hand_sizes

        card_ranks = actions % 13
        action_categories = actions // 13
        
        played_two   = (card_ranks == 0) & (actions < 78)
        played_three = (card_ranks == 1) & (actions < 78)
        played_seven = (card_ranks == 5) & (actions < 78)
        played_ten   = (card_ranks == 8) & (actions < 78)

        # Multiplier is (action_categories + 1) to account for number of cards played
        if played_two.any():
            mult = action_categories[played_two] + 1
            rewards[played_two, self.active_players[played_two]] += cfg['played_two'] * mult
      
        if played_three.any():
            mult = action_categories[played_three] + 1
            rewards[played_three, self.active_players[played_three]] += cfg['played_three'] * mult
            
        if played_seven.any():
            mult = action_categories[played_seven] + 1
            rewards[played_seven, self.active_players[played_seven]] += cfg['played_seven'] * mult
            
        if played_ten.any():
            mult = action_categories[played_ten] + 1
            rewards[played_ten, self.active_players[played_ten]] += cfg['played_ten'] * mult

        rewards[actions < 78, self.active_players[actions < 78]] += cfg['card_played_base'] + cfg['per_card_bonus'] * (torch.where(action_categories[actions < 78] + 1 <= 4, action_categories[actions < 78] + 1, 1))
        num_cards_played = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        num_cards_played += torch.where(actions < 52, action_categories + 1, 0)
        num_cards_played += torch.where((actions >= 52) & (actions < 78), 1, 0)

        # check if the rank matches the run
        run_match = (card_ranks == self.run_ranks)
        is_empty_pile = (self.top_cards == -1)
        self.run_count = torch.where(run_match, self.run_count + num_cards_played, num_cards_played)
        self.run_ranks = card_ranks

        # PLAY FROM HAND OR FACE-UP LOGIC
        self.hands[batch_ids, self.active_players, card_ranks] -= torch.where(actions < 52, action_categories + 1, 0)
        self.face_up_piles[batch_ids, self.active_players, card_ranks] -= torch.where((actions >= 52) & (actions < 65), 1, 0)
        rewards[(actions >= 52) & (actions < 65), self.active_players[(actions >= 52) & (actions < 65)]] += cfg['faceup_milestone_bonus']

        # FACEDOWN LOGIC
        facedown_mask = (actions // 13 == 5) & (self.face_down_piles[batch_ids, self.active_players].sum(dim = 1) > 0)
        if facedown_mask.any():
            rewards[facedown_mask, self.active_players[facedown_mask]] += cfg['facedown_milestone_bonus']
            facedown_player_ids = self.active_players[facedown_mask]
            chosen_ranks = torch.multinomial(self.face_down_piles[facedown_mask, facedown_player_ids].float(), 1).squeeze(1)
            self.face_down_piles[facedown_mask, self.active_players[facedown_mask], chosen_ranks] -= 1

            # evaluate success/failure of the pull
            top_discard = self.top_cards[facedown_mask] if (self.discard_counts[facedown_mask] > 0).all() else -1
            is_wild = (chosen_ranks == 0) | (chosen_ranks == 1) | (chosen_ranks == 5) | (chosen_ranks == 8)
            fail_7 = (top_discard == 5) & (chosen_ranks >= 6) & (~is_wild)
            fail_normal = (top_discard != 5) & (top_discard != -1) & (chosen_ranks < top_discard) & (~is_wild)
            is_fail = fail_7 | fail_normal

            facedown_batch_inds = batch_ids[facedown_mask]
            
            fail_batch_ids = facedown_batch_inds[is_fail]
            if fail_batch_ids.numel() > 0:
                self.hands[fail_batch_ids, self.active_players[fail_batch_ids], chosen_ranks[is_fail]] += 1
                rewards[fail_batch_ids] += cfg['pickup_base_penalty'] + cfg['pickup_per_card_penalty'] * self.discard_counts[fail_batch_ids].sum(dim=1) + cfg['played_facedown_failure_penalty']

            success_batch_ids = facedown_batch_inds[~is_fail]
            if success_batch_ids.numel() > 0:
                is_ten = (chosen_ranks[~is_fail] == 8)
                self.discard_counts[success_batch_ids, chosen_ranks[~is_fail]] += 1
                self.top_cards[success_batch_ids] = chosen_ranks[~is_fail]
                rewards[success_batch_ids, self.active_players[success_batch_ids]] += cfg['card_played_base'] + cfg['per_card_bonus']

        # PICKUP LOGIC 
        pickup_mask = (actions == 78)
        if pickup_mask.any():
            trapped_by_three = pickup_mask & (self.top_cards == 1)
            if trapped_by_three.any():
                self.discard_counts[trapped_by_three, 1] -= 1
                self.discard_counts = torch.clamp(self.discard_counts, min=0)
            
            self.hands[pickup_mask, self.active_players[pickup_mask]] += self.discard_counts[pickup_mask]
            self.discard_counts[pickup_mask] = torch.zeros_like(self.discard_counts[pickup_mask])
            self.top_cards[pickup_mask] = -1
            self.run_ranks[pickup_mask] = -1
            self.run_count[pickup_mask] = 0
            rewards[pickup_mask, self.active_players[pickup_mask]] += cfg['pickup_base_penalty'] + cfg['pickup_per_card_penalty'] * self.discard_counts[pickup_mask].sum(dim=1)
            rewards[pickup_mask, self.active_players[pickup_mask]] += cfg['hand_size_penalty'] * self.hands[pickup_mask, self.active_players[pickup_mask]].sum(dim=1)

        # DRAW CARDS IF NEEDED (UP TO 3)
        current_hand_totals = self.hands[torch.arange(self.batch_size), self.active_players].sum(dim = 1)
        missing_counts = torch.clamp(3 - current_hand_totals, min = 0)
        can_draw = (missing_counts > 0) & (self.drawpile_counts[batch_ids].sum(dim = 1) > 0)
        if can_draw.any():
            for _ in range(3):
                deck_full = (self.drawpile_counts[batch_ids].sum(dim = 1) > 0)
                draw_mask = (self.hands[torch.arange(self.batch_size), self.active_players].sum(dim = 1) < 3) & deck_full
                if not draw_mask.any(): break
                drawn_ranks = torch.multinomial(self.drawpile_counts[draw_mask].float(), 1).squeeze(1)
                self.hands[batch_ids[draw_mask], self.active_players[draw_mask], drawn_ranks] += 1
                self.drawpile_counts[draw_mask, drawn_ranks] -= 1

                rewards[draw_mask, self.active_players[draw_mask]] += cfg['hand_size_penalty']

        # Calculate who has NO cards left in any pile
        # Shape: (Batch, Players)
        has_no_cards = (self.hands.sum(dim=2) == 0) & \
                    (self.face_up_piles.sum(dim=2) == 0) & \
                    (self.face_down_piles.sum(dim=2) == 0)

        # Update players_out for anyone who is finished but not yet recorded
        for p in range(self.num_players):
            # Players who are finished but not in players_out
            already_recorded = (self.players_out == p).any(dim=1)
            just_finished = has_no_cards[:, p] & ~already_recorded
            
            if just_finished.any():
                jf_indices = torch.where(just_finished)[0]
                for idx in jf_indices:
                    # Find first empty slot (-1)
                    slots = (self.players_out[idx] == -1).nonzero(as_tuple=True)[0]
                    if slots.numel() > 0:
                        self.players_out[idx, slots[0]] = p
                        # Optional: Give reward to that specific player
                        rewards[idx, p] += cfg['win_reward'] + cfg['card_difference_bonus_rate'] * (self.num_players * 9 - (self.hands[idx].sum() + self.face_up_piles[idx].sum() + self.face_down_piles[idx].sum()))

        # GLOBAL DONE CHECK
        # A game is done if (num_players - 1) players have finished
        players_finished_count = (self.players_out != -1).sum(dim=1)
        newly_done = (players_finished_count >= self.num_players - 1) & ~self.done

        if newly_done.any():
            done_batches = torch.where(newly_done)[0]
            self.done[done_batches] = True
            
            for i in done_batches:
                # Find the one player NOT in self.players_out[i]
                out_list = self.players_out[i].tolist()
                loser_id = next(p for p in range(self.num_players) if p not in out_list)
                rewards[i, loser_id] += cfg['lose_penalty'] + cfg['card_difference_penalty_rate'] * (self.hands[i].sum() + self.face_up_piles[i].sum() + self.face_down_piles[i].sum())

        # add cards to discard piles, unless 10 (clear) or pickup
        not_pickup = (actions != 78)
        if not_pickup.any():
            self.discard_counts[not_pickup].scatter_add_(
                1,
                card_ranks[not_pickup].unsqueeze(1),
                num_cards_played[not_pickup].unsqueeze(1)
            )
        self.top_cards = torch.where(not_pickup, card_ranks, torch.tensor(-1, device=self.device))

        # ROTATE TO NEXT PLAYER
        is_burn = (card_ranks == 8) | (self.run_count >= 4)

        new_player = (self.active_players + 1) % self.num_players
        self.active_players = torch.where(~is_burn, new_player, self.active_players)

        for _ in range(self.num_players - 1):
            still_active_mask = ~self.done
            is_out_mask = (self.players_out == self.active_players.unsqueeze(1)).any(dim = 1)
            to_shift = is_out_mask & still_active_mask
            if not to_shift.any(): 
                break
            self.active_players[to_shift] = (self.active_players[to_shift] + 1) % self.num_players

        self.discard_counts = torch.where(is_burn.unsqueeze(1),
                                          torch.zeros_like(self.discard_counts),
                                          self.discard_counts)
        self.top_cards[pickup_mask] = -1
        self.run_ranks[pickup_mask] = -1
        self.run_count[pickup_mask] = 0

        # STALEMATE CHECK
        self.turn_counts += (~self.done).long()
        reached_limit = (self.turn_counts) >= self.max_turns
        stalemate_mask = reached_limit & (~self.done)
        if stalemate_mask.any():
            rewards[stalemate_mask, :] += cfg['stalemate_penalty']
            self.done[stalemate_mask] = True
            self.stalemates[stalemate_mask] = True
            
        return rewards, self.done

    def get_valid_mask(self):
        """
        All inputs except discard_pile_list should be torch tensors on 'device'.
        """
        mask = torch.zeros((self.batch_size, 6, 13), dtype=torch.bool, device= self.device)
        batch_ids = torch.arange(self.batch_size, device=self.device)

        # get the current player's piles
        hand = self.hands[batch_ids, self.active_players]
        faceup_pile = self.face_up_piles[batch_ids, self.active_players]
        facedown_pile = self.face_down_piles[batch_ids, self.active_players]

        is_hand_empty = (hand.sum(dim=1) == 0)
        is_faceup_empty = (faceup_pile.sum(dim=1) == 0)

        # 1. Play from hand
        for num in range(1, 5):
            mask[:, num - 1, :] = (hand >= num) & ~is_hand_empty.unsqueeze(1)

        # 2. Play from face-up pile
        mask[:, 4, :] = (faceup_pile > 0) & is_hand_empty.unsqueeze(1) & ~is_faceup_empty.unsqueeze(1)

        # 3. Play from face-down pile
        mask[:, 5, :] = (facedown_pile > 0) & is_hand_empty.unsqueeze(1) & is_faceup_empty.unsqueeze(1)

        # 4. Apply restrictions from table
        ranks = torch.arange(13, device = self.device)
        is_wild = (ranks == 0) | (ranks == 1) | (ranks == 5) | (ranks == 8)

        # three is top card
        invalid_3 = (self.top_cards == 1).unsqueeze(1) & (ranks.unsqueeze(0) != 1)
        mask[:, :5, :] &= ~invalid_3.unsqueeze(1)
        # top card on discard pile
        invalid = (ranks.unsqueeze(0) < self.top_cards.unsqueeze(1)) & (~is_wild.unsqueeze(0))
        # seven is top card
        invalid = torch.where((self.top_cards == 5).unsqueeze(1), (ranks.unsqueeze(0) >= 6) & (~is_wild.unsqueeze(0)), invalid)

        mask[:, :5, :] &= ~invalid.unsqueeze(1)
        flat_mask = mask.view(self.batch_size, 78)

        #5. Pickup option
        mask_pickup = (self.top_cards != -1) & (self.discard_counts.sum(dim=1) > 0)
        pickup_tensor = mask_pickup.unsqueeze(1)

        return torch.cat([flat_mask, pickup_tensor], dim=1)
