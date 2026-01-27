import torch
from torch import nn
import random
from PalacePlayer import PalacePlayer

REWARD_CONFIG = {
    # Frequency/Urgency
    'step_penalty': -0.1,
    'pickup_penalty_per_card': -0.1,
    'stalemate_penalty': -20.0,

    # Hand management
    'card_played_base': 0.2,
    'per_card_bonus': 0.02,
    'hand_size_penalty': -0.1,

    # Game Milestones
    'burn_pile_bonus': 2.0,
    'pickup_base_penalty': -0.05,
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
    'played_ten': 0.7,

    # Other rewards/penalties
    'forced_pickup': 0.5
}

class PalaceEnv:
    def __init__(self, batch_size = 32, num_players=3, device = 'cpu'):
        self.batch_size = batch_size
        self.num_players = num_players
        self.device = device

        self.turn_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.max_turns = 300

        # each state within the environment isof shape (batch, players, ranks)
        self.hands = torch.zeros((batch_size, num_players, 13), dtype=torch.float32, device=device)
        self.face_up_piles = torch.zeros((batch_size, num_players, 13), dtype=torch.long, device=device)
        self.face_down_piles = torch.zeros((batch_size, num_players, 13), dtype=torch.long, device=device)

        # other piles (1, batch_size)
        self.discard_counts = torch.zeros((batch_size, 13), dtype=torch.float32, device=device)
        self.top_cards = -torch.ones(batch_size, dtype=torch.long, device=device) # -1 means empty discard pile
        self.drawpile_decks = torch.arange(13, device=self.device).repeat(self.batch_size, 4)
        self.drawpile_ptrs = torch.zeros((batch_size,), dtype=torch.long, device=device)
        self.run_ranks = -1*torch.ones((batch_size,), dtype=torch.long, device=device)
        self.run_count = torch.zeros((batch_size,), dtype=torch.long, device=device)

        # meta-states
        self.active_players = torch.zeros(batch_size, dtype = torch.long, device = device)
        self.finish_times = torch.zeros((batch_size, num_players), dtype = torch.long, device = device) # 0 means still in game
        self.done = torch.zeros(batch_size, dtype = torch.bool, device = device)
        self.stalemates = torch.zeros(batch_size, dtype = torch.bool, device = device)

    def reset(self, players, levels = [1.0, 0.0, 0.0, 0.0]):
        """
        level 0: standard game (start from beginning)
        level 1: noisy hands, no drawpile
        level 2: 0 cards in hand, face-up piles only
        level 3: 0 cards in hand, face-down piles only
        """
        self.turn_counts.fill_(0)
        self.hands.fill_(0)
        self.face_up_piles.fill_(0)
        self.face_down_piles.fill_(0)
        self.discard_counts.fill_(0)
        self.top_cards.fill_(-1)
        self.run_ranks.fill_(-1)
        self.run_count.fill_(0)
        self.active_players.fill_(0)
        self.finish_times.fill_(0)
        self.done.fill_(False)
        self.stalemates.fill_(False)
        self.players = players

        level_distribution = torch.tensor(levels, device=self.device)
        levels = torch.multinomial(level_distribution, self.batch_size, replacement=True)

        self.drawpile_decks = torch.arange(13, device=self.device).repeat(self.batch_size, 4)
        for i in range(self.batch_size):
            self.drawpile_decks[i] = self.drawpile_decks[i, torch.randperm(52)]
        
        for p in range(self.num_players):
            start_idx = p * 9
            facedown_ranks = self.drawpile_decks[:, start_idx:start_idx + 3]
            faceup_ranks = self.drawpile_decks[:, start_idx + 3:start_idx + 6]
            hand_ranks = self.drawpile_decks[:, start_idx + 6:start_idx + 9]

            self.face_down_piles[:, p].scatter_add_(1, facedown_ranks, torch.ones_like(facedown_ranks, dtype=torch.long))
            self.face_up_piles[:, p].scatter_add_(1, faceup_ranks, torch.ones_like(faceup_ranks, dtype=torch.long))
            self.hands[:, p].scatter_add_(1, hand_ranks, torch.ones_like(hand_ranks, dtype=self.hands.dtype))

        # store the remaining cards in the draw pile
        l0_mask = (levels == 0)
        l1_mask = (levels == 1)
        hi_mask = (levels >= 1)
        l2_mask = (levels >= 2)
        l3_mask = (levels >= 3)

        if l0_mask.any():
            self.drawpile_ptrs[l0_mask] = 27  # 52 - 27 = 25 cards remaining in draw pile
            # make sure the player with the lowest card after the dealer goes first
            self.get_starting_player(l0_mask)
            
        # pick a random player to go first
        self.active_players[hi_mask] = torch.randint(0, self.num_players, self.active_players[hi_mask].shape, device=self.device)

        if l1_mask.any():
            self.hands[l1_mask] = 0
            l1_batch_indices_raw = torch.where(l1_mask)[0]
            num_l1 = l1_batch_indices_raw.shape[0]
            num_players = self.num_players

            draw_counts = torch.randint(0, 5, (num_l1, num_players), device=self.device)
            total_to_draw = draw_counts.sum(dim=1)
            max_draw = total_to_draw.max().item()
            
            offsets = torch.arange(max_draw, device=self.device).unsqueeze(0)
            fetch_indices = self.drawpile_ptrs[l1_mask].unsqueeze(1) + offsets
            fetch_indices = torch.clamp(fetch_indices, max=51)

            drawn_ranks = torch.gather(self.drawpile_decks[l1_mask], 1, fetch_indices)
            
            boundaries = torch.cumsum(draw_counts, dim=1)
            lower_bounds = torch.cat([torch.zeros((num_l1, 1), device=self.device), boundaries[:, :-1]], dim=1)
            upper_bounds = boundaries
            
            card_indices = torch.arange(max_draw, device=self.device).view(1, 1, -1).expand(num_l1, num_players, -1)
            
            assign_mask = (card_indices >= lower_bounds.unsqueeze(2)) & (card_indices < upper_bounds.unsqueeze(2))

            final_batch_indices = l1_batch_indices_raw.view(-1, 1, 1).expand(-1, num_players, max_draw)[assign_mask]
            final_player_indices = torch.arange(num_players, device=self.device).view(1, -1, 1).expand(num_l1, -1, max_draw)[assign_mask]
            final_ranks = drawn_ranks.unsqueeze(1).expand(-1, num_players, -1)[assign_mask]

            if final_ranks.numel() > 0:
                self.hands.index_put_(
                    (final_batch_indices, final_player_indices, final_ranks.long()), 
                    torch.ones(final_ranks.shape[0], device=self.device), 
                    accumulate=True
                )
                self.drawpile_ptrs[l1_mask] += total_to_draw

        # add top discard to the discard pile to start for hi levels
        batch_indices = torch.where(hi_mask)[0]
        first_card_ranks = self.drawpile_decks[hi_mask, self.drawpile_ptrs[hi_mask]]
        self.drawpile_ptrs[hi_mask] += 1
        flat_idx = (batch_indices.long() * 13) + first_card_ranks
        source_ones = torch.ones(flat_idx.shape[0], device=self.device, dtype=self.discard_counts.dtype)
        self.discard_counts.view(-1).index_add_(0, flat_idx, source_ones)
        self.top_cards[hi_mask] = first_card_ranks

        self.drawpile_ptrs[hi_mask] = 52  # no draw pile for hi levels
        self.hands[l2_mask] = 0
        self.face_up_piles[l3_mask] = 0
    
        return levels
    
    def get_starting_player(self, mask):
        priority_order = [2, 3, 4, 6, 7, 9, 10, 11, 12, 0, 5, 1, 8]
        mask_indices = torch.where(mask)[0]
        for rank in priority_order:
            has_rank = self.hands[mask][:, :, rank].nonzero(as_tuple=True) # (batch_indices, player_indices)
            if has_rank[0].numel() > 0:
                self.active_players[mask_indices[has_rank[0]]] = has_rank[1].min()
                break

    def step(self, actions):
        
        # actions: Tensor (batch_size, ) where each values is 0-78
        cfg = REWARD_CONFIG
        rewards = torch.zeros((self.batch_size, self.num_players), device = self.device)
        penalty = cfg['step_penalty'] * (1.0 + self.turn_counts.float() / 100.0)
        players_still_in_game = (self.finish_times == 0)
        rewards = torch.where(players_still_in_game, rewards + penalty.unsqueeze(1), rewards)
        batch_ids = torch.arange(self.batch_size, device=self.device)
        active_hand_sizes = (
            self.hands[batch_ids, self.active_players].sum(dim=1) + 
            self.face_up_piles[batch_ids, self.active_players].sum(dim=1) + 
            self.face_down_piles[batch_ids, self.active_players].sum(dim=1)
        )
        rewards[batch_ids, self.active_players] += cfg['hand_size_penalty'] * active_hand_sizes

        self.chosen_ranks = -1*torch.ones(self.batch_size, dtype=torch.long, device=self.device) # for facedown tracking
        card_ranks = actions % 13
        action_categories = actions // 13

        # region FACEDOWN LOGIC
        facedown_mask = (actions // 13 == 5) & (self.face_down_piles[batch_ids, self.active_players].sum(dim = 1) > 0)
        if facedown_mask.any():
            rewards[facedown_mask, self.active_players[facedown_mask]] += cfg['facedown_milestone_bonus']
            facedown_player_ids = self.active_players[facedown_mask]
            card_ranks[facedown_mask] = torch.multinomial(self.face_down_piles[facedown_mask, facedown_player_ids].float(), 1).squeeze(1)
            self.chosen_ranks = card_ranks[facedown_mask]
            self.face_down_piles[facedown_mask, self.active_players[facedown_mask], card_ranks[facedown_mask]] -= 1

            # evaluate success/failure of the pull
            top_discard = self.top_cards[facedown_mask] if (self.discard_counts[facedown_mask].sum() > 0) else -1
            is_wild = (card_ranks[facedown_mask] == 0) | (card_ranks[facedown_mask] == 1) | (card_ranks[facedown_mask] == 5) | (card_ranks[facedown_mask] == 8)
            fail_7 = (top_discard == 5) & (card_ranks[facedown_mask] >= 6) & (~is_wild)
            fail_3 = (top_discard == 1) & (card_ranks[facedown_mask] != 1)
            fail_normal = (top_discard != 5) & (top_discard != -1) & (card_ranks[facedown_mask] < top_discard) & (~is_wild)
            is_fail = fail_7 | fail_normal | fail_3

            facedown_batch_inds = batch_ids[facedown_mask]
            
            fail_batch_ids = facedown_batch_inds[is_fail]
            if fail_batch_ids.numel() > 0:
                self.hands[fail_batch_ids, self.active_players[fail_batch_ids], card_ranks[facedown_mask][is_fail]] += 1
                rewards[fail_batch_ids, self.active_players[fail_batch_ids]] += cfg['pickup_base_penalty'] + cfg['pickup_per_card_penalty'] * self.discard_counts[fail_batch_ids].sum(dim=1)

                self.discard_counts[fail_batch_ids, 1] = 0 # make sure no one picks up a three
                self.hands[fail_batch_ids, self.active_players[fail_batch_ids]] += self.discard_counts[fail_batch_ids] # pick up on fail
                self.discard_counts[fail_batch_ids] = torch.zeros_like(self.discard_counts[fail_batch_ids])
                self.top_cards[fail_batch_ids] = -1
                self.run_ranks[fail_batch_ids] = -1
                self.run_count[fail_batch_ids] = 0
        
        # region PLAY FROM HAND OR FACE-UP LOGIC
        played_two   = (card_ranks == 0) & (actions < 78)
        played_three = (card_ranks == 1) & (actions < 78)
        played_seven = (card_ranks == 5) & (actions < 78)
        played_ten   = (card_ranks == 8) & (actions < 78)

        # Multiplier is (action_categories + 1) to account for number of cards played
        rewards += (played_two.float() * cfg['played_two'] * (action_categories.float() + 1)).unsqueeze(1)
        rewards += (played_three.float() * cfg['played_three'] * (action_categories.float() + 1)).unsqueeze(1)
        rewards += (played_seven.float() * cfg['played_seven'] * (action_categories.float() + 1)).unsqueeze(1)
        rewards += (played_ten.float() * cfg['played_ten'] * (action_categories.float() + 1)).unsqueeze(1)

        rewards[actions < 78, self.active_players[actions < 78]] += cfg['card_played_base'] + cfg['per_card_bonus'] * (torch.where(action_categories[actions < 78] + 1 <= 4, action_categories[actions < 78] + 1, 1))
        num_cards_played = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        num_cards_played += torch.where(actions < 52, action_categories + 1, 0)
        num_cards_played += torch.where((actions >= 52) & (actions < 78), 1, 0)

        # check if the rank matches the run
        run_match = (card_ranks == self.run_ranks)
        is_empty_pile = (self.top_cards == -1)
        self.run_count = torch.where(run_match, self.run_count + num_cards_played, num_cards_played)
        self.run_ranks = card_ranks

        # take counts from hands/face-up piles and place on discard pile/make top card
        play_from_hand_mask = (actions < 52)
        play_from_faceup_mask = (actions >= 52) & (actions < 65)
        self.hands[play_from_hand_mask, self.active_players[play_from_hand_mask], card_ranks[play_from_hand_mask]] -= action_categories[play_from_hand_mask] + 1
        self.face_up_piles[play_from_faceup_mask, self.active_players[play_from_faceup_mask], card_ranks[play_from_faceup_mask]] -= 1
        rewards[(actions >= 52) & (actions < 65), self.active_players[(actions >= 52) & (actions < 65)]] += cfg['faceup_milestone_bonus']
        # endregion

        # region PICKUP LOGIC 
        pickup_mask = (actions == 78)

        trapped_by_three = pickup_mask & (self.top_cards == 1)
        self.discard_counts[trapped_by_three, 1] = 0
        self.discard_counts = torch.clamp(self.discard_counts, min=0)
        
        # pickup penalty before zeroing out discard pile
        pickup_penalty = cfg['pickup_base_penalty'] + cfg['pickup_per_card_penalty'] * self.discard_counts.sum(dim=1)
        
        self.hands[pickup_mask, self.active_players[pickup_mask]] += self.discard_counts[pickup_mask]
        self.discard_counts[pickup_mask] = torch.zeros_like(self.discard_counts[pickup_mask])
        self.top_cards[pickup_mask] = -1
        self.run_ranks[pickup_mask] = -1
        self.run_count[pickup_mask] = 0
        rewards[pickup_mask, self.active_players[pickup_mask]] += pickup_penalty[pickup_mask]
        rewards[pickup_mask, self.active_players[pickup_mask]] += cfg['hand_size_penalty'] * self.hands[pickup_mask, self.active_players[pickup_mask]].sum(dim=1)
        
        only_pickup_available = pickup_mask & (self.get_valid_mask().sum(dim=1) == 1)
        rewards[only_pickup_available, (self.active_players[only_pickup_available] - 1) % self.num_players] += cfg['forced_pickup']
        
        # endregion

        # region DRAW CARDS IF NEEDED (UP TO 3)
        current_hand_totals = self.hands[torch.arange(self.batch_size), self.active_players].sum(dim = 1)
        missing_counts = torch.clamp(3 - current_hand_totals, min = 0)
        deck_totals = 52 - self.drawpile_ptrs
        can_draw = (missing_counts > 0) & (deck_totals > 0) & ~self.done
        if can_draw.any():
            draw_batch_ids = batch_ids[can_draw]
            player_indices = self.active_players[can_draw]
            
            actual_draw_counts = torch.minimum(missing_counts[can_draw], deck_totals[can_draw])

            offsets = torch.arange(3, device=self.device, dtype=torch.long)
            fetch_idx = self.drawpile_ptrs[draw_batch_ids].unsqueeze(1) + offsets  # (num_draws, 3)
            fetch_idx = torch.clamp(fetch_idx, max=51)

            drawn_ranks = torch.gather(self.drawpile_decks[draw_batch_ids], 1, fetch_idx)

            # mask out what we don't need
            draw_idx_grid = torch.arange(3, device=self.device).expand(actual_draw_counts.shape[0], -1)
            valid_mask = draw_idx_grid < actual_draw_counts.unsqueeze(1)

            # flatten for indexing
            flat_batches = draw_batch_ids.unsqueeze(1).expand(-1, 3)[valid_mask]
            flat_players = player_indices.unsqueeze(1).expand(-1, 3)[valid_mask]
            flat_ranks = drawn_ranks[valid_mask]

            if flat_ranks.numel() > 0:
                # update hands and drawpile counts
                self.hands[flat_batches, flat_players, flat_ranks] += 1
                self.drawpile_ptrs[draw_batch_ids] += actual_draw_counts.long()
                
            # update rewards
            cards_per_player = actual_draw_counts
            relevant_player_ids = self.active_players[can_draw]
            relevant_batch_ids = batch_ids[can_draw]
            rewards[relevant_batch_ids, relevant_player_ids] += cards_per_player * cfg['hand_size_penalty']

        # Calculate who has NO cards left in any pile
        # Shape: (Batch, Players)
        has_no_cards = (self.hands.sum(dim=2) == 0) & (self.face_up_piles.sum(dim=2) == 0) & (self.face_down_piles.sum(dim=2) == 0)
        # determine if they were already recorded as out
        is_out_already = (self.finish_times != 0).any(dim = 1, keepdim=True)
        is_already_recorded = (self.finish_times != 0)

        # check if anyone just finished
        just_finished_mask = has_no_cards & ~is_already_recorded
        if just_finished_mask.any():
            # determine if they are the first out
            winner_mask = ~is_out_already & just_finished_mask.any(dim = 1)
            winner_batch_ids = torch.where(winner_mask)[0]
            total_cards_in_game = self.hands[winner_batch_ids].sum(dim=(1,2)) + \
                                    self.face_up_piles[winner_batch_ids].sum(dim=(1,2)) + \
                                    self.face_down_piles[winner_batch_ids].sum(dim=(1,2))
            # give rewards to the winners
            bonus = cfg['card_difference_bonus_rate'] * (self.num_players * 9 - total_cards_in_game)
            rewards[winner_batch_ids, self.active_players[winner_batch_ids]] += cfg['win_reward'] + bonus

            self.finish_times[just_finished_mask] = self.turn_counts[just_finished_mask.any(dim = 1)]

        # region GLOBAL DONE CHECK
        # A game is done if (num_players - 1) players have finished
        players_finished_count = (self.finish_times != 0).sum(dim=1)
        newly_done = (players_finished_count >= self.num_players - 1) & ~self.done

        if newly_done.any():
            self.done[newly_done] = True

            # find losers in each newly done batch
            is_loser = (self.finish_times == 0) & newly_done.unsqueeze(1)
            self.finish_times[is_loser] = self.turn_counts[is_loser.any(dim = 1)] + 1

            # assign penalties to the losers
            negative_bonus = cfg['card_difference_penalty_rate'] * (
                self.hands[newly_done].sum(dim=(1,2)) +
                self.face_up_piles[newly_done].sum(dim=(1,2)) +
                self.face_down_piles[newly_done].sum(dim=(1,2))
            )
            rewards[is_loser] += cfg['lose_penalty'] + negative_bonus
        # endregion

        # region UPDATE DISCARD PILE COUNTS AND TOP CARDS
        # add cards to discard piles, unless 10 (clear) or pickup
        standard_play_mask = (actions < 78) 
        if standard_play_mask.any():
            self.discard_counts.scatter_add_(
                1,
                card_ranks[standard_play_mask].unsqueeze(1),
                num_cards_played[standard_play_mask].unsqueeze(1).to(self.discard_counts.dtype)
            )
            self.top_cards[standard_play_mask] = card_ranks[standard_play_mask]
        # endregion

        # region ROTATE TO NEXT PLAYER
        is_burn = (card_ranks == 8) | (self.run_count >= 4) 
        self.active_players = torch.where(is_burn, self.active_players, (self.active_players + 1) % self.num_players)

        # this for loop ensures that we skip over players who are already done,
        # computationally efficient over vectorization for this case
        for _ in range(self.num_players - 1):
            still_active_mask = ~self.done
            is_out_mask = (torch.gather(self.finish_times, 1, self.active_players.unsqueeze(1)).squeeze(1) != 0)
            to_shift = is_out_mask & still_active_mask
            if not to_shift.any(): 
                break
            self.active_players[to_shift] = (self.active_players[to_shift] + 1) % self.num_players
        # endregion

        # region EXECUTE BURN
        self.discard_counts[pickup_mask | is_burn] = 0
        self.top_cards[pickup_mask | is_burn] = -1
        self.run_ranks[pickup_mask | is_burn] = -1
        self.run_count[pickup_mask | is_burn] = 0
        # endregion

        # region STALEMATE CHECK
        self.turn_counts += (~self.done).long()
        reached_limit = (self.turn_counts) >= self.max_turns
        stalemate_mask = reached_limit & (~self.done)
        if stalemate_mask.any():
            total_cards = self.hands.sum(dim=2) + self.face_up_piles.sum(dim=2) + self.face_down_piles.sum(dim=2)
            is_still_in = (total_cards > 0)
            
            involved_players = stalemate_mask.unsqueeze(1) & is_still_in

            rewards[involved_players] += cfg['stalemate_penalty']

            self.done[stalemate_mask] = True
            self.stalemates[stalemate_mask] = True
        # endregion
            
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
