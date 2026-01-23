import pygame
import torch
import numpy as np
import shap
from VectorizedCardGame import PalaceEnv
from PalacePlayer import PalacePlayer

SCREEN_WIDTH, SCREEN_HEIGHT = 1400, 900
CARD_WIDTH, CARD_HEIGHT = 70, 100
OVERLAP_SPACING = 30
SPACING = 75
FPS = 60

class PalaceExplainer:
    def __init__(self, model, background_path, device):
        self.model = model
        self.device = device
        background = np.load(background_path)
        self.explainer = shap.Explainer(self.model_predict, background)

    def model_predict(self, data):
        # expand flattened data back to original shape
        X = torch.tensor(data, dtype=torch.float32, device = self.device)
        batch_size = X.shape[0]
        
        # first 6 * 79 = 474 are action history
        action_dim = 79
        seq_len = 6
        action_history_flat = X[:, :seq_len * action_dim] # (B, 474)
        static_obs = X[:, seq_len * action_dim:] # (B, 82)
        action_history = action_history_flat.view(batch_size, seq_len, action_dim) # (B, 6, 79)

        # initialize hidden states to zero
        h0 = torch.zeros(self.model.num_rnn_layers, batch_size, self.model.hidden_dim, device=self.device)
        c0 = torch.zeros(self.model.num_rnn_layers, batch_size, self.model.hidden_dim, device=self.device)

        # the model will see all actions as valid for SHAP analysis
        dummy_mask = torch.ones((batch_size, 79), device=self.device)
        with torch.no_grad():
            logits, _ = self.model(action_history, static_obs, dummy_mask, (h0, c0))
            return logits.cpu().numpy()
        
    def explain_turn(self, history, static, action_idx):

        flat_hist = history.view(1, -1).cpu().numpy()  # (1, 474)
        flat_static = static.cpu().numpy()
        current_input = np.hstack((flat_hist, flat_static))  # (1, 556)

        shap_values = self.explainer(current_input, max_evals=2048, batch_size=256)

        return shap_values
    
    def get_turn_explanation(self, history, static, action_idx, current_player):
        card_names = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
        action_dim = 79
        seq_len = 6
        
        # 1. Generate Action Labels
        labels = []
        for i in range(79):
            if i == 78:
                labels.append("picking up the pile")
            else:
                card_rank = i % 13
                cat = i // 13
                if cat == 0: labels.append(f"you play a {card_names[card_rank]}")
                elif 0 < cat < 4: labels.append(f"you play {cat + 1} {card_names[card_rank]}s")
                elif cat == 4: labels.append(f"you play face-up {card_names[card_rank]}")
                else: labels.append(f"you play a face-down card")

        # 2. Get SHAP impacts for this specific action
        # This should return an array of size (474 + 73)
        feature_impacts = self.explain_turn(history, static, action_idx)
        feature_impacts = feature_impacts.values[0, :, action_idx]  # (547,)
        
        # 3. Build the explanation string
        document_list = [f"The AI suggested {labels[action_idx]} because: "]
        
        # Get top three feature indices by absolute impact
        top_feature_indices = np.argsort(-np.abs(feature_impacts))[:3]
        
        # Convert static tensor to numpy for easy indexing
        static_np = static.squeeze().cpu().numpy()

        for reason_idx, feature_idx in enumerate(top_feature_indices):
            # Formatting prefixes/suffixes
            if reason_idx == 0: prefix, suffix = "  1. ", ","
            elif reason_idx == 1: prefix, suffix = "  2. ", ", and"
            else: prefix, suffix = "  3. ", "."

            if feature_idx >= seq_len * action_dim: # static features
                idx = np.int32(feature_idx - seq_len * action_dim)
                if idx < 13:
                    mult = np.int32(static_np[idx])  # number of that card in hand
                    document_list.append(f"{prefix}you have {mult} {card_names[idx]}{suffix}")
                elif idx >= 13 and idx < 26:
                    mult = np.int32(static_np[idx])
                    document_list.append(f"{prefix}you have {mult} faceup {card_names[idx - 13]}{suffix}")
                elif idx == 26:
                    document_list.append(f"{prefix}you have {np.int32(static_np[idx])} facedown cards{suffix}")
                elif idx == 27:
                    document_list.append(f"{prefix}opponent 1 has {np.int32(static_np[idx])} cards in their hand{suffix}")
                elif idx >= 28 and idx < 41:
                    mult = np.int32(static_np[idx])
                    document_list.append(f"{prefix}opponent 1 has {mult} faceup {card_names[idx - 28]}{suffix}")
                elif idx == 41:
                    document_list.append(f"{prefix}opponent 1 has {np.int32(static_np[idx])} facedown cards{suffix}")
                elif idx == 42:
                    document_list.append(f"{prefix}opponent 2 has {np.int32(static_np[idx])} cards in their hand{suffix}")
                elif idx >= 43 and idx < 56:
                    mult = np.int32(static_np[idx])
                    document_list.append(f"{prefix}opponent 2 has {mult} faceup {card_names[idx - 43]}{suffix}")
                elif idx == 56:
                    document_list.append(f"{prefix}opponent 2 has {np.int32(static_np[idx])} facedown cards{suffix}")
                elif idx >= 57 and idx < 70:
                    # say which card in discard pile
                    mult = np.int32(static_np[idx])
                    if mult == 1:
                        document_list.append(f"{prefix}there is 1 {card_names[idx - 57]} in the discard pile{suffix}")
                    else:
                        document_list.append(f"{prefix}there are {mult} {card_names[idx - 57]}s in the discard pile{suffix}")
                elif idx == 70:
                    document_list.append(f"{prefix}the top card on the pile is a {card_names[np.int32(static_np[idx])]}{suffix}")
                elif idx == 71:
                    document_list.append(f"{prefix}the run count is {np.int32(static_np[idx])}{suffix}")
                elif idx == 72:
                    document_list.append(f"{prefix}the drawpile has {np.int32(static_np[idx])} cards left{suffix}")
            else:
                history_turn = (feature_idx // action_dim) + 1
                action_idx = (feature_idx - 73) % action_dim
                document_list.append(f"{prefix}on turn -{seq_len - history_turn}, someone {labels[action_idx]}{suffix}")

        return document_list


class PalaceUI:
    def __init__(self, king_path, device):
        self.device = device

        # State for different pages
        self.state = "MENU"

        # menu buttons
        button_width, button_height = 200, 50
        self.start_button = pygame.Rect(SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2 - (button_height * 3) // 2, button_width, button_height)
        self.rules_button = pygame.Rect(SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2 + (button_height * 3) // 2, button_width, button_height)
        self.quit_button = pygame.Rect(SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2 + button_height * 3, button_width, button_height)
        self.show_rules = False

        # environment initialization
        self.env = PalaceEnv(batch_size=1, num_players=3, device=device)
        self.king_model = PalacePlayer().to(device)
        self.king_model.load_state_dict(torch.load(king_path, map_location=device))
        self.king_model.eval()

        # explainer class
        self.ai_explainer = PalaceExplainer(self.king_model, "shap_background.npy", device)
        self.show_insight_overlay = False
        self.current_explanation = ""

        # rnn state tracking
        self.action_history = torch.zeros((1, 6, 79)).to(device)
        self.hidden_states = {
            p: (torch.zeros(self.king_model.num_rnn_layers, 1, self.king_model.hidden_dim).to(device),
                torch.zeros(self.king_model.num_rnn_layers, 1, self.king_model.hidden_dim).to(device))
            for p in range(3)
        }

        # pygame initialization
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Palace AI - Human vs. 2 Kings")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Georgia", 24)
        self.big_font = pygame.font.SysFont("Georgia", 48)
        self.small_font = pygame.font.SysFont("Georgia", 14)
        self.rank_strings = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        
        self.clickable_hand = {}
        self.clickable_faceup = {}
        self.clickable_facedown = {}
        self.pile_rect = pygame.Rect(SCREEN_WIDTH//2 - CARD_WIDTH//2, SCREEN_HEIGHT//2 - CARD_HEIGHT, CARD_WIDTH, CARD_HEIGHT)

        # selection toggles
        self.selected_hand_indices = []
        self.selected_faceup_indices = []
        self.selected_facedown_indices = False

        # animations
        self.animations = []

        # player positions on table
        self.player_positions = {
            0: (SCREEN_WIDTH//2, SCREEN_HEIGHT - 150),  # Bottom - Human Player
            1: (SCREEN_WIDTH//4, CARD_HEIGHT // 2),                             # Top-Left - AI Player
            2: (SCREEN_WIDTH//4 * 3, CARD_HEIGHT // 2)               # Top-Right - AI Player 
        }

        icon_size = 40
        self.help_icon_rect = pygame.Rect(self.player_positions[0][0] + CARD_WIDTH * 2, self.player_positions[0][1], icon_size, icon_size)

        # Colors to use
        self.GOLD = (255, 215, 0)
        self.BLUE = (0, 191, 255)
        self.GREEN = (34, 139, 34)
        self.WHITE = (255, 255, 255)
        self.OFFWHITE = (245, 245, 245)

    # region ADJUST_COLOR
    def adjust_color(self, color, factor):
        return tuple(min(255, max(0, int(c * factor))) for c in color)

    # region RENDER_MENU
    def render_menu(self):
        self.screen.fill(self.GREEN)

        # Title
        title_surf = self.big_font.render("Palace - Play the Kings", True, self.OFFWHITE)
        self.screen.blit(title_surf, (SCREEN_WIDTH//2 - title_surf.get_width()//2, SCREEN_HEIGHT//4 - title_surf.get_height()//2))

        # Draw Buttons
        for btn, text in [(self.start_button, "Start Game"), (self.rules_button, "Rules"), (self.quit_button, "Quit")]:
            mouse_pos = pygame.mouse.get_pos()
            color = self.OFFWHITE if btn.collidepoint(mouse_pos) else self.adjust_color(self.OFFWHITE, 1.2)
            pygame.draw.rect(self.screen, color, btn, border_radius = 10)

            btn_text = self.font.render(text, True, self.BLUE)
            self.screen.blit(btn_text, (btn.centerx - btn_text.get_width()//2, btn.centery - btn_text.get_height()//2))

        if self.show_rules:
            self.draw_rules_overlay()

        pygame.display.flip()

    # region DRAW_INSIGHT_OVERLAY
    def draw_insight_overlay(self, explanation_text):
        # dimensions and position
        box_width = SCREEN_WIDTH // 3
        box_height = SCREEN_HEIGHT // 2
        box_rect = pygame.Rect(
            0, 0,
            box_width, box_height
        )
        box_rect.center = (SCREEN_WIDTH // 6, SCREEN_HEIGHT // 4 * 3)

        # Draw the background (Semi-transparent dark blue)
        overlay = pygame.Surface((box_width, box_height), pygame.SRCALPHA)
        overlay.fill((10, 25, 50, 235)) 
        self.screen.blit(overlay, box_rect.topleft)
        
        # Gold border
        pygame.draw.rect(self.screen, self.GOLD, box_rect, 3, border_radius=12)

        # 4. Blit lines to screen
        y_offset = box_rect.y
        padding = 20
        for line in explanation_text:
            text_surf = self.small_font.render(line, True, self.WHITE)
            self.screen.blit(text_surf, (box_rect.x + padding, y_offset + padding))
            y_offset += 30 # Line spacing

        # Instruction to close
        close_txt = self.font.render("Click anywhere to close", True, self.GOLD)
        self.screen.blit(close_txt, (box_rect.centerx - close_txt.get_width()//2, box_rect.bottom - 35))

    # region DRAW_RULES_OVERLAY
    def draw_rules_overlay(self):
        # Use a list of tuples: (Text, is_header)
        # This allows us to center headers and left-align the details
        rules_data = [
            ("Palace Rules", True),
            ("Primary Objective", True),
            ("Do not be the last player to play all of your cards.", True),
            ("", False),
            ("1. Play cards equal to or higher than the top of the pile.", False),
            ("2. Special Cards:", False),
            ("   [2]: Resets the pile. (play on anything except 3).", False),
            ("   [3]: Violence. Force next player to pick up and skip. (play on anything)", False),
            ("   [7]: Transparency. Next card must be 7 or lower. (play on anything except 3)", False),
            ("   [10]: Burn. Clears the pile and take another turn. (play on anything except 3)", False),
            ("3. If you can't play, you must pick up the pile.", False),
            ("4. You must be holding at least 3 cards while there are cards in the discard pile.", False),
            ("5. Face-Up and then Face-Down cards are played when you have no hand cards left.", False),
            ("Click anywhere to exit rules.", True),
            ("", False),
        ]

        overlay_w, overlay_h = SCREEN_WIDTH - 40, SCREEN_HEIGHT - 40
        overlay_rect = pygame.Rect(
            SCREEN_WIDTH//2 - overlay_w//2, 
            SCREEN_HEIGHT//2 - overlay_h//2, 
            overlay_w, overlay_h
        )

        overlay = pygame.Surface((overlay_w, overlay_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 225)) 
        self.screen.blit(overlay, overlay_rect.topleft)
        pygame.draw.rect(self.screen, self.GOLD, overlay_rect, 3, border_radius=15)

        padding = 30
        current_y = overlay_rect.y + padding

        for text, is_header in rules_data:
            color = self.WHITE if is_header else self.OFFWHITE
            font = self.font # Or use a bold font for headers
            
            surf = font.render(text, True, color)
            
            if is_header:
                # CENTER ALIGN headers
                x_pos = overlay_rect.centerx - surf.get_width() // 2
            else:
                # LEFT ALIGN body text with a margin
                x_pos = overlay_rect.x + 40
                
            self.screen.blit(surf, (x_pos, current_y))
            current_y += 35 # Line spacing

    # region LERP
    def lerp(self, start, end, t):
        return start + t * (end - start)
    
    # region ANIMATE_CARD_MOVE
    def animate_card_move(self, card_surf, start_pos, end_pos, duration = 30):
        self.animations.append({
            'start': start_pos,
            'end': end_pos,
            'surf': card_surf,
            'frame': 0,
            'duration': duration
        })

    # region RUN_ANIMATIONS
    def run_animations(self):
        for anim in self.animations[:]:
            t = anim['frame'] / anim['duration']
            # ease out effect
            t = 1 - (1 - t) ** 2

            curr_x = self.lerp(anim['start'][0], anim['end'][0], t)
            curr_y = self.lerp(anim['start'][1], anim['end'][1], t)

            self.screen.blit(anim['surf'], (curr_x, curr_y))

            anim['frame'] += 1
            if anim['frame'] >= anim['duration']:
                self.animations.remove(anim)
                          
    # region HANDLE_CLICK
    def handle_click(self, mouse_pos):
        # check hand
        if self.pile_rect.collidepoint(mouse_pos):
            return ('pile', None, None)
        
        for rect, data in reversed(list(self.clickable_hand.items())):
            if pygame.Rect(rect).collidepoint(mouse_pos):
                return ('hand', data['id'], data['rank'])
            
        # check faceup
        for rect, data in reversed(list(self.clickable_faceup.items())):
            if pygame.Rect(rect).collidepoint(mouse_pos):
                return ('faceup', data['id'], data['rank'])
            
        # check facedown
        for rect, data in reversed(list(self.clickable_facedown.items())):
            if pygame.Rect(rect).collidepoint(mouse_pos):
                return ('facedown', data['id'], data['rank'])
            
        return (None, None, None)
    
    # region GET_CLICKED_CARD_ID
    def get_clicked_card_id(self, mouse_pos):
        id_list = []
        for rect_coords in reversed(list(self.clickable_hand.keys())):
            if pygame.Rect(rect_coords).collidepoint(mouse_pos):
                id_list.extend(self.clickable_hand[rect_coords]['id'])
        for rect_coords in reversed(list(self.clickable_faceup.keys())):
            if pygame.Rect(rect_coords).collidepoint(mouse_pos):
                id_list.extend(self.clickable_faceup[rect_coords]['id'])
        return id_list
    
    # region CONVERT_SELECTION_TO_ACTION
    def convert_selection_to_action(self):

        remaining_cards = torch.where(self.env.face_down_piles[0, 0] == 1)[0]

        current_hand = self.get_flat_hand(self.env.hands[0, 0])
        if len(current_hand) != 0 and self.selected_hand_indices:
            selected_ranks = [current_hand[i] for i in self.selected_hand_indices]
            rank = selected_ranks[0] # validation to make sure all selected cards are the same rank
            if any(r != rank for r in selected_ranks): return None  # Invalid selection
            return (len(selected_ranks) - 1) * 13 + rank  # Calculate action index for hand cards
        
        if len(current_hand) == 0 and self.selected_faceup_indices:
            print(self.selected_faceup_indices)
            return 52 + self.selected_faceup_indices[0]  # Face-up card action index
        
        if remaining_cards.numel() == 0:
            print("Warning: Tried to play facedown card, but none are left.")
            return 78
        
        if len(current_hand) == 0 and self.env.face_up_piles[0, 0].sum().item() == 0 and self.selected_facedown_indices:
            # make this a random choice among facedown cards
            return torch.where(self.env.face_down_piles[0, 0] == 1)[0][0].item() + 65
        
        return 78

    # region GET_AI_ACTION
    def get_ai_action(self, player_idx):
        with torch.no_grad():
            masks = self.env.get_valid_mask()
            self.static_obs = self.create_static_input_vector(self.env)
            h, c = self.hidden_states[player_idx]
            probs, (h_new, c_new) = self.king_model(
                self.action_history,
                self.static_obs,
                masks[0],
                hidden_state=(h, c)
            )
            self.hidden_states[player_idx] = (h_new, c_new)
            return torch.argmax(probs, dim=-1).item(), probs[0, torch.argmax(probs, dim=-1)].item()
        
    # region UPDATE_HISTORY
    def update_history(self, action):
        new_action_onehot = torch.zeros((1, 1, 79), device=self.device)
        new_action_onehot[0, 0, action] = 1.0
        self.action_history = torch.cat((self.action_history[:, 1:, :], new_action_onehot), dim=1)

    # region CREATE_STATIC_INPUT_VECTOR
    def create_static_input_vector(self, env):
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

    # region GET_FLAT_HAND
    def get_flat_hand(self, hand_tensor):
        hand_list = []
        for rank_idx, count in enumerate(hand_tensor):
            hand_list.extend([rank_idx] * int(count.item()))
        return hand_list
    
    # region RENDER
    def render_game(self, suggested_action = None, confidence = None):
        self.screen.fill(self.GREEN)  # Green background
        self.clickable_hand = {}
        self.clickable_faceup = {}
        self.clickable_facedown = {}

        # Draw Top Card & Pile
        top_card = self.env.top_cards[0].item()
        if top_card >= 0:
            if suggested_action is not None and suggested_action == 78:
                glow_rect = self.pile_rect.inflate(8, 8)
                pygame.draw.rect(self.screen, self.GOLD, glow_rect, width=4)
            card_surf = self.get_card_surface(top_card)
            self.screen.blit(card_surf, self.pile_rect.topleft)
            pile_text = self.font.render(f"Pile Size: {int(self.env.discard_counts[0].sum().item())}", True, (255, 255, 255))
            run_text = self.font.render(f"Run: {int(self.env.run_count[0].item())}", True, (255, 255, 255))
            self.screen.blit(run_text, (self.pile_rect.centerx - run_text.get_width()//2, self.pile_rect.top - 5 - run_text.get_height()))
            self.screen.blit(pile_text, (self.pile_rect.centerx - pile_text.get_width()//2, self.pile_rect.bottom + 5))
        else:
            no_card_text = self.font.render("Pile is Empty", True, (255, 255, 255))
            pygame.draw.rect(self.screen, (0,0,0), self.pile_rect.inflate(4,4), width=2, border_radius=8)
            empty_pile_rect = pygame.draw.rect(self.screen, self.GREEN, self.pile_rect, border_radius=8, width=2)
            self.screen.blit(no_card_text, (SCREEN_WIDTH//2 - no_card_text.get_width()//2, SCREEN_HEIGHT//2 - 50))

        # Draw Draw Pile
        draw_pile_rect = pygame.Rect(SCREEN_WIDTH - CARD_WIDTH * 2, SCREEN_HEIGHT - CARD_HEIGHT - 50, CARD_WIDTH, CARD_HEIGHT)
        pygame.draw.rect(self.screen, (0, 0, 139), draw_pile_rect)  # Brown back of card
        draw_text = self.font.render("Draw Pile", True, (255, 255, 255))
        draw_card_text = self.font.render(f"{52 - self.env.drawpile_ptrs[0].item()}", True, (255, 255, 255))
        self.screen.blit(draw_text, (draw_pile_rect.centerx - draw_text.get_width()//2, draw_pile_rect.bottom + 5))
        self.screen.blit(draw_card_text, (draw_pile_rect.centerx - draw_card_text.get_width()//2, draw_pile_rect.centery - draw_card_text.get_height()//2))

        # Draw Face-Down Cards, for opponents then player
        for i, player_idx in enumerate(range(3)):
            facedown_count = self.env.face_down_piles[0, player_idx].sum().item()
            for j in range(facedown_count):
                x = self.player_positions[player_idx][0] - (CARD_WIDTH + 5) * 3 // 2 + j * (CARD_WIDTH + 5)
                y = self.player_positions[player_idx][1]
                pygame.draw.rect(self.screen, (0, 0, 139), (x, y, CARD_WIDTH, CARD_HEIGHT))  # Dark blue back of card
                if player_idx == 0 and self.env.hands[0, 0].sum().item() == 0 and self.env.face_up_piles[0, 0].sum().item() == 0:
                    rect = pygame.Rect(x, y, CARD_WIDTH, CARD_HEIGHT)
                    self.clickable_facedown[tuple(rect)] = {'id': j, 'rank': -1} 

        # Draw Face-Up Cards, for opponents then player
        for i, player_idx in enumerate(range(3)):
            faceup_cards = self.get_flat_hand(self.env.face_up_piles[0, player_idx])
            for j, card_idx in enumerate(faceup_cards):
                x = self.player_positions[player_idx][0] + j * (CARD_WIDTH + 5) + 10 - (CARD_WIDTH + 5) * 3 // 2
                y = self.player_positions[player_idx][1] + 10
                if player_idx == 0 and self.env.hands[0, 0].sum().item() == 0:
                    is_suggested = (suggested_action == card_idx + 52)
                    is_selected = (card_idx in self.selected_faceup_indices)
                    y -= 20 if is_selected else 0
                    rect = pygame.Rect(x, y, CARD_WIDTH, CARD_HEIGHT)
                    self.clickable_faceup[tuple(rect)] = {'id': j, 'rank': card_idx}
                    if is_suggested:
                        pygame.draw.rect(self.screen, self.GOLD, rect.inflate(8, 8), width=4)
                    if is_selected:
                        pygame.draw.rect(self.screen, self.BLUE, rect.inflate(4, 4), width=4)
                card_surf = self.get_card_surface(card_idx)
                self.screen.blit(card_surf, (x, y))

        # Draw Opponents' Hand Counts
        for i, player_idx in enumerate(range(1, 3)):
            x_card = self.player_positions[player_idx][0] + CARD_WIDTH + 5 - (CARD_WIDTH + 5) * 3 // 2
            y_card = self.player_positions[player_idx][1] + CARD_HEIGHT + 20
            hand_count = self.env.hands[0, player_idx].sum().item()
            hand_text = self.font.render(f"{int(hand_count)}", True, (255, 255, 255))
            x_text = x_card + CARD_WIDTH // 2 - hand_text.get_width() // 2
            y_text = y_card + CARD_HEIGHT // 2 - hand_text.get_height() // 2
            pygame.draw.rect(self.screen, (0, 0, 139), (x_card, y_card, CARD_WIDTH, CARD_HEIGHT))
            self.screen.blit(hand_text, (x_text, y_text))

        # Draw Player's Hand
        current_hand = self.get_flat_hand(self.env.hands[0, 0])
        num_cards = len(current_hand)
        if num_cards > 0:

            total_width = (num_cards - 1) * OVERLAP_SPACING + CARD_WIDTH
            start_x = self.player_positions[0][0] - total_width // 2
            y_pos = self.player_positions[0][1] - CARD_HEIGHT - 20

            suggested_rank = -1
            if suggested_action is not None and suggested_action < 78:
                suggested_rank = suggested_action % 13

            for i, rank_idx in enumerate(current_hand):
                x = start_x + (i * OVERLAP_SPACING)

                is_selected = i in self.selected_hand_indices
                is_suggested = rank_idx == suggested_rank

                if is_selected: y = y_pos - 20
                else: y = y_pos
                
                # Use full card width for the last card, visible slice for others
                click_width = CARD_WIDTH if i == num_cards - 1 or is_selected else OVERLAP_SPACING
                rect = pygame.Rect(x, y, click_width, CARD_HEIGHT)
                glow_rect = pygame.Rect(x, y, CARD_WIDTH, CARD_HEIGHT)
                
                # Draw suggestion highlight
                if is_suggested:
                    pygame.draw.rect(self.screen, self.GOLD, glow_rect.inflate(8, 8), width=4)
                if is_selected:
                    pygame.draw.rect(self.screen, self.BLUE, glow_rect.inflate(4, 4), width=4)

                # 2. DRAW CARD SURFACE
                card_surf = self.get_card_surface(rank_idx)
                self.screen.blit(card_surf, (x, y))
                pygame.draw.rect(self.screen, (0, 0, 0), rect, width=2)

                # 4. REGISTER CLICK ZONE
                self.clickable_hand[tuple(rect)] = {'id': i, 'rank': rank_idx}

        # Draw Help Icon
        mouse_pos = pygame.mouse.get_pos()
        help_color = self.OFFWHITE if self.help_icon_rect.collidepoint(mouse_pos) else self.adjust_color(self.OFFWHITE, 1.2)
        pygame.draw.circle(self.screen, self.GOLD, self.help_icon_rect.center, self.help_icon_rect.width // 2)
        pygame.draw.circle(self.screen, help_color, self.help_icon_rect.center, self.help_icon_rect.width // 2, 2)

        quest_surf = self.font.render("?", True, (0, 0, 0))
        self.screen.blit(quest_surf, (self.help_icon_rect.centerx - quest_surf.get_width()//2, self.help_icon_rect.centery - quest_surf.get_height()//2))

        if self.show_insight_overlay:
            self.draw_insight_overlay(self.current_explanation)

        pygame.display.flip()

    # region GET_CARD_SURFACE
    def get_card_surface(self, card_idx):
        rank = card_idx % 13
        suit = card_idx // 13
        ranks = self.rank_strings

        surf = pygame.Surface((CARD_WIDTH, CARD_HEIGHT))
        surf.fill((255, 255, 255))
        pygame.draw.rect(surf, (0, 0, 0), surf.get_rect(), 2)

        rank_text = self.font.render(ranks[rank], True, (0, 0, 0))

        surf.blit(rank_text, (5, 5))

        return surf

# region RUN_GAME
def run_game(king_path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ui = PalaceUI(king_path, device)
    running = True
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

            # menu logic
            if ui.state == "MENU":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if ui.show_rules:
                        ui.show_rules = False
                    elif ui.start_button.collidepoint(event.pos):
                        print("Starting Game...")
                        ui.env.reset([ui.king_model] * 3, levels = [1.0, 0.0, 0.0, 0.0])
                        ui.state = "GAME"
                    elif ui.rules_button.collidepoint(event.pos):
                        ui.show_rules = True
                    elif ui.quit_button.collidepoint(event.pos):
                        running = False
               

        if ui.state == "MENU":
            ui.render_menu()
        elif ui.state == "GAME":
            while not ui.env.done.all():
                current_player = ui.env.active_players[0].item()
                action_taken = None

                if current_player == 0:
                    # Player's turn
                    suggested_action, confidence = ui.get_ai_action(current_player)
                    valid_actions = torch.where(ui.env.get_valid_mask()[0])[0].cpu().numpy().tolist()
                    ui.render_game(suggested_action, confidence)

                    # Wait for player input
                    waiting = True
                    while waiting:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT: return
                            if event.type == pygame.MOUSEBUTTONDOWN:
                                if ui.help_icon_rect.collidepoint(event.pos):
                                    think_text = ui.font.render("Thinking...", True, (255, 255, 255))
                                    ui.screen.blit(think_text, ui.help_icon_rect.bottomright)

                                    pygame.display.flip()

                                    current_player = ui.env.active_players[0].item()
                                    explanation = ui.ai_explainer.get_turn_explanation(ui.action_history, ui.static_obs, suggested_action, current_player)

                                    ui.current_explanation = explanation
                                    ui.show_insight_overlay = True

                                elif ui.show_insight_overlay:
                                    ui.show_insight_overlay = False

                                zone, item_id, rank = ui.handle_click(event.pos)
                                if zone == 'hand':
                                    if item_id in ui.selected_hand_indices:
                                        ui.selected_hand_indices.remove(item_id)
                                    else:
                                        ui.selected_hand_indices.append(item_id)
                                elif zone == 'faceup':
                                    if rank in ui.selected_faceup_indices:
                                        ui.selected_faceup_indices.remove(rank)
                                    else:
                                        ui.selected_faceup_indices.append(rank)
                                elif zone == 'facedown':
                                    ui.selected_facedown_indices = not ui.selected_facedown_indices
                                elif zone == 'pile':
                                    action_taken = ui.convert_selection_to_action()
                                    if action_taken in valid_actions:
                                        if action_taken < 78 and action_taken >= 65:
                                            ui.env.step(torch.tensor([action_taken], device=device))
                                            start_pos =  (ui.player_positions[0][0] - (CARD_WIDTH + 5) * 3 // 2 + (CARD_WIDTH + 5),
                                                            ui.player_positions[0][1])
                                            end_pos = ui.pile_rect.topleft
                                            card_surf = ui.get_card_surface( ui.env.chosen_ranks[0].item() )
                                            ui.animate_card_move( card_surf, start_pos, end_pos, duration = 50 )

                                            while ui.animations:
                                                ui.clock.tick(FPS)
                                                ui.render_game(suggested_action, confidence)
                                                ui.run_animations()
                                                pygame.display.flip()

                                        waiting = False
                                    else:
                                        print("Invalid action selected. Please try again.")

                                ui.render_game(suggested_action, confidence)

                else:
                    # AI's turn
                    action_taken, _ = ui.get_ai_action(current_player)
                    if action_taken < 52:
                        rank = action_taken % 13

                        start_pos = (ui.player_positions[current_player][0] + CARD_WIDTH + 5 - (CARD_WIDTH + 5) * 3 // 2,
                                        ui.player_positions[current_player][1] + CARD_HEIGHT + 20)
                        end_pos = ui.pile_rect.topleft 
                        card_surf = ui.get_card_surface(rank)

                        ui.animate_card_move( card_surf, start_pos, end_pos)
                    
                    elif action_taken < 65 and action_taken >= 52:
                        rank = action_taken - 52

                        start_pos = (ui.player_positions[current_player][0] - (CARD_WIDTH + 5) * 3 // 2 + (CARD_WIDTH + 5),
                                        ui.player_positions[current_player][1] + 10)
                        end_pos = ui.pile_rect.topleft 
                        card_surf = ui.get_card_surface(rank)

                        ui.animate_card_move( card_surf, start_pos, end_pos)

                    elif action_taken < 78 and action_taken >= 65:
                        rank = action_taken - 65

                        start_pos = (ui.player_positions[current_player][0] - (CARD_WIDTH + 5) * 3 // 2 +  (CARD_WIDTH + 5),
                                        ui.player_positions[current_player][1])
                        end_pos = ui.pile_rect.topleft 
                        card_surf = ui.get_card_surface(rank)

                        ui.animate_card_move( card_surf, start_pos, end_pos)

                    else:
                        start_pos = ui.pile_rect.topleft
                        end_pos = (ui.player_positions[current_player][0] + CARD_WIDTH + 5 - (CARD_WIDTH + 5) * 3 // 2,
                                        ui.player_positions[current_player][1] + CARD_HEIGHT + 20)
                        card_surf = ui.get_card_surface(ui.env.top_cards[0].item())
                        ui.animate_card_move( card_surf, start_pos, end_pos)

                    while ui.animations:
                        ui.clock.tick(FPS)
                        ui.render_game()
                        ui.run_animations()
                        pygame.display.flip()

                # Step the environment
                if action_taken is not None :
                    if not (current_player == 0 and action_taken >= 65 and action_taken < 78):
                        ui.env.step(torch.tensor([action_taken], device=device))
                    ui.update_history(action_taken)
                    ui.selected_hand_indices = []
                    ui.selected_faceup_indices = []
                    ui.selected_facedown_indices = False

        ui.state = "MENU"

if __name__ == "__main__":
    run_game('Palace_king.pth')