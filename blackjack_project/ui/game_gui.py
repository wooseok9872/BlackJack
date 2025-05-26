import pygame
import os
from game.blackjack_env import BlackjackEnv, calculate_hand_value
from ai.dqn_agent_expanded import DQNAgent

WHITE = (255, 255, 255)
GREEN = (0, 128, 0)
BLACK = (0, 0, 0)
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
CARD_WIDTH = 80
CARD_HEIGHT = 120

class BlackjackGUI:
    def __init__(self, env, agent):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Blackjack: Human vs AI")
        self.font = pygame.font.SysFont("arial", 24)
        self.large_font = pygame.font.SysFont("arial", 32, bold=True)

        self.env = env
        self.agent = agent
        self.done = False
        self.awaiting_bet = True
        self.human_turn = False
        self.selected_bet = 0
        self.clock = pygame.time.Clock()

        self.card_images = self.load_card_images()
        self.background = pygame.transform.scale(pygame.image.load(os.path.join("assets", "table", "background.jpg")), (SCREEN_WIDTH, SCREEN_HEIGHT))

        self.show_result = False
        self.dealer_action_text = ""
        self.dealer_steps = []
        self.dealer_step_index = 0
        self.dealer_step_timer = 0
        self.buttons = {}
        self.bet_buttons = []
        self.final_results = {}
        self.error_message = ""

    def load_card_images(self):
        images = {}
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
        values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'jack', 'queen', 'king', 'ace']
        for suit in suits:
            for value in values:
                filename = f"{value}_of_{suit}.png"
                path = os.path.join("assets", "cards", filename)
                if os.path.exists(path):
                    images[(value, suit)] = pygame.transform.scale(pygame.image.load(path), (CARD_WIDTH, CARD_HEIGHT))
        images["back"] = pygame.transform.scale(pygame.image.load(os.path.join("assets", "cards", "back.png")), (CARD_WIDTH, CARD_HEIGHT))
        return images

    def draw_text(self, text, x, y, color=WHITE, font=None):
        font = font or self.font
        label = font.render(text, True, color)
        self.screen.blit(label, (x, y))

    def draw_hand(self, hand, x_start, y, reveal_all=True):
        VALUE_TO_FILENAME = {'J': 'jack', 'Q': 'queen', 'K': 'king', 'A': 'ace'}
        for i, card in enumerate(hand):
            val = str(card[0])
            suit = card[1].lower()
            val_name = VALUE_TO_FILENAME.get(val, val)
            card_key = (val_name, suit)
            if reveal_all or i == 0:
                image = self.card_images.get(card_key, self.card_images["back"])
            else:
                image = self.card_images["back"]
            self.screen.blit(image, (x_start + i * (CARD_WIDTH + 10), y))

    def draw_game_state(self):
        self.screen.blit(self.background, (0, 0))
        self.draw_text("[H]: HIT   [S]: STAND", 20, 20, font=self.font)
        self.draw_text("DEALER", 460, 50, font=self.large_font)

        reveal_dealer = self.env.done['HUMAN'] and self.env.done['AI']
        self.draw_hand(self.env.get_hand("DEALER"), 370, 90, reveal_all=reveal_dealer)
        if reveal_dealer:
            dealer_score = calculate_hand_value(self.env.get_hand("DEALER"))
            self.draw_text(f"Score: {dealer_score}", 460, 230)

        self.draw_text("HUMAN", 100, 400, font=self.large_font)
        self.draw_hand(self.env.get_hand("HUMAN"), 80, 440, reveal_all=True)
        human_score = calculate_hand_value(self.env.get_hand("HUMAN"))
        self.draw_text(f"Score: {human_score}", 100, 580)
        self.draw_text(f"Chips: €{self.env.chips['HUMAN']}", 100, 610)

        self.draw_text("AI", 740, 400, font=self.large_font)
        self.draw_hand(self.env.get_hand("AI"), 720, 440, reveal_all=True)
        ai_score = calculate_hand_value(self.env.get_hand("AI"))
        self.draw_text(f"Score: {ai_score}", 740, 580)
        self.draw_text(f"Chips: €{self.env.chips['AI']}", 740, 610)

        if self.dealer_action_text:
            self.draw_text(self.dealer_action_text, 420, 260, font=self.large_font)

        if self.awaiting_bet:
            self.draw_text("Place your bet:", 300, 550, font=self.large_font)
            self.bet_buttons = []
            bet_options = [50, 100, 200]
            for i, amount in enumerate(bet_options):
                if self.env.chips["HUMAN"] >= amount:
                    rect = pygame.Rect(300 + i * 110, 590, 100, 50)
                    pygame.draw.rect(self.screen, (200, 200, 200), rect)
                    self.draw_text(f"{amount}", rect.x + 30, rect.y + 15)
                    self.bet_buttons.append((rect, amount))
            if not self.bet_buttons:
                self.draw_text("칩이 부족하여 베팅할 수 없습니다!", 300, 650, (255, 0, 0), font=self.large_font)

        if self.env.chips["HUMAN"] <= 0:
            self.draw_text("GAME OVER - 칩이 모두 소진되었습니다!", 250, 300, (255, 0, 0), font=self.large_font)
            self.draw_button("restart", "RESTART GAME", 400, 400)

        if self.error_message:
            self.draw_text(self.error_message, 320, 650, (255, 0, 0), font=self.font)

        if self.show_result:
            results = self.final_results
            self.draw_text("RESULTS:", 400, 300, font=self.large_font)
            self.draw_text("HUMAN: " + results.get("HUMAN", ""), 400, 330)
            self.draw_text("AI: " + results.get("AI", ""), 400, 360)
            self.draw_button("next", "NEXT ROUND", 400, 420)
            self.draw_button("restart", "RESTART GAME", 400, 500)

    def draw_button(self, name, text, x, y):
        rect = pygame.Rect(x, y, 200, 70)
        pygame.draw.rect(self.screen, (180, 180, 180), rect)
        text_surface = self.large_font.render(text, True, BLACK)
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)
        self.buttons[name] = rect

    def handle_bet_click(self, pos):
        for rect, amount in self.bet_buttons:
            if rect.collidepoint(pos):
                self.selected_bet = amount
                success = self.env.place_bet("HUMAN", amount)
                if not success:
                    self.error_message = "칩이 부족해서 베팅할 수 없습니다."
                    return
                self.error_message = ""
                state = self.env.get_state("AI")
                ai_bet_action = self.agent.choose_action(state, phase='BETTING')
                self.env.step("AI", ai_bet_action)
                self.awaiting_bet = False
                self.human_turn = True
                break

    def run_game(self):
        while not self.done:
            self.clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.done = True
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.awaiting_bet:
                        self.handle_bet_click(event.pos)
                    elif self.show_result:
                        if self.buttons.get("next") and self.buttons["next"].collidepoint(event.pos):
                            self.env.reset()
                            self.awaiting_bet = True
                            self.selected_bet = 0
                            self.human_turn = False
                            self.show_result = False
                            self.dealer_action_text = ""
                            self.final_results = {}
                            self.dealer_steps = []
                            self.dealer_step_index = 0
                            self.dealer_step_timer = 0
                            self.error_message = ""
                        elif self.buttons.get("restart") and self.buttons["restart"].collidepoint(event.pos):
                            self.env = BlackjackEnv()
                            self.agent = DQNAgent()
                            self.awaiting_bet = True
                            self.selected_bet = 0
                            self.human_turn = False
                            self.show_result = False
                            self.dealer_action_text = ""
                            self.final_results = {}
                            self.dealer_steps = []
                            self.dealer_step_index = 0
                            self.dealer_step_timer = 0
                            self.error_message = ""
                elif event.type == pygame.KEYDOWN and self.human_turn:
                    if event.key == pygame.K_h:
                        self.env.step("HUMAN", 0)  # HIT
                        if self.env.done["HUMAN"]:
                            self.human_turn = False
                    elif event.key == pygame.K_s:
                        self.env.step("HUMAN", 1)  # STAND
                        self.human_turn = False

            self.draw_game_state()

            if not self.awaiting_bet and not self.human_turn and not self.env.done['AI']:
                state = self.env.get_state("AI")
                ai_action = self.agent.choose_action(state, phase='PLAY')
                self.env.step("AI", ai_action)

            if self.env.done['HUMAN'] and self.env.done['AI'] and not self.show_result:
                self.env.dealer_play()
                self.final_results = self.env.evaluate_game()
                self.agent.replay()
                self.show_result = True

            pygame.display.flip()

        pygame.quit()
