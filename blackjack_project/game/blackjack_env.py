# blackjack_env.py
import random

def calculate_hand_value(hand):
    value = 0
    aces = 0
    for rank, suit in hand:
        if rank in ['J', 'Q', 'K']:
            value += 10
        elif rank == 'A':
            value += 11
            aces += 1
        else:
            try:
                value += int(rank)
            except ValueError:
                print(f"[ERROR] Invalid card rank: {rank}")
                value += 0

    while value > 21 and aces:
        value -= 10
        aces -= 1

    return value

class BlackjackEnv:
    def __init__(self):
        self.deck = self.create_deck()
        self.hands = {'HUMAN': [], 'AI': [], 'DEALER': []}
        self.done = {'HUMAN': False, 'AI': False}
        self.chips = {'HUMAN': 700, 'AI': 700}
        self.bets = {'HUMAN': 0, 'AI': 0}
        self.reset()

    def create_deck(self):
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
        values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        deck = [(v, s) for v in values for s in suits]
        random.shuffle(deck)
        return deck

    def deal_card(self):
        if not self.deck:
            self.deck = self.create_deck()
        return self.deck.pop()

    def reset(self):
        self.hands = {'HUMAN': [], 'AI': [], 'DEALER': []}
        self.done = {'HUMAN': False, 'AI': False, 'DEALER': False}
        for player in ['HUMAN', 'AI', 'DEALER']:
            self.hands[player] = [self.deal_card(), self.deal_card()]

    def force_hand(self, player, cards):
        self.hands[player] = cards

    def get_state(self, player):
        player_hand = self.hands[player]
        dealer_visible = self.hands['DEALER'][0]
        player_score = calculate_hand_value(player_hand)
        dealer_score = calculate_hand_value([dealer_visible])
        usable_ace = any(card[0] == 'A' for card in player_hand) and player_score + 10 <= 21
        is_soft = int(usable_ace)
        safe_cards = max(0, 21 - player_score)
        bust_risk = 1.0 - min(safe_cards / 13.0, 1.0)

        return {
            'player_value': player_score,
            'dealer_value': dealer_score,
            'usable_ace': int(usable_ace),
            'is_soft_hand': is_soft,
            'bust_risk': bust_risk
        }

    def get_hand(self, player):
        return self.hands[player]

    def place_bet(self, player, amount):
        if self.chips[player] >= amount:
            self.bets[player] = amount
            self.chips[player] -= amount
            return True
        else:
            self.bets[player] = 0
            return False

    def step(self, player, action):
        if player == 'AI' and action in [50, 100, 200]:
            self.place_bet('AI', action)
            return

        if action == 0:
            self.hands[player].append(self.deal_card())
            if calculate_hand_value(self.hands[player]) > 21:
                self.done[player] = True
        elif action == 1:
            self.done[player] = True

    def dealer_play(self):
        while calculate_hand_value(self.hands["DEALER"]) < 17:
            self.hands["DEALER"].append(self.deal_card())

    def evaluate_game(self):
        results = {}
        dealer_score = calculate_hand_value(self.hands['DEALER'])

        for player in ['HUMAN', 'AI']:
            player_score = calculate_hand_value(self.hands[player])
            if player_score > 21:
                if dealer_score > 21:
                    results[player] = 'DRAW'
                    self.chips[player] += self.bets[player]
                else:
                    results[player] = 'LOSE'
            elif dealer_score > 21 or player_score > dealer_score:
                results[player] = 'WIN'
                self.chips[player] += self.bets[player] * 2
            elif player_score == dealer_score:
                results[player] = 'DRAW'
                self.chips[player] += self.bets[player]
            else:
                results[player] = 'LOSE'

        for player in ['HUMAN', 'AI']:
            if self.chips[player] <= 0:
                self.done[player] = True

        return results
