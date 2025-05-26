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
                print(f"[ERROR] 잘못된 카드 랭크: {rank}")
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
        self.done = {'HUMAN': False, 'AI': False, 'DEALER': False}  # ← 여기 수정
        for player in ['HUMAN', 'AI', 'DEALER']:
            self.hands[player] = [self.deal_card(), self.deal_card()]

    def get_state(self, player):
        return {'value': calculate_hand_value(self.hands[player])}

    def get_hand(self, player):
        return self.hands[player]

    def place_bet(self, player, amount):
        if self.chips[player] >= amount:
            self.bets[player] = amount
            self.chips[player] -= amount
            return True
        else:
            print(f"[WARNING] {player}의 칩이 부족하여 베팅할 수 없습니다.")
            self.bets[player] = 0
            return False  # → 실패를 알려줌

    def step(self, player, action):
        if player == 'AI' and action in [50, 100, 200]:
            self.place_bet('AI', action)
            return

        if action == 0:  # HIT
            self.hands[player].append(self.deal_card())
            if calculate_hand_value(self.hands[player]) > 21:
                self.done[player] = True
        elif action == 1:  # STAND
            self.done[player] = True

    def get_dealer_play_steps(self):
        steps = []
        while calculate_hand_value(self.hands['DEALER']) < 17:
            steps.append(0)  # HIT
        steps.append(1)  # STAND
        return steps

    def dealer_step(self, action):
        if action == 0:
            self.hands['DEALER'].append(self.deal_card())
        elif action == 1:
            self.done['DEALER'] = True

    def dealer_play(self):
        while True:
            value = calculate_hand_value(self.hands["DEALER"])
            
            if value >= 17:
                break
            
            # 딜러가 HIT함 (카드를 실제로 받음)
            new_card = self.deal_card()
            self.hands["DEALER"].append(new_card)


    def evaluate_game(self):
        results = {}
        dealer_score = calculate_hand_value(self.hands['DEALER'])

        for player in ['HUMAN', 'AI']:
            player_score = calculate_hand_value(self.hands[player])
            if player_score > 21:
                if dealer_score > 21:
                    results[player] = 'DRAW'
                    self.chips[player] += self.bets[player]  # 원금만 반환
                else:
                    results[player] = 'LOSE'
            elif dealer_score > 21:
                results[player] = 'WIN'
                self.chips[player] += self.bets[player] * 2
            elif player_score > dealer_score:
                results[player] = 'WIN'
                self.chips[player] += self.bets[player] * 2
            elif player_score == dealer_score:
                results[player] = 'DRAW'
                self.chips[player] += self.bets[player]
            else:
                results[player] = 'LOSE'

        for player in ['HUMAN', 'AI']:
            if self.chips[player] <= 0:
                self.done[player] = True  # 플레이어 게임 오버

        return results
