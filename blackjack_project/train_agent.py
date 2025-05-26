from game.blackjack_env import BlackjackEnv, calculate_hand_value
from ai.dqn_agent_expanded import DQNAgent
import torch

EPISODES = 5000
agent = DQNAgent()

for episode in range(EPISODES):
    env = BlackjackEnv()
    env.place_bet("HUMAN", 100)  # HUMAN dummy bet

    # === AI BETTING PHASE ===
    state = env.get_state("AI")
    bet_action = agent.choose_action(state, phase="BETTING")
    env.step("AI", bet_action)
    next_state = env.get_state("AI")
    agent.store(state, bet_action, 0, next_state, False, phase="BETTING")
    bet_index = len(agent.memory_bet) - 1

    # === AI PLAY PHASE ===
    done = False
    while not env.done["AI"]:
        state = env.get_state("AI")
        action = agent.choose_action(state, phase="PLAY")
        env.step("AI", action)
        done = env.done["AI"]
        next_state = env.get_state("AI")
        agent.store(state, action, 0, next_state, done, phase="PLAY")

    # === DEALER & RESULT ===
    env.dealer_play()
    results = env.evaluate_game()
    result = results["AI"]

    # === ASSIGN REWARD ===
    reward = 1 if result == "WIN" else -1 if result == "LOSE" else 0
    agent.memory_bet[bet_index] = (*agent.memory_bet[bet_index][0:2], reward, *agent.memory_bet[bet_index][3:])

    # === TRAIN ===
    agent.replay()
    agent.update_target_model()

    # === LOG ===
    if (episode + 1) % 100 == 0:
        print(f"[Episode {episode + 1}/{EPISODES}] AI result: {result:>4} | Epsilon: {agent.epsilon:.3f} | Chips: {env.chips['AI']}")

# === SAVE MODELS ===
torch.save(agent.model_bet.state_dict(), "model_bet.pt")
torch.save(agent.model_play.state_dict(), "model_play.pt")
print("✅ 모델 저장 완료: model_bet.pt, model_play.pt")
