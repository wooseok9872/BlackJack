# train_agent.py
from game.blackjack_env import BlackjackEnv, calculate_hand_value
from ai.dqn_agent_expanded import DQNAgent

EPISODES = 500_000

env = BlackjackEnv()
agent = DQNAgent()

for e in range(EPISODES):
    env.reset()
    env.chips["AI"] = 700

    if e % 5 == 0:
        env.force_hand("AI", [("4", "hearts"), ("2", "clubs")])

    state = env.get_state("AI")
    bet_action = agent.choose_action(state, phase="BETTING", chips=env.chips["AI"], episode=e)
    env.step("AI", bet_action)

    if env.bets['AI'] == 0:
        env.chips["AI"] = 700
        continue

    last_state = None
    last_action = None

    while not env.done["AI"]:
        state = env.get_state("AI")
        action = agent.choose_action(state, phase="PLAY", episode=e)
        env.step("AI", action)
        last_state = state
        last_action = action
        if env.done["AI"]:
            break

    env.dealer_play()
    results = env.evaluate_game()
    result = results["AI"]
    final_score = calculate_hand_value(env.get_hand("AI"))
    bust = final_score > 21

    if bust:
        reward = -3.0
    elif result == "WIN":
        reward = 2.5 if last_action == 0 else 1.5
    elif result == "DRAW":
        reward = 0.5 if last_action == 0 else 0.0
    else:
        reward = -0.5 if last_action == 0 else -1.0
        if last_action == 1 and final_score <= 15:
            reward = -2.0

    if last_action == 0 and not bust:
        reward += 0.5

    if e < 100_000 and last_action == 1 and final_score <= 9:
        reward = -5.0

    if result == "DRAW" and last_action == 1 and final_score <= 15:
        reward = -1.0

    next_state = env.get_state("AI")
    agent.remember(last_state, last_action, reward, next_state, True)
    agent.replay()

    if e % 1000 == 0:
        print(f"[EPISODE {e}] AI Chips: {env.chips['AI']} | Reward: {reward}")
        agent.save("ai_dqn_trained.pt")