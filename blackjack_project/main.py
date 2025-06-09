from ui.game_gui import BlackjackGUI
from game.blackjack_env import BlackjackEnv
from ai.dqn_agent_expanded import DQNAgent
import torch

def main():
    env = BlackjackEnv()
    agent = DQNAgent()
    
    # ✅ 학습된 통합 모델 불러오기
    agent.load("ai_dqn_trained.pt")
    agent.epsilon = 0.0  # 추론 모드 (탐험 X)

    gui = BlackjackGUI(env, agent)
    gui.run_game()

if __name__ == "__main__":
    main()
