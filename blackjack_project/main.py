from ui.game_gui import BlackjackGUI
from game.blackjack_env import BlackjackEnv
from ai.dqn_agent_expanded import DQNAgent
import torch


def main():
    env = BlackjackEnv()
    agent = DQNAgent()

    # ✅ 학습된 모델 불러오기
    agent.model_bet.load_state_dict(torch.load("model_bet.pt"))
    agent.model_play.load_state_dict(torch.load("model_play.pt"))
    agent.epsilon = 0.0  # 학습된 정책만 사용

    gui = BlackjackGUI(env, agent)
    gui.run_game()


if __name__ == "__main__":
    main()