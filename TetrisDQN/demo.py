"""
demo.py – Visual demo of the Tetris environment with a random agent.
Close the window or press Ctrl+C to exit.
"""
import random
from environment import make_env

env = make_env(render_mode='human')
obs, _ = env.reset()
done = False

# Bias action sampling: avoid DROP (4) so pieces fall slowly and are visible
ACTIONS = [0, 1, 2, 3, 5]   # LEFT RIGHT ROT_CW ROT_CCW NO_OP

while not done:
    action = random.choice(ACTIONS)
    obs, reward, done, _, info = env.step(action)
    env.render()

print(f"Game over!  Score: {info['score']}  "
      f"Lines: {info['lines_cleared']}  Steps: {info['steps']}")
env.close()
