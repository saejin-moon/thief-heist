from env import HeistEnv

env = HeistEnv({"map_size": (11, 11), "guard_count": 0})
obs, _ = env.reset()
print(obs["scout"]["observation"])
print("Mask:", obs["scout"]["action_mask"])