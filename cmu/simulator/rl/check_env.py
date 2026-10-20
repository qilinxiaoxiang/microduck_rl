from stable_baselines3.common.env_checker import check_env
from jaw_env import MicroDuckJawEnv

if __name__ == "__main__":
    env = MicroDuckJawEnv(randomize=False)
    try:
        check_env(env, warn=True)
        obs, info = env.reset(seed=92610)
        total = 0
        while True:
            result = env.rpc(op="step", action=0, scripted=True)
            total += result["reward"]
            if result["terminated"]:
                assert result["info"]["acceptance"]["accepted"], result
                print({"check_env":"passed", "scripted":result["info"], "reward":total}, flush=True)
                break
    finally:
        env.close()
