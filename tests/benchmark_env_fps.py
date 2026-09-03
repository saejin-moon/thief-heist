import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("rust/python"))

from vec_env_rust import RustVectorEnv

from constants import AGENTS
from vec_env import VectorEnv as PyVectorEnv


def benchmark(env_class, name, num_envs=16, num_steps=500):
    env = env_class(num_envs=num_envs)
    _obs, _state = env.reset(seed=42)
    actions = {a: np.zeros(num_envs, dtype=np.int32) for a in AGENTS}

    # Warmup
    for _ in range(20):
        env.step(actions)

    t0 = time.perf_counter()
    for _ in range(num_steps):
        # Generate random actions
        acts = {
            a: np.random.randint(0, 6, size=num_envs, dtype=np.int32) for a in AGENTS
        }
        env.step(acts)
    t1 = time.perf_counter()

    env.close()
    elapsed = t1 - t0
    total_steps = num_steps * num_envs
    fps = total_steps / elapsed
    print(
        f"[{name}] {num_envs} Envs | {num_steps} Steps ({total_steps} env-steps) in {elapsed:.3f}s -> {fps:,.1f} FPS (Steps/Sec)"
    )
    return fps


if __name__ == "__main__":
    print("=" * 70)
    print("HEIST ENVIRONMENT THROUGHPUT BENCHMARK (Python vs Native Rust)")
    print("=" * 70)
    fps_py = benchmark(
        PyVectorEnv, "Python Multiprocessing VectorEnv", num_envs=16, num_steps=500
    )
    fps_rs = benchmark(
        RustVectorEnv, "Native Rust Rayon VectorEnv", num_envs=16, num_steps=500
    )
    speedup = fps_rs / fps_py
    print("=" * 70)
    print(f"SPEEDUP: Native Rust is {speedup:.1f}x FASTER than Python Multiprocessing!")
    print("=" * 70)
