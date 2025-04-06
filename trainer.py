import os
import jax
import functools
import matplotlib.pyplot as plt
from datetime import datetime
import mujoco
import mujoco.viewer
import numpy as np
import joblib
from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.io import model  
from brax.io import html

# # Optional TPU setup
# if 'COLAB_TPU_ADDR' in os.environ:
#     from jax.tools import colab_tpu
#     colab_tpu.setup_tpu()

# --- Configuration ---
env_name = 'ant'
backend = 'generalized'
output_path = './ant_policy_params'

# Create the environment
env = envs.get_environment(env_name=env_name, backend=backend)

# Set up training function for Ant
train_fn = functools.partial(
    ppo.train,
    num_timesteps=200_000,           
    num_evals=3,                     
    reward_scaling=10,
    episode_length=500,              
    normalize_observations=True,
    action_repeat=1,
    unroll_length=5,
    num_minibatches=4,              
    num_updates_per_batch=1,
    discounting=0.97,
    learning_rate=3e-4,
    entropy_cost=1e-2,
    num_envs=32,                     
    batch_size=128,
    seed=1,
)

# Store training progress
xdata, ydata = [], []
times = [datetime.now()]

def progress(num_steps, metrics):
    times.append(datetime.now())
    xdata.append(num_steps)
    ydata.append(metrics['eval/episode_reward'])
    print(f"Step: {num_steps:,} \t Reward: {metrics['eval/episode_reward']:.2f}")

# Train the policy
print("Training started...")
make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)
print("Training completed.")

# Save trained parameters
model.save_params(output_path, params)
print(f"Policy parameters saved to {output_path}")

# Save plot
plt.figure(figsize=(8, 4))
plt.plot(xdata, ydata, label='Episode Reward')
plt.xlabel("Environment Steps")
plt.ylabel("Reward")
plt.title("Training Progress - Brax Ant PPO")
plt.grid()
plt.tight_layout()
plt.savefig("ant_training_curve.png")
print("Training curve saved to ant_training_curve.png")

# Timing info
print(f'Time to JIT compile: {times[1] - times[0]}')
print(f'Time to train: {times[-1] - times[1]}')


params = model.load_params('./ant_policy_params')
inference_fn = make_inference_fn(params,deterministic=False)

# Evaluate the policy in the environment
env_name = 'ant'
backend = 'generalized'

env = envs.get_environment(env_name=env_name, backend=backend)
# params = model.load_params('./ant_policy_params')
# inference_fn = make_ppo_inference_fn(env.observation_size, env.action_size)
# inference = inference_fn(params)

# --- JIT-compile functions ---
jit_env_reset = jax.jit(env.reset)
jit_env_step = jax.jit(env.step)
jit_inference_fn = jax.jit(inference_fn)

# --- Rollout ---
rollout = []
rng = jax.random.PRNGKey(1)
state = jit_env_reset(rng=rng)

for _ in range(1000):
    rollout.append(state)  
    act_rng,rng = jax.random.split(rng)
    action,_ = jit_inference_fn(state.obs, act_rng)
    state = jit_env_step(state, action)
    

# print(type(rollout[0]))  
# print(dir(rollout[0]))
# print(rollout[0].pipeline_state)

# html_ouptut = html.render(env.sys,rollout)

# with open('ant_rollout.html','w') as f:
#     f.write(html_ouptut)
    
print(f"Rollout complete — {len(rollout)} frames collected.")





