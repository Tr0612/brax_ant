import time
import jax
import numpy as np
import mujoco
import mujoco.viewer
import functools
from brax import envs
from brax.io import model as brax_model
from brax.training.agents.ppo import train as ppo
import imageio



env = envs.get_environment(env_name='ant', backend='generalized')



make_inference_fn ,params,_ = ppo.train(environment=env,num_timesteps=0,episode_length=0)


params = brax_model.load_params('ant_policy_params')
inference_fn = make_inference_fn(params, deterministic=False)
jit_inference_fn = jax.jit(inference_fn)


mj_model = mujoco.MjModel.from_xml_path("ant.xml")
mj_data = mujoco.MjData(mj_model)
rng = jax.random.PRNGKey(0)
renderer = mujoco.Renderer(mj_model)
frames = []


with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    while viewer.is_running():
        # Create observation vector: [qpos + qvel]
        # mj_data.qvel[:] = np.random.uniform(low=-0.1, high=0.1, size=mj_data.qvel.shape)
        obs = np.concatenate([mj_data.qpos[2:], mj_data.qvel]).astype(np.float32)
        print("qpos[2:]:", np.round(mj_data.qpos[2:], 3))
        print("qvel:", np.round(mj_data.qvel, 3))
        # print(obs)
        rng, subkey = jax.random.split(rng)
        action,_ = jit_inference_fn(obs, subkey)
        print("PPO action:", np.round(action, 3))
        mj_data.ctrl[:] = np.asarray(action)
        # mj_data.ctrl[:] = np.array([1.0] * 8)
        mujoco.mj_step(mj_model, mj_data)
        renderer.update_scene(mj_data)
        frames.append(np.copy(renderer.render()))
        viewer.sync()
        # time.sleep(mj_model.opt.timestep)
# renderer.free()
imageio.mimsave("ant_sim.gif",frames,fps=30)