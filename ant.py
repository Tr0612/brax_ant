import mujoco
import mujoco.viewer
import numpy as np
import time

# Load the Ant model
model = mujoco.MjModel.from_xml_path("replicated_robots.xml")  
data = mujoco.MjData(model)

# Launch the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Apply random control inputs
        data.ctrl[:] = np.random.uniform(low=-1.0, high=1.0, size=model.nu)
        
        # Step the simulation
        mujoco.mj_step(model, data)

        # Update the viewer
        viewer.sync()

        
        time.sleep(model.opt.timestep)
