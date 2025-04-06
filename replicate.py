def rename_all_elements(element, suffix):
    """Recursively appends a suffix to all name attributes in an XML subtree."""
    if 'name' in element.attrib:
        element.attrib['name'] += f"_{suffix}"
    for child in element:
        rename_all_elements(child, suffix)

def replicate_and_simulate_robots(
    xml_path: str,
    num_envs: int,
    env_separation: float,
    envs_per_row: int
):
    import xml.etree.ElementTree as ET
    import numpy as np
    import mujoco
    import mujoco.viewer
    import time

    # Load and parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    worldbody = root.find('worldbody')
    original_robot = worldbody.find('body')
    actuator_root = root.find('actuator')
    original_actuators = list(actuator_root)

    if original_robot is None:
        raise ValueError("No <body> found in <worldbody>")

    for i in range(1, num_envs):
        row = i // envs_per_row
        col = i % envs_per_row

        x_offset = col * env_separation
        y_offset = row * env_separation

        # Deep copy the original body
        new_robot = ET.fromstring(ET.tostring(original_robot))
        new_robot.set('pos', f"{x_offset} {y_offset} 0")
        new_robot.set('name', f"robot_{i}")
        rename_all_elements(new_robot, i)  

        worldbody.append(new_robot)
        for act in original_actuators:
            new_act = ET.fromstring(ET.tostring(act))
            if 'joint' in new_act.attrib:
                new_act.attrib['joint'] += f"_{i}"  
            if 'name' in new_act.attrib:
                new_act.attrib['name'] += f"_{i}"
            actuator_root.append(new_act)

    # Save the new XML
    new_xml_path = "replicated_ant.xml"
    tree.write(new_xml_path)
    print(f"✅ Replicated model saved to: {new_xml_path}")

    # Load and simulate
    model = mujoco.MjModel.from_xml_path(new_xml_path)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            data.ctrl[:] = np.random.uniform(-1.0, 1.0, size=model.nu)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


replicate_and_simulate_robots(
    xml_path='ant.xml',
    num_envs=5,
    env_separation=3.0,
    envs_per_row=3
)