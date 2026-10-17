# Sim2Real Submission

## Team
Team name - TechStars

Shahid Pathan | 2nd year | EEE
Mayank Mehta | 2nd year | MECH
Kushal Kambar | 2nd year | MECH






### **How to Run (Step-by-Step)**

Always open **three separate terminals** and run `source install/setup.bash` in **every single one**:

#### **Terminal 1 — Launch Gazebo Simulation**

```
bash
cd /mnt/e/Codes/sim2real/sim2real_simulation_files
source /opt/ros/jazzy/setup.bash
source install/setup.bash
# Tier 1 (clean arena, no obstacles):
ros2 launch postman_pkg sim.launch.py tier:=1 seed:=42
```

*(Wait until Gazebo and RViz open and waypoints are spawned)*

#### **Terminal 2 — Run the Evaluator**

```
bash
cd /mnt/e/Codes/sim2real/sim2real_simulation_files
source /opt/ros/jazzy/setup.bash
source install/setup.bash
python3 src/postman_pkg/scripts/evaluator.py--tier1
```

#### **Terminal 3 — Launch the Solution Node**

```
bash
cd /mnt/e/Codes/sim2real/sim2real_simulation_files
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch postman_solution solution.launch.py tier:=1
```