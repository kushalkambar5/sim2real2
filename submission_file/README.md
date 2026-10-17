# Sim2Real Robotics Hackathon — Simulation Round

## Team Information
- **Team Name**: TechStars
- **Team Members**:
  - **Shahid Pathan** | 2nd Year | Electrical & Electronics Engineering (EEE)
  - **Mayank Mehta** | 2nd Year | Mechanical Engineering (MECH)
  - **Kushal Kambar** | 2nd Year | Mechanical Engineering (MECH)

---

## 1. Overview & Architecture

This repository contains the autonomous software solution developed by **TechStars** for the **Sim2Real Robotics Hackathon (Simulation Round)**. The solution enables a differential-drive robot to navigate autonomously inside a $3.0 \times 3.0\,\text{m}$ digital-twin Gazebo arena across three difficulty tiers:

- **Tier 1 (Dead Reckoning)**: Navigates procedurally generated static and dynamic waypoints using dead reckoning based on wheel encoders and IMU sensor fusion.
- **Tier 2 (Perception-Guided)**: Navigates without coordinate topics by detecting ground-plane targets via an overhead camera, tracking dynamic targets at $T+7.0\,\text{s}$, and returning to home.
- **Tier 3 (Cluttered Arena & Obstacle Avoidance)**: Full visual autonomy in a cluttered arena with 4 interior static obstacle walls, dual-inflation occupancy grid mapping, 8-connected A* planning, Line-of-Sight (LOS) path smoothing, A*-based TSP tour optimization, and forward-only path tracking.

### Package Directory Structure

```text
submission_file/
├── README.md
└── src/
    ├── postman_pkg/                  # Organizer simulation package (unmodified)
    └── postman_solution/             # Contestant autonomy package
        ├── package.xml
        ├── setup.py
        ├── setup.cfg
        ├── launch/
        │   └── solution.launch.py    # Unified multi-tier launch file
        └── postman_solution/
            ├── __init__.py
            ├── solution_node.py      # Unified CLI dispatcher
            ├── tier1_node.py         # Tier 1 Dead Reckoning Solution
            ├── tier2_node.py         # Tier 2 Perception-Guided Solution
            └── tier3_node.py         # Tier 3 Obstacle Avoidance Solution
```

---

## 2. Prerequisites & Build Instructions

### System Requirements
- **OS**: Ubuntu 24.04 LTS (Native or WSL2)
- **ROS Version**: ROS 2 Jazzy Jalisco
- **Simulator**: Gazebo Harmonic
- **Dependencies**: `ros-jazzy-ros-gz`, `ros-jazzy-cv-bridge`, `ros-jazzy-tf2-ros`, `python3-opencv`, `python3-numpy`

### Step-by-Step Build Instructions

1. Open a terminal and navigate to the root workspace directory containing `src/`:
   ```bash
   cd /path/to/workspace
   ```
2. Source the ROS 2 underlay:
   ```bash
   source /opt/ros/jazzy/setup.bash
   ```
3. Build the workspace:
   ```bash
   colcon build --symlink-install
   ```
4. Source the built workspace overlay:
   ```bash
   source install/setup.bash
   ```

> **Note**: In every newly opened terminal, always source both ROS 2 and the workspace:
> ```bash
> source /opt/ros/jazzy/setup.bash
> source install/setup.bash
> ```

---

## 3. How to Run (Step-by-Step for Each Tier)

Each tier requires running **three separate terminals**:
1. **Terminal 1**: Simulation & World Launch
2. **Terminal 2**: Automated Evaluator Script
3. **Terminal 3**: Autonomous Solution Node

---

### 🟢 Tier 1: Dead Reckoning Navigation

#### Terminal 1 — Gazebo Simulation
```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch postman_pkg sim.launch.py tier:=1 seed:=42
```
*(Wait until Gazebo and RViz have fully initialized and targets are spawned)*

#### Terminal 2 — Automated Evaluator
```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
python3 src/postman_pkg/scripts/evaluator.py --tier 1
```

#### Terminal 3 — Solution Node
```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch postman_solution solution.launch.py tier:=1
```

---

### 🟡 Tier 2: Perception-Guided Navigation

#### Terminal 1 — Gazebo Simulation
```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch postman_pkg sim.launch.py tier:=2 seed:=42
```
*(Wait until Gazebo and RViz have fully initialized)*

#### Terminal 2 — Automated Evaluator
```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
python3 src/postman_pkg/scripts/evaluator.py --tier 2
```

#### Terminal 3 — Solution Node
```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch postman_solution solution.launch.py tier:=2
```

---

### 🔴 Tier 3: Cluttered Arena & Obstacle Avoidance

#### Terminal 1 — Gazebo Simulation
```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch postman_pkg sim.launch.py tier:=3 seed:=42
```
*(Wait until Gazebo and RViz have fully initialized and obstacle walls are spawned)*

#### Terminal 2 — Automated Evaluator
```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
python3 src/postman_pkg/scripts/evaluator.py --tier 3
```

#### Terminal 3 — Solution Node
```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch postman_solution solution.launch.py tier:=3
```

---

## 4. Deep-Dive: Code Logic & Algorithms by Tier

### 🟢 Tier 1 (`tier1_node.py`): Dead Reckoning Navigation

Tier 1 addresses autonomous navigation across procedurally generated waypoints where waypoint coordinates are published directly via ROS topics (`/waypoints` and `/dynamic_waypoints`), but wheel encoders contain injected Gaussian velocity noise ($\sigma = 0.002\,\text{rad/s}$).

```
/joint_states (noisy) \
                       --> [ Odometry Estimator ] --> /odom & TF (odom -> base_link)
/imu (angular_z)      /              |
                                     v
/waypoints (5 static)  \     [ TSP Route Planner ]
                        -->          |
/dynamic_waypoints (2) /             v
                             [ State Machine ] -----> /cmd_vel (Forward-Only)
```

#### 1. Sensor Fusion & Odometry Estimator
- **Kinematic Formulation**:
  - Wheel radius: $r = 0.0225\,\text{m}$, Wheel track separation: $L = 0.159\,\text{m}$.
  - Encoder linear velocity: $v_{\text{enc}} = \frac{v_R + v_L}{2}$.
  - Encoder yaw rate: $\omega_{\text{enc}} = \frac{v_R - v_L}{L}$.
- **Complementary Yaw Fusion**:
  $$\omega_{\text{fused}} = 0.95 \cdot \omega_{\text{enc}} + 0.05 \cdot \omega_{\text{imu}, z}$$
  This suppresses encoder random walk and wheel-slip bias while rejecting high-frequency IMU gyro noise.
- **Dead Reckoning Integration**:
  At each 20 Hz control tick ($\Delta t \le 0.20\,\text{s}$):
  $$x_{t+\Delta t} = x_t + v_{\text{enc}} \cos(\theta_t) \Delta t$$
  $$y_{t+\Delta t} = y_t + v_{\text{enc}} \sin(\theta_t) \Delta t$$
  $$\theta_{t+\Delta t} = \text{wrap\_to\_pi}(\theta_t + \omega_{\text{fused}} \Delta t)$$
- **TF & Odometry Broadcasting**: Publishes `nav_msgs/msg/Odometry` on `/odom` and broadcasts dynamic transform `odom` $\to$ `base_link`.

#### 2. Waypoint Deduplication & TSP Route Optimization
- Waypoints are deduplicated using spatial quantization keys `(round(x, 4), round(y, 4))`.
- An exhaustive Traveling Salesperson Problem (TSP) permutation solver computes the shortest route visiting all unvisited targets starting from current $(x, y)$ and terminating at home $(0.0, 0.0)$.
- **Dynamic Replanning**: When `/dynamic_waypoints` arrive at $T+5.0\,\text{s}$, the planner marks existing completed waypoints as immutable and computes the optimal TSP path over only the remaining unvisited waypoints.

#### 3. Control Policy & State Machine
- **State Flow**: `WAIT_FOR_WAYPOINTS` $\to$ `PLAN_ROUTE` $\to$ `ALIGN` $\to$ `DRIVE` $\to$ `BRAKE` $\to$ `DWELL` $\to$ `HOME` $\to$ `HOME_DWELL` $\to$ `FINISHED`.
- **Pure In-Place Turn (`ALIGN`)**:
  $v = 0$, $\omega = \text{clamp}(K_{\text{angle}} \cdot e_\theta, -1.5, 1.5)$ with $K_{\text{angle}} = 2.0$, rotating until $|e_\theta| \le 8^\circ$.
- **Forward-Only Cruise (`DRIVE`)**:
  Linear speed is governed by proportional distance gain ($K_{\text{dist}} = 0.8$) bounded by conservative approach caps:
  - $d > 0.50\,\text{m} \implies v \le 0.30\,\text{m/s}$
  - $0.25 < d \le 0.50\,\text{m} \implies v \le 0.15\,\text{m/s}$
  - $0.12 < d \le 0.25\,\text{m} \implies v \le 0.08\,\text{m/s}$
  - Heading error damping: $v$ is halved if $|e_\theta| > 15^\circ$ and zeroed if $|e_\theta| > 30^\circ$.
- **Continuous Dwell Protocol (`DWELL`)**:
  Upon entering capture radius ($d \le 0.20\,\text{m}$) and coming to a complete stop ($|v| \le 0.03\,\text{m/s}$), a continuous timer counts up to $2.30\,\text{s}$ (providing a $0.30\,\text{s}$ safety margin over the evaluator's $2.00\,\text{s}$ threshold).
- **Perimeter Safety Guard**:
  Monitors arena boundaries ($[-0.125, 2.875]\,\text{m}$). If the robot is within $0.10\,\text{m}$ of a perimeter wall, forward velocity is zeroed and the robot turns inward.

---

### 🟡 Tier 2 (`tier2_node.py`): Perception-Guided Navigation

In Tier 2, coordinate topics are suppressed. Waypoint locations must be discovered and tracked visually through the overhead RGB camera.

```
/overhead_camera/image_raw   --> [ HSV Segmenter & Filter ]
                                           |
/overhead_camera/camera_info --> [ Pinhole Ground Projector ]
                                           |
                                 [ Waypoint Tracker (EMA) ]
                                           |
/joint_states + /imu ----------> [ Odometry Estimator ]
                                           |
                                 [ TSP Route Planner ]
                                           |
                                 [ State Machine ] ---------> /cmd_vel
```

#### 1. Visual Target Detection Pipeline (OpenCV)
- Runs at 10 Hz (every 2nd tick of the 20 Hz loop).
- **Color Thresholding**: HSV segmentation for green target disks:
  $$\text{Hue} \in [35, 90], \quad \text{Sat} \in [80, 255], \quad \text{Val} \in [60, 255]$$
- **Morphological Filtering**:
  $5\times 5$ elliptical morphological opening removes camera sensor noise; morphological closing bridges contour gaps.
- **Geometric Contours & Verification**:
  - Area filter: $400 \le \text{Area} \le 3500\,\text{px}^2$.
  - Circularity filter: $C = \frac{4\pi \cdot \text{Area}}{\text{Perimeter}^2} \ge 0.65$.
  - Expected pixel radius verification:
    $$R_{\text{expected}} = \frac{R_{\text{physical}} \cdot f_x}{Z_{\text{cam}}} = \frac{0.15 \cdot f_x}{4.00}$$
    Verifies that the enclosing circle radius $R_{\text{px}} \in [0.50 \cdot R_{\text{expected}}, 1.60 \cdot R_{\text{expected}}]$.
  - Fill ratio: $\frac{\text{Area}}{\pi R_{\text{px}}^2} \ge 0.55$.
  - Centroid: Sub-pixel centroids computed via image spatial moments:
    $$c_u = \frac{m_{10}}{m_{00}}, \quad c_v = \frac{m_{01}}{m_{00}}$$

#### 2. Pinhole Ground-Plane Metric Projection
- With downward camera at $(x_{\text{cam}}, y_{\text{cam}}, z_{\text{cam}}) = (1.50, 1.50, 4.00)\,\text{m}$:
  $$x_c = \frac{u - c_x}{f_x} \cdot z_{\text{cam}}, \quad y_c = \frac{v - c_y}{f_y} \cdot z_{\text{cam}}$$
- Coordinate system alignment:
  $$w_x = x_{\text{cam}} + \text{sign}_v \cdot y_c = 1.50 - y_c$$
  $$w_y = y_{\text{cam}} + \text{sign}_u \cdot x_c = 1.50 - x_c$$

#### 3. Identity Tracking & Temporal Gating (`WaypointTracker`)
- Detections are matched to existing tracks using Euclidean association ($\delta \le 0.12\,\text{m}$).
- Positions are smoothed via Exponential Moving Average (EMA) with $\alpha = 0.30$:
  $$p_{t} = (1 - \alpha) p_{t-1} + \alpha p_{\text{obs}}$$
- **Confirmation Threshold**: A target must be consistently observed across at least 3 frames before being confirmed for route planning.
- **Dynamic Waypoint Handling**:
  The perception pipeline runs continuously during navigation. When the 2 dynamic waypoints appear at $T+7.0\,\text{s}$, they are automatically detected and confirmed. The state machine transitions to `REPLAN`, inserting newly discovered targets into the remaining route while preserving completed targets.

---

### 🔴 Tier 3 (`tier3_node.py`): Cluttered Arena & Obstacle Avoidance

Tier 3 introduces 4 static interior obstacle walls with magenta tops inside the arena, requiring full simultaneous target perception, obstacle segmentation, grid inflation, path planning, and obstacle-aware TSP ordering.

```
/overhead_camera/image_raw   --> [ Green Waypoints & Magenta Walls Detectors ]
                                                 |
                                 [ CameraMapper (Pinhole Extrinsics) ]
                                                 |
                                 [ Obstacle Tracker & Waypoint Tracker ]
                                                 |
                                 [ 2D Dual-Inflation Occupancy Grid ]
                                                 |
                                 [ Pairwise A* TSP Tour Optimizer ]
                                                 |
                                 [ 8-Connected A* + LOS Simplifier ]
                                                 |
                                 [ Forward-Only PathTracker ] ------> /cmd_vel
```

#### 1. Dual Visual Perception
- **Green Waypoint Detector**: Detects 5 initial + 2 dynamic green target disks using the calibrated HSV pipeline.
- **Magenta Obstacle Detector**:
  - HSV segmentation: $\text{Hue} \in [125, 170]$, $\text{Sat} \in [80, 255]$, $\text{Val} \in [60, 255]$.
  - Morphological opening ($3\times 3$) and closing ($7\times 7$).
  - Minimum area rotated bounding box (`cv2.minAreaRect`).
  - Aspect ratio filter ($\ge 2.5$), length verification ($0.25 \le L \le 0.75\,\text{m}$), width verification ($0.02 \le W \le 0.20\,\text{m}$).
- **Temporal Obstacle Tracker (`ObstacleTracker`)**:
  - Maintains a sliding history of 10 detections.
  - Computes median length/width to eliminate perspective flicker.
  - Averages orientation using cyclic yaw vectors:
    $$\bar{\psi} = \frac{1}{2} \text{atan2}\left(\sum \sin(2\psi_i), \sum \cos(2\psi_i)\right)$$
    This prevents $\pm 180^\circ$ and $\pm 90^\circ$ symmetric box orientation flipping.

#### 2. 2D Dual-Inflation Occupancy Grid (`OccupancyGrid`)
- Resolution: $\Delta = 0.03\,\text{m}$ ($104 \times 104$ grid covering $[-0.15, 2.95]\,\text{m}$).
- Obstacle rasterization: Oriented bounding boxes are projected into 2D grid coordinates.
- **Dual-Inflation Architecture**:
  - **Interior Obstacles Inflation**: Inflated by $0.28\,\text{m}$ (robot circumradius $\approx 0.22\,\text{m}$ plus clearance margin).
  - **Perimeter Walls Inflation**: Inflated by only $0.10\,\text{m}$.
  > **Why Dual Inflation?**  
  > If perimeter walls were inflated by $0.28\,\text{m}$, the home position $(0.0, 0.0)$ would be treated as an obstacle, making it mathematically impossible to plan a collision-free return home. Dual inflation preserves home reachability while providing a wide clearance around interior obstacles.

#### 3. Path Planning: 8-Connected A* with Line-of-Sight (LOS) Smoothing
- **8-Connected A* Search**:
  - Uses Octile distance heuristic:
    $$h(\Delta r, \Delta c) = \max(|\Delta r|, |\Delta c|) + (\sqrt{2} - 1) \min(|\Delta r|, |\Delta c|)$$
  - Enforces diagonal corner-cutting checks to ensure diagonal moves do not graze obstacle corners.
  - Nearest free-cell snapping ensures paths can start or end cleanly if numerical drift places the robot slightly into an inflation boundary.
- **Line-of-Sight (LOS) Greedy Simplification**:
  - Ray-casts between non-adjacent waypoints along the A* path.
  - Collapses stair-stepped grid points into clean straight-line segments if the direct line segment has zero collision with inflated cells.

#### 4. A*-Cost TSP Tour Optimization (`order_waypoints_a_star`)
- Standard TSP solvers use Euclidean straight-line distance, which produces severely suboptimal routes in cluttered environments (e.g. attempting to travel through a wall).
- Our solver computes pairwise **actual A* path lengths** between all unvisited targets and home, populating a cost matrix to find the true minimum-time sequence.

#### 5. Forward-Only Path Tracking (`PathTracker`)
- Tracks polygonal path sub-waypoints strictly with forward motion ($0.0 \le v \le 0.30\,\text{m/s}$).
- **Dynamic Corner-Cutting**: Continuously ray-casts forward along remaining path waypoints. If a future waypoint is in direct line-of-sight and collision-free, advances `target_idx` immediately for smooth, arc-free navigation.
- Smooth deceleration when within $0.40\,\text{m}$ and $0.20\,\text{m}$ of the target, stopping at $0.12\,\text{m}$ from target center (well inside the $0.20\,\text{m}$ evaluator capture radius).
- **Replanning Trigger**: If the path becomes blocked or dynamic targets appear, instantly halts motion and replans a safe collision-free path.

---

## 5. Competition Rules & Integrity Compliance

| Rule / Constraint | Organizer Specification | TechStars Implementation | Status |
| :--- | :--- | :--- | :--- |
| **Linear Velocity** | $v \in [0.0, 0.30]\,\text{m/s}$ (No reverse) | Strictly clamped $v \in [0.0, 0.30]\,\text{m/s}$; all maneuvers are forward-only | **PASS** |
| **Angular Velocity** | $\omega \in [-1.50, 1.50]\,\text{rad/s}$ | Strictly clamped $\omega \in [-1.50, 1.50]\,\text{rad/s}$ | **PASS** |
| **Dwell Time** | $\ge 2.00\,\text{s}$ stopped ($|v| \le 0.03\,\text{m/s}$) | Enforces $2.30\,\text{s}$ dwell ($+0.30\,\text{s}$ safety margin) | **PASS** |
| **Capture Radius** | $\le 0.20\,\text{m}$ center-to-center | Targets captured at $\le 0.12\,\text{m}$ from center | **PASS** |
| **Return Home** | Return to $(0.0, 0.0) \le 0.20\,\text{m}$ & dwell $\ge 2.00\,\text{s}$ | Dedicated `HOME` + `HOME_DWELL` states at origin | **PASS** |
| **Dynamic Waypoints** | Collect 2 dynamic targets ($T+5\text{s}$ / $T+7\text{s}$) | Continuous perception & seamless mid-run TSP replanning | **PASS** |
| **Collision Avoidance** | Zero contact with perimeter or obstacles | Dual-inflation grid ($0.28\,\text{m} / 0.10\,\text{m}$) + LOS-checked A* | **PASS** |
| **Anti-Cheating** | No `/evaluator/*` or `/gz/*` topics | Only uses public contestant topics (`/joint_states`, `/imu`, camera) | **PASS** |
| **Arbitrary Seeds** | Works on undisclosed evaluation seeds | Zero hardcoded coordinates; 100% dynamic perception & planning | **PASS** |
| **Package Integrity** | `postman_pkg` must not be modified | All contestant code resides strictly in `postman_solution` | **PASS** |

---

## 6. Key Parameters Reference

| Parameter | Value | Scope | Description |
| :--- | :--- | :--- | :--- |
| `WHEEL_RADIUS` | `0.0225 m` | All Tiers | Driven wheel radius |
| `WHEEL_SEPARATION`| `0.159 m` | All Tiers | Wheel center-to-center gauge |
| `CONTROL_HZ` | `20.0 Hz` | All Tiers | Main control loop frequency |
| `ENCODER_YAW_WEIGHT`| `0.95` | All Tiers | Complementary filter encoder weight |
| `IMU_YAW_WEIGHT` | `0.05` | All Tiers | Complementary filter IMU weight |
| `DWELL_TIME` | `2.30 s` | All Tiers | Dwell duration (organizer minimum is 2.00 s) |
| `STOP_SPEED` | `0.03 m/s` | All Tiers | Upper velocity threshold for stopped state |
| `SAFETY_MARGIN` | `0.28 m` | Tier 3 | Occupancy grid obstacle inflation margin |
| `PERIMETER_MARGIN`| `0.10 m` | Tier 3 | Occupancy grid perimeter wall inflation margin |
| `GRID_RESOLUTION` | `0.03 m` | Tier 3 | Occupancy grid cell size |