# forzadrivingai

![d](https://github.com/null-Exception1/forzadrivingai/blob/main/VID_20250730_23264721-ezgif.com-video-to-gif-converter%20(1).gif)

[Forza Capture] -> [OpenCV Ray-Caster] -> [8-Vector State Space] -> [TorchRL Gym Env] -> [DQN Optimization]

# forzadrivingai

An autonomous driving system for Forza that combines OpenCV geometric feature extraction with a PolicyNet optimization loop built using PyTorch, TorchRL, and Gymnasium.

## Architecture

The system processes video capture frames in real time, converts track boundaries into low-dimensional distance arrays, and determines optimal steering inputs through reinforcement learning.


###  Vision Processing
rather than sending raw image frames directly to the neural network, the agent uses a ray-casting optimization loop to establish spatial tracking context:
* casts 5 linear rays forward at distinct angles to map upcoming track curvature.
* casts 3 rays backward from the asset origin to track longitudinal changes.
* tracks changes in pixel intensity values (`sum(img[j][i])/3`). When pixel values cross the threshold, the tracking loop calculates the edge boundary.
* converts visual boundaries into line coordinates using Euclidean distance calculations (`math.hypot`), outputting an 8-dimensional state vector.

### Reinforcement Learning Environment
The state variables are parsed by a custom Gymnasium environment configured with the following parameters:
* a continuous array tracking positional coordinates relative to the track centerline.
* a discrete configuration managing pathing steps or vehicle steering adjustments.
* tracks target lane alignment metrics and applies a penalty for tracking errors:
  $$Reward = -|Value_{\text{current}} - Value_{\text{target}}|$$
* triggers an automatic environment reset if the vehicle hits the track limits (boundaries capped at absolute values of 0 or 10).

### Model Optimization Framework
The training pipeline uses TorchRL components to manage model parameter updates:
* built using a Multi-Layer Perceptron (MLP) containing two 64-node hidden layers mapped directly to action value nodes.
* uses an E-Greedy exploration module (`EGreedyModule`) that anneals random behaviors over a step threshold.
* uses a decoupled memory configuration (`ReplayBuffer` with `LazyTensorStorage`) to cache and sample historical states for training updates.

---

## Technical Specifications

the model optimization loop relies on the following operational settings:

| Parameter | Operational Value | Functional Context |
| :--- | :--- | :--- |
| `INIT_RAND_STEPS` | 5000 | Number of initial environment frames collected before optimization begins |
| `FRAMES_PER_BATCH` | 100 | Frame collection batch size per sync collection pass |
| `OPTIM_STEPS` | 10 | Training loops run per state sample update |
| `ALPHA` | 0.05 | Optimization learning rate for the Adam optimizer |
| `BUFFER_LEN` | 100,000 | Maximum capacity threshold for the Replay Buffer |
| `REPLAY_BUFFER_SAMPLE` | 128 | Batch size extracted from memory for loss calculations |

---

## Installation and Execution

### Dependencies

install the required packages using the explicit package configurations below:

```bash
pip install torch torchrl tensordict gymnasium opencv-python numpy matplotlib
```

### Module Structure
* **`image_processing`**: manages pixel processing, line tracking loops, and boundary calculations.
  * `place_dot(img, x, y)`: draws visual markers at track checkpoints.
  * `lines(img, x, y)`: casts 8 direction vectors, draws the visual tracking lines, and outputs the distance data.
* **`net.py`**: builds the custom environment, configures the MLP layers, handles exploration loops, and manages the main optimization steps.
