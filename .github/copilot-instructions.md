# Copilot Instructions for DRL Workspace

## Project Overview
This workspace contains Deep Reinforcement Learning (DRL) experiments and implementations, organized by algorithm:
- `DQN/`: Deep Q-Network implementation in Jupyter notebook (`dqn.ipynb`).
- `REINFORCE/`: REINFORCE algorithm implementations, including a standard PyTorch version (`REINFORCE.ipynb`) and a TorchRL-based version (`REINFORCE_torchrl.ipynb`).

## Architecture & Patterns
- Each algorithm is implemented in a separate notebook, with code, documentation, and experiment results.
- Notebooks use PyTorch, Gym, and TorchRL for environment interaction and model training.
- Data flow: Environment setup → Model definition → Training loop → Evaluation.
- Replay buffers (DQN) use Python `deque` for efficient experience storage.
- REINFORCE notebooks use policy gradient methods, with `Categorical` distributions for action sampling.

## Developer Workflows
- **Run experiments:** Open the relevant notebook and execute cells sequentially.
- **Dependencies:**
  - DQN: `numpy`, `random`, `collections` (deque)
  - REINFORCE: `gym`, `torch`, `torchrl`, `tensordict`
- **Environment setup:**
  - Use `torch.device("cuda" if torch.cuda.is_available() else "cpu")` for device selection.
  - Gym environments are created with `gym.make("CartPole-v1")`.
- **Debugging:**
  - Print statements and cell outputs are used for debugging and monitoring training progress.

## Conventions & Practices
- All code is in Jupyter notebooks; no standalone Python scripts or modules.
- Model architectures are defined inline in notebooks.
- Training loops are explicit and not abstracted into utility functions.
- Use markdown cells for documentation and explanations.
- No custom logging or configuration files; all settings are hardcoded in notebooks.

## Integration Points
- TorchRL is used in `REINFORCE_torchrl.ipynb` for advanced RL abstractions.
- Standard PyTorch and Gym are used in other notebooks.
- No external data sources or APIs; all experiments are self-contained.

## Examples
- **Replay Buffer (DQN):**
  ```python
  from collections import deque
  buffer = deque(maxlen=10000)
  ```
- **Device Selection:**
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ```
- **Environment Creation:**
  ```python
  env = gym.make("CartPole-v1")
  ```

## Key Files & Directories
- `DQN/dqn.ipynb`: DQN implementation and experiments
- `REINFORCE/REINFORCE.ipynb`: REINFORCE (PyTorch) implementation
- `REINFORCE/REINFORCE_torchrl.ipynb`: REINFORCE (TorchRL) implementation

---
**If any conventions or workflows are unclear or missing, please provide feedback for further refinement.**
