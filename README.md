---
title: Sst Hackathon 2k26
emoji: 🏢
colorFrom: green
colorTo: purple
sdk: docker
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference




# AdaptiveLearner-v0

AdaptiveLearner-v0 is an adaptive learning path optimization environment designed for personalized education. It is implemented according to the OpenEnv specification and built for reinforcement learning (RL) agents. 

The core goal of this environment is to provide a standardized, compliant platform to train and evaluate AI agents that can generate optimal educational learning paths. The environment simulates a student learning various concepts that have complex prerequisite dependencies.

## Key Features & Overview
An RL Environment to simulate student learning over various educational concepts with prerequisites. The reward design measures learning gains, engagement, fatigue factor, and completion.

### Tasks Available
The environment includes three difficulty levels (as defined in `openenv.yaml`):
- **Easy**: Elementary Arithmetic (5 concepts, max 50 steps)
- **Medium**: Foundation STEM (15 concepts, max 100 steps)
- **Hard**: Full K-12+ Curriculum (35+ concepts, max 150 steps)

### Environment Details
- **Action Space**: `Discrete(N+1)`. Actions `0` through `N-1` teach specific concepts. Action `N` is a NO-OP (student rests, resetting fatigue and engagement). 
- **Observation Space**: `Box` representing continuous metrics. Emits a flattened scalar array containing concept mastery levels, prerequisites met, time invested tracking, historical review timing, student ability, fatigue, and aggregate metrics.
- **Reward Scale**: Every step emits a multi-component mathematical continuous reward normalized between `-1.0` and `1.0`. Scores output in grading evaluate environments between `0.0` and `1.0`.

## Installation Guide

To set up the environment locally, it is highly recommended to use a virtual environment to avoid dependency conflicts. 

**1. Clone the repository:**
```bash
git clone <your-repo-url>
cd adaptive-learner-v0
```

**2. Create and activate a virtual environment:**
```bash
# Create the virtual environment
python -m venv venv

# Activate it (Linux/macOS)
source venv/bin/activate

# Activate it (Windows)
venv\Scripts\activate
```

**3. Install dependencies:**
Install the required packages and install the environment in development (`editable`) mode:
```bash
pip install -r requirements.txt
pip install -e .
```

## Running Baseline

To run the baseline inference script:
```bash
python agents/inference_script.py
```

## Run Tests
To run tests, make sure you installed using pip, then:
```bash
pytest tests/
```
