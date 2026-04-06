# AdaptiveLearner-v0

Adaptive learning path optimization environment for personalized education, implemented according to the OpenEnv specification.

## Overview
An RL Environment to simulate student learning over various concepts with prerequisites. The reward design measures learning gains, engagement, fatigue factor, and completion.

## Environment Details
- **Action Space**: Discrete(N+1). Actions `0` through `N-1` teach specific concepts. Action `N` is a NO-OP (student rests, resetting fatigue and engagement). 
- **Observation Space**: Box representing continuous metrics. Emits a flattened scalar array containing concept mastery levels, prerequisites met, time invested tracking, historical review timing, student ability, fatigue, and aggregate metrics.
- **Reward Scale**: Every step emits a multi-component mathematical continuous reward normalized between `-1.0` and `1.0`. Scores output in grading evaluate environments between `0.0` and `1.0`.

## Installation

```bash
git clone <your-repo-url>
cd adaptive-learner-v0
pip install -r requirements.txt
pip install -e .
```

## Running Baseline

```bash
python agents/inference_script.py
```

## Run Tests
To run tests, make sure you installed using pip, then:
```bash
pytest tests/
```
