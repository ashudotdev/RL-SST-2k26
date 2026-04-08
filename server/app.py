import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import Any, Dict, Optional

from tasks.easy import make_easy_task

app = FastAPI(title="AdaptiveLearner-v0 Space API")

# Keep a global instance serving the default task
# For more advanced deployments, state management per session is recommended,
# but for the hackathon baseline checker, a global singleton handles the validation ping.
global_env = None

class ResetRequest(BaseModel):
    model_config = ConfigDict(extra='allow')
    seed: Optional[int] = None

class StepRequest(BaseModel):
    model_config = ConfigDict(extra='allow')
    action: int
    # Allow extra fields

@app.get("/")
def read_root():
    return {"status": "ok", "environment": "AdaptiveLearner-v0"}

@app.get("/reset")
@app.post("/reset")
def reset_endpoint(req: ResetRequest = None):
    global global_env
    if global_env is None:
        global_env = make_easy_task()
    
    seed = req.seed if req else None
    
    obs, info = global_env.reset(seed=seed)
    
    # Needs to match subset of openenv.yaml reset_signature returns
    # openenv.yaml says: keys: [mastery_levels, prerequisites_met, time_invested]
    state_dict = global_env.state()
    return {
        "mastery_levels": state_dict["mastery_levels"].tolist(),
        "prerequisites_met": state_dict["prerequisites_met"].tolist(),
        "time_invested": state_dict["time_invested"].tolist()
    }

@app.post("/step")
def step_endpoint(req: StepRequest):
    global global_env
    if global_env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized.")
    
    try:
        obs, reward, done, truncated, info = global_env.step(req.action)
        return {
            "observation": obs.tolist(),
            "reward": float(reward),
            "done": done,
            "truncated": truncated,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/state")
def state_endpoint_post():
    global global_env
    if global_env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized.")
    state_dict = global_env.state()
    
    # Convert numpy arrays to lists for JSON serialization
    safe_state = {}
    for k, v in state_dict.items():
        if hasattr(v, "tolist"):
            safe_state[k] = v.tolist()
        else:
            safe_state[k] = v
            
    return safe_state

@app.get("/state")
def state_endpoint_get():
    return state_endpoint_post()

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
