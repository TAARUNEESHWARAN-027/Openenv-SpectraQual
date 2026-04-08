from fastapi import FastAPI, HTTPException
import sys
import os

# Add src to path so standard imports work
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, src_path)

from env import SpectraQualEnv
from models import PCBAction, StepResult

app = FastAPI(
    title="SpectraQual OpenEnv API",
    description="REST API for automated OpenEnv space evaluation",
    version="1.0.0"
)

# Initialize a default environment instance
# In a real deployed evaluator, they may instantiate isolated environments
# but for the "ping space URL" test, a global instance is standard.
env_instance = SpectraQualEnv(task_id="task_easy")

@app.get("/")
def health_check():
    """Returns 200 to pass automated ping test."""
    return {"status": "ok", "environment": "SpectraQual"}

@app.post("/reset")
def reset_env() -> StepResult:
    """Reset the environment and return initial observation."""
    try:
        return env_instance.reset()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step")
def step_env(action: PCBAction) -> StepResult:
    """Take a step in the environment."""
    try:
        if env_instance.state()["done"]:
            # If done, returning an error or auto-resetting depends on the logic.
            # Best practice: raise 400 that episode is done.
            raise HTTPException(status_code=400, detail="Episode is done. Call /reset first.")
        return env_instance.step(action)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state")
def get_state():
    """Return the internal state of the environment."""
    try:
        return env_instance.state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
