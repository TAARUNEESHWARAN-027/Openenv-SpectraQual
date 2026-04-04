"""
PCB Defect Triage Models - OpenEnv Compliant

Defines typed Observation, Action, and Reward models for the PCB environment.
"""

from pydantic import BaseModel
from typing import Literal


class PCBObservation(BaseModel):
    """
    Observation state for a PCB board in the triage queue.
    
    Attributes:
        defect_type: Category of detected defect
        criticality_score: Severity of defect (0-1)
        component_cost: Cost of affected component ($)
        inspection_confidence: Confidence of inspection result (0-1)
        queue_length: Current queue size
        available_slots: Free repair slots
    """
    defect_type: str
    criticality_score: float
    component_cost: float
    inspection_confidence: float
    queue_length: int
    available_slots: int

    class Config:
        json_schema_extra = {
            "example": {
                "defect_type": "solder_bridge",
                "criticality_score": 0.85,
                "component_cost": 25.50,
                "inspection_confidence": 0.92,
                "queue_length": 3,
                "available_slots": 2,
            }
        }


class PCBAction(BaseModel):
    """
    Action to take for PCB triage.
    
    Attributes:
        action: Triage decision
            - PASS: Board is acceptable, ship to customer
            - SCRAP: Board is irreparable, discard
            - ROUTE_COMPONENT_REPLACEMENT: Send for component replacement
            - ROUTE_SOLDERING: Send for solder rework
            - ROUTE_DIAGNOSTICS: Send for advanced diagnostics
            - WAIT: Hold board in queue (penalty applied)
    """
    action: Literal[
        "PASS",
        "SCRAP",
        "ROUTE_COMPONENT_REPLACEMENT",
        "ROUTE_SOLDERING",
        "ROUTE_DIAGNOSTICS",
        "WAIT",
    ]

    class Config:
        json_schema_extra = {
            "example": {
                "action": "ROUTE_SOLDERING"
            }
        }


class PCBReward(BaseModel):
    """
    Reward signal for the taken action.
    
    Attributes:
        reward: Scalar reward value (typically -1.0 to 1.0)
    """
    reward: float

    class Config:
        json_schema_extra = {
            "example": {
                "reward": 1.0
            }
        }
