"""
Keeping track of my curves

by Stéphane Vujasinovic
"""

# - IMPORTS ---
from dataclasses import dataclass, field
from typing import Dict, List

# - CLASSES ---
@dataclass
class SequenceLevelCurveRecorder:
    pass



@dataclass
class FrameLevelCurveRecorder:
    call_user_inter_for_obx: List = field(default_factory=list)
    call_pseudo_inter_for_obx: List = field(default_factory=list)
    entropy_memory_frame: Dict = field(default_factory=dict)
    entropy_memory_frame_map: Dict = field(default_factory=dict)
    

@dataclass
class UpdateMemoryFlag:
    flag: bool = True
