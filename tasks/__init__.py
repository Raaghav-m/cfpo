# Tasks package
from .base import Task
from .gsm8k import GSM8KTask
from .multiple_choice import MultipleChoiceTask, BBHTask, ARCTask, MMLUTask

__all__ = [
    'Task', 
    'GSM8KTask',
    'MultipleChoiceTask',
    'BBHTask',
    'ARCTask', 
    'MMLUTask',
]
