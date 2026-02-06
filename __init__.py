"""
<<<<<<< HEAD
__paper__ = Beyond Prompt Content: Enhancing LLM Performance via Content-Format Integrated Prompt Optimization (CFPO)
__author__ = Yuanye Liu#, Jiahang Xu#
__version__ = 1.0
__date__ = 2025-1
"""
=======
CFPO Simple - Content Format Prompt Optimization

A simplified, runnable version of the CFPO framework.
"""

__version__ = "1.0.0"
__author__ = "CFPO Team"

from .optimizer import Optimizer
from .prompts import Prompt, PromptHistory
from .models import LLMModel, OllamaModel, HuggingFaceModel
from .tasks import Task, GSM8KTask
from .mutators import Mutator, CaseDiagnosisMutator, MonteCarloMutator, FormatMutator

__all__ = [
    'Optimizer',
    'Prompt',
    'PromptHistory', 
    'LLMModel',
    'OllamaModel',
    'HuggingFaceModel',
    'Task',
    'GSM8KTask',
    'Mutator',
    'CaseDiagnosisMutator',
    'MonteCarloMutator',
    'FormatMutator',
]
>>>>>>> 8122d50 (Initial commit)
