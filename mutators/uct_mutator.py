"""
UCT Mutator - Upper Confidence Bound for Trees

Implements UCT algorithm for intelligent format selection.
Now with:
- Two-level format system (Prompt Renderer + Query Format)
- LLM-guided format generation
- Format extractors for each renderer
"""

import math
import random
import logging
from typing import List, Dict, Optional, Tuple, Any, Callable
from copy import deepcopy
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base import Mutator
from prompts import Prompt
from models import LLMModel
from tasks import Task

# Lazy imports to avoid circular dependency
# These will be imported on first use
_format_pool_imported = False
PROMPT_RENDERERS = None
PROMPT_RENDERER_DESCRIPTIONS = None
QUERY_FORMATS = None
QUERY_FORMAT_DESCRIPTIONS = None
QA_QUERY_FORMATS = None
MC_QUERY_FORMATS = None
LLMFormatGenerator = None
GeneratedFormat = None

def _ensure_format_pool_imports():
    """Lazy import of format search pool to avoid circular imports."""
    global _format_pool_imported
    global PROMPT_RENDERERS, PROMPT_RENDERER_DESCRIPTIONS
    global QUERY_FORMATS, QUERY_FORMAT_DESCRIPTIONS, QA_QUERY_FORMATS, MC_QUERY_FORMATS
    global LLMFormatGenerator, GeneratedFormat
    
    if _format_pool_imported:
        return
    
    from .format_search_pool.prompt_renderer import (
        PROMPT_RENDERERS as _PR,
        PROMPT_RENDERER_DESCRIPTIONS as _PRD,
    )
    from .format_search_pool.query_format import (
        QUERY_FORMATS as _QF,
        QUERY_FORMAT_DESCRIPTIONS as _QFD,
        QA_QUERY_FORMATS as _QAQF,
        MC_QUERY_FORMATS as _MCQF,
        get_formats_for_task as _get_formats_for_task,
    )
    from .format_search_pool.format_generator import (
        LLMFormatGenerator as _LLMFG,
        GeneratedFormat as _GF,
    )
    
    PROMPT_RENDERERS = _PR
    PROMPT_RENDERER_DESCRIPTIONS = _PRD
    QUERY_FORMATS = _QF
    QUERY_FORMAT_DESCRIPTIONS = _QFD
    QA_QUERY_FORMATS = _QAQF
    MC_QUERY_FORMATS = _MCQF
    LLMFormatGenerator = _LLMFG
    GeneratedFormat = _GF
    
    _format_pool_imported = True


def get_prompt_renderer(name):
    """Get a prompt renderer by name."""
    _ensure_format_pool_imports()
    from .format_search_pool.prompt_renderer import get_prompt_renderer as _get
    return _get(name)


def get_query_format(name, task_type='qa'):
    """Get a query format by name."""
    _ensure_format_pool_imports()
    from .format_search_pool.query_format import get_query_format as _get
    return _get(name, task_type)


def get_formats_for_task(task_type):
    """Get formats appropriate for a task type."""
    _ensure_format_pool_imports()
    from .format_search_pool.query_format import get_formats_for_task as _get
    return _get(task_type)


def _log_to_file_and_console(message: str):
    """Helper to log to both console and the global log file."""
    print(message)
    try:
        from optimizer import get_global_logger
        dlog = get_global_logger()
        if dlog:
            dlog.log(message, also_print=False)
    except:
        pass


class UCTNode:
    """Node in the UCT tree representing a format choice."""
    
    def __init__(self, name: str, parent=None):
        self.name = name
        self.parent = parent
        self.visits = 0
        self.total_reward = 0.0
        self.children: Dict[str, 'UCTNode'] = {}
    
    @property
    def average_reward(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits
    
    def ucb1_score(self, exploration_constant: float = 1.414) -> float:
        """Calculate UCB1 score for this node."""
        if self.visits == 0:
            return float('inf')
        if self.parent is None or self.parent.visits == 0:
            return self.average_reward
        exploration_term = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return self.average_reward + exploration_term
    
    def add_child(self, name: str) -> 'UCTNode':
        if name not in self.children:
            self.children[name] = UCTNode(name, parent=self)
        return self.children[name]
    
    def update(self, reward: float) -> None:
        self.visits += 1
        self.total_reward += reward
        if self.parent:
            self.parent.update(reward)


# ============================================================================
# LEGACY FORMAT POOL (for backward compatibility)
# ============================================================================

def _render_markdown(p):
    return PROMPT_RENDERERS['markdown'][0](p)

def _render_xml(p):
    return PROMPT_RENDERERS['xml'][0](p)

def _render_plain(p):
    return PROMPT_RENDERERS['plain'][0](p)

def _render_structured(p):
    return PROMPT_RENDERERS['structured'][0](p)

def _render_conversational(p):
    return PROMPT_RENDERERS['conversational'][0](p)

def _render_numbered(p):
    return PROMPT_RENDERERS['numbered'][0](p)

def _render_academic(p):
    return PROMPT_RENDERERS['academic'][0](p)

def _render_chain_of_thought(p):
    return PROMPT_RENDERERS['chain_of_thought'][0](p)

# Additional legacy formats
def _render_bullet(p):
    return (
        "* Task: " + p.task_instruction + "\n\n"
        "* Details:\n" + p.task_detail + "\n\n"
        "* Output Format:\n" + p.output_format + "\n\n"
        "* Examples:\n" + p.examples + "\n\n"
        "* Your Turn:\n" + p.cot_hinter
    )

def _render_socratic(p):
    return (
        "What is the problem?\n" + p.task_instruction + "\n\n"
        "What approach should we take?\n" + p.task_detail + "\n\n"
        "How should we format the answer?\n" + p.output_format + "\n\n"
        "Can you show me examples?\n" + p.examples + "\n\n"
        "Now, can you solve this?\n" + p.cot_hinter
    )

def _render_concise(p):
    return (
        "Task: " + p.task_instruction + "\n"
        "Instructions: " + p.task_detail + "\n"
        "Format: " + p.output_format + "\n"
        "Examples: " + p.examples + "\n" +
        p.cot_hinter
    )

def _render_emphasized(p):
    return (
        "**TASK:** " + p.task_instruction + "\n\n"
        "**IMPORTANT DETAILS:** " + p.task_detail + "\n\n"
        "**REQUIRED FORMAT:** " + p.output_format + "\n\n"
        "**EXAMPLES:**\n" + p.examples + "\n\n"
        "**YOUR RESPONSE:**\n" + p.cot_hinter
    )

def _render_dialogue(p):
    return (
        "User: " + p.task_instruction + "\n\n"
        "Assistant: I understand. Let me help you with that.\n\n" +
        p.task_detail + "\n\n"
        "The expected format is: " + p.output_format + "\n\n"
        "Here are some examples:\n" + p.examples + "\n\n" +
        p.cot_hinter
    )

def _render_step_by_step(p):
    return (
        "Follow these steps to solve the problem:\n\n"
        "Step 1: Understand the task\n" + p.task_instruction + "\n\n"
        "Step 2: Apply the method\n" + p.task_detail + "\n\n"
        "Step 3: Format your answer\n" + p.output_format + "\n\n"
        "Step 4: Learn from examples\n" + p.examples + "\n\n"
        "Step 5: Solve\n" + p.cot_hinter
    )

def _render_minimal(p):
    return (
        p.task_instruction + " " + p.task_detail + "\n\n" +
        p.examples + "\n\n" +
        p.cot_hinter
    )


# Legacy format pool for backward compatibility
FORMAT_POOL = {
    'markdown': _render_markdown,
    'xml': _render_xml,
    'plain': _render_plain,
    'structured': _render_structured,
    'conversational': _render_conversational,
    'numbered': _render_numbered,
    'bullet': _render_bullet,
    'academic': _render_academic,
    'socratic': _render_socratic,
    'chain_of_thought': _render_chain_of_thought,
    'concise': _render_concise,
    'emphasized': _render_emphasized,
    'dialogue': _render_dialogue,
    'step_by_step': _render_step_by_step,
    'minimal': _render_minimal,
}

FORMAT_DESCRIPTIONS = {
    'markdown': 'Uses markdown headers and formatting',
    'xml': 'Uses XML-style tags for structure',
    'plain': 'Simple plain text with line breaks',
    'structured': 'Uses clear section markers (===)',
    'conversational': 'Natural conversational tone',
    'numbered': 'Numbered list format',
    'bullet': 'Bullet point format',
    'academic': 'Formal academic structure',
    'socratic': 'Question-driven format',
    'chain_of_thought': 'Emphasizes step-by-step reasoning',
    'concise': 'Minimal text, direct approach',
    'emphasized': 'Uses emphasis markers (**)',
    'dialogue': 'User-assistant dialogue format',
    'step_by_step': 'Explicit step sequence',
    'minimal': 'Extremely condensed',
}


class UCTMutator(Mutator):
    """
    Mutator using UCT algorithm for intelligent format selection.
    
    Features:
    - Two-level format system (Prompt Renderer + Query Format)
    - UCT-based selection balancing exploration/exploitation
    - LLM-guided format generation
    - Format extractors for response parsing
    """
    
    COMPONENT_KEYS = ['PROMPT_RENDERER', 'QUERY_FORMAT']
    
    def __init__(
        self,
        llm: LLMModel,
        task: Task,
        logger: Optional[logging.Logger] = None,
        exploration_constant: float = 1.414,
        enable_llm_generation: bool = True,
    ):
        super().__init__(llm, task, self.COMPONENT_KEYS, logger)
        self.exploration_constant = exploration_constant
        self.enable_llm_generation = enable_llm_generation
        
        # Ensure format pool is imported
        _ensure_format_pool_imports()
        
        # Determine task type
        task_name = task.__class__.__name__.lower()
        self.task_type = 'multiple_choice' if any(x in task_name for x in ['multiplechoice', 'bbh', 'arc', 'mmlu']) else 'qa'
        
        # Initialize prompt renderer pool with UCT nodes
        self.prompt_renderer_root = UCTNode('prompt_renderer_root')
        self.prompt_renderers: Dict[str, Tuple[Callable, Callable]] = dict(PROMPT_RENDERERS)
        for name in self.prompt_renderers:
            self.prompt_renderer_root.add_child(name)
        
        # Initialize query format pool with UCT nodes
        self.query_format_root = UCTNode('query_format_root')
        self.query_formats: Dict[str, Tuple[Callable, Callable]] = get_formats_for_task(self.task_type)
        for name in self.query_formats:
            self.query_format_root.add_child(name)
        
        # LLM format generator
        self.format_generator = LLMFormatGenerator(llm, task, logger) if enable_llm_generation else None
        
        # Track history
        self.format_history: List[Tuple[str, str, float]] = []  # (renderer, query_fmt, reward)
        
        # Current selections
        self.current_prompt_renderer = 'markdown'
        self.current_query_format = 'qa_plain' if self.task_type == 'qa' else 'mc_plain'
    
    def select_prompt_renderer_uct(self, exclude: Optional[List[str]] = None) -> str:
        """Select next prompt renderer using UCT algorithm."""
        exclude = exclude or []
        best_score = -float('inf')
        best_renderer = None
        
        _log_to_file_and_console(
            "\n      [UCT] Prompt Renderer Selection (exploration_c=" +
            str(self.exploration_constant) + "):"
        )
        
        for name, node in self.prompt_renderer_root.children.items():
            if name in exclude:
                continue
            score = node.ucb1_score(self.exploration_constant)
            
            if node.visits == 0:
                status = "UNEXPLORED"
            else:
                status = f"avg={node.average_reward:.3f}, visits={node.visits}"
            
            _log_to_file_and_console(f"         - {name}: UCB1={score:.3f} ({status})")
            
            if score > best_score:
                best_score = score
                best_renderer = name
        
        _log_to_file_and_console(f"         -> Selected renderer: {best_renderer} (UCB1={best_score:.3f})")
        return best_renderer or 'markdown'
    
    def select_query_format_uct(self, exclude: Optional[List[str]] = None) -> str:
        """Select next query format using UCT algorithm."""
        exclude = exclude or []
        best_score = -float('inf')
        best_format = None
        
        _log_to_file_and_console(
            "\n      [UCT] Query Format Selection (exploration_c=" +
            str(self.exploration_constant) + "):"
        )
        
        for name, node in self.query_format_root.children.items():
            if name in exclude:
                continue
            score = node.ucb1_score(self.exploration_constant)
            
            if node.visits == 0:
                status = "UNEXPLORED"
            else:
                status = f"avg={node.average_reward:.3f}, visits={node.visits}"
            
            _log_to_file_and_console(f"         - {name}: UCB1={score:.3f} ({status})")
            
            if score > best_score:
                best_score = score
                best_format = name
        
        default = 'qa_plain' if self.task_type == 'qa' else 'mc_plain'
        _log_to_file_and_console(f"         -> Selected query format: {best_format or default} (UCB1={best_score:.3f})")
        return best_format or default
    
    def update_format_reward(self, prompt_renderer: str, query_format: str, reward: float) -> None:
        """Update UCT trees with observed reward."""
        if prompt_renderer in self.prompt_renderer_root.children:
            self.prompt_renderer_root.children[prompt_renderer].update(reward)
        
        if query_format in self.query_format_root.children:
            self.query_format_root.children[query_format].update(reward)
        
        self.format_history.append((prompt_renderer, query_format, reward))
        _log_to_file_and_console(
            f"      [UCT] Updated: renderer={prompt_renderer}, query_fmt={query_format}, reward={reward:.3f}"
        )
    
    def try_generate_new_format(self, current_prompt, round_num: int) -> Optional[Any]:
        """Try to generate a new format using LLM."""
        _ensure_format_pool_imports()
        
        if not self.enable_llm_generation or not self.format_generator:
            return None
        
        # Only try generation after first round and with some probability
        if round_num < 2 or random.random() > 0.3:
            return None
        
        _log_to_file_and_console("\n      [LLM-Gen] Attempting to generate new format...")
        
        try:
            # Generate prompt renderer
            existing_renderers = list(self.prompt_renderers.keys())
            generated = self.format_generator.generate_prompt_renderer(current_prompt, existing_renderers)
            
            if generated:
                # Generate the code
                generated = self.format_generator.generate_format_code(generated, 'prompt_renderer')
                
                if generated.is_valid:
                    # Add to pool
                    self.prompt_renderers[generated.name] = (generated.render_fn, generated.extract_fn)
                    self.prompt_renderer_root.add_child(generated.name)
                    self.format_generator.add_generated_format(generated)
                    
                    _log_to_file_and_console(f"      [LLM-Gen] Successfully generated: {generated.name}")
                    _log_to_file_and_console(f"                Description: {generated.description}")
                    return generated
                else:
                    _log_to_file_and_console(f"      [LLM-Gen] Generation failed: {generated.error}")
            
        except Exception as e:
            _log_to_file_and_console(f"      [LLM-Gen] Error: {e}")
        
        return None
    
    def mutate(
        self,
        prompt: Prompt,
        num_mutations: int = 2,
        temperature: float = 0.7,
        round: int = 0,
    ) -> List[Prompt]:
        """Generate prompts with UCT-selected formats."""
        self.logger.info(f"[UCTMutator] Generating {num_mutations} format variations using UCT")
        
        _log_to_file_and_console(f"\n      [UCTMutator] Generating {num_mutations} format variations")
        _log_to_file_and_console(f"         Prompt Renderers: {len(self.prompt_renderers)}")
        _log_to_file_and_console(f"         Query Formats: {len(self.query_formats)}")
        _log_to_file_and_console(f"         Task Type: {self.task_type}")
        
        # Try to generate new format with LLM
        if round >= 2:
            self.try_generate_new_format(prompt, round)
        
        new_prompts = []
        used_combinations = []
        
        for i in range(num_mutations):
            # Select formats using UCT
            exclude_renderers = [c[0] for c in used_combinations]
            exclude_queries = [c[1] for c in used_combinations]
            
            selected_renderer = self.select_prompt_renderer_uct(exclude_renderers)
            selected_query = self.select_query_format_uct(exclude_queries)
            
            used_combinations.append((selected_renderer, selected_query))
            
            # Apply formats
            new_prompt = self._apply_formats(prompt, selected_renderer, selected_query, round)
            if new_prompt:
                new_prompts.append(new_prompt)
                self.logger.info(f"[UCTMutator] Created variant: {selected_renderer} + {selected_query}")
                _log_to_file_and_console(
                    f"         [OK] Variant {i+1}: renderer={selected_renderer}, query_fmt={selected_query}"
                )
        
        return new_prompts
    
    def _apply_formats(self, prompt: Prompt, renderer_name: str, query_format_name: str, round_num: int):
        """Apply both prompt renderer and query format to a prompt."""
        if renderer_name not in self.prompt_renderers:
            renderer_name = 'markdown'
        
        new_prompt = deepcopy(prompt)
        new_prompt.prompt_renderer = renderer_name
        new_prompt.query_format = query_format_name
        new_prompt.format_style = f"{renderer_name}+{query_format_name}"
        new_prompt.action = f"format_{renderer_name}"
        new_prompt.round = round_num
        
        # Store renderer/extractor functions
        renderer_fn, extractor_fn = self.prompt_renderers[renderer_name]
        new_prompt._format_renderer = renderer_fn
        new_prompt._format_extractor = extractor_fn
        
        # Store query format functions
        if query_format_name in self.query_formats:
            query_renderer, query_extractor = self.query_formats[query_format_name]
            new_prompt._query_renderer = query_renderer
            new_prompt._query_extractor = query_extractor
        
        return new_prompt
    
    def traverse_all_formats(self, prompt: Prompt, round_num: int) -> List[Prompt]:
        """Traverse all format combinations (for exhaustive search in later rounds)."""
        _log_to_file_and_console(f"\n      [UCT] Traversing all format combinations...")
        
        new_prompts = []
        current_renderer = getattr(prompt, 'prompt_renderer', 'markdown')
        current_query = getattr(prompt, 'query_format', 'qa_plain')
        
        for renderer_name in self.prompt_renderers:
            for query_name in self.query_formats:
                if renderer_name != current_renderer or query_name != current_query:
                    new_prompt = self._apply_formats(prompt, renderer_name, query_name, round_num)
                    if new_prompt:
                        new_prompt.action = "traverse"
                        new_prompts.append(new_prompt)
        
        _log_to_file_and_console(f"         Generated {len(new_prompts)} format combinations")
        return new_prompts
    
    def get_renderer_statistics(self) -> Dict[str, Any]:
        """Get UCT statistics for prompt renderers."""
        stats = {}
        for name, node in self.prompt_renderer_root.children.items():
            stats[name] = {
                'visits': node.visits,
                'average_reward': node.average_reward,
                'total_reward': node.total_reward,
                'ucb1_score': node.ucb1_score(self.exploration_constant),
                'description': PROMPT_RENDERER_DESCRIPTIONS.get(name, ''),
            }
        return stats
    
    def get_query_format_statistics(self) -> Dict[str, Any]:
        """Get UCT statistics for query formats."""
        stats = {}
        for name, node in self.query_format_root.children.items():
            stats[name] = {
                'visits': node.visits,
                'average_reward': node.average_reward,
                'total_reward': node.total_reward,
                'ucb1_score': node.ucb1_score(self.exploration_constant),
                'description': QUERY_FORMAT_DESCRIPTIONS.get(name, ''),
            }
        return stats
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get combined statistics."""
        return {
            'prompt_renderers': self.get_renderer_statistics(),
            'query_formats': self.get_query_format_statistics(),
            'generated_formats': list(self.format_generator.generated_formats.keys()) if self.format_generator else [],
            'history_length': len(self.format_history),
        }
    
    def get_best_format(self) -> Tuple[str, str, float]:
        """Get the best performing format combination."""
        best_renderer = None
        best_renderer_avg = -float('inf')
        
        for name, node in self.prompt_renderer_root.children.items():
            if node.visits > 0 and node.average_reward > best_renderer_avg:
                best_renderer_avg = node.average_reward
                best_renderer = name
        
        best_query = None
        best_query_avg = -float('inf')
        
        for name, node in self.query_format_root.children.items():
            if node.visits > 0 and node.average_reward > best_query_avg:
                best_query_avg = node.average_reward
                best_query = name
        
        combined_avg = (best_renderer_avg + best_query_avg) / 2 if best_renderer and best_query else 0
        return best_renderer or 'markdown', best_query or 'qa_plain', combined_avg


class FormatSearchPool:
    """
    A pool of format templates with UCT-based selection.
    Supports both prompt renderers and query formats.
    """
    
    def __init__(self, task_type: str = 'qa', exploration_constant: float = 1.414):
        self.task_type = task_type
        self.exploration_constant = exploration_constant
        
        # Prompt renderers with extractors
        self.prompt_renderers = dict(PROMPT_RENDERERS)
        self.prompt_renderer_descriptions = dict(PROMPT_RENDERER_DESCRIPTIONS)
        
        # Query formats with extractors
        self.query_formats = get_formats_for_task(task_type)
        self.query_format_descriptions = dict(QUERY_FORMAT_DESCRIPTIONS)
        
        # UCT nodes for prompt renderers
        self.renderer_root = UCTNode('renderer_root')
        for name in self.prompt_renderers:
            self.renderer_root.add_child(name)
        
        # UCT nodes for query formats
        self.query_root = UCTNode('query_root')
        for name in self.query_formats:
            self.query_root.add_child(name)
        
        self.history = []
    
    def get_prompt_renderer_names(self) -> List[str]:
        """Get all prompt renderer names."""
        return list(self.prompt_renderers.keys())
    
    def get_query_format_names(self) -> List[str]:
        """Get all query format names."""
        return list(self.query_formats.keys())
    
    def render_prompt(self, prompt, renderer_name: str) -> str:
        """Render a prompt using the specified renderer."""
        if renderer_name not in self.prompt_renderers:
            renderer_name = 'plain'
        render_fn = self.prompt_renderers[renderer_name][0]
        return render_fn(prompt)
    
    def extract_from_prompt(self, text: str, renderer_name: str) -> Dict[str, str]:
        """Extract components from a prompt text."""
        if renderer_name not in self.prompt_renderers:
            renderer_name = 'plain'
        extract_fn = self.prompt_renderers[renderer_name][1]
        return extract_fn(text)
    
    def render_query(self, question: str, answer: str, format_name: str, choices: List[str] = None) -> str:
        """Render a Q&A pair using the specified format."""
        if format_name not in self.query_formats:
            format_name = 'qa_plain' if self.task_type == 'qa' else 'mc_plain'
        
        render_fn = self.query_formats[format_name][0]
        
        # Check if it's a multiple choice format
        if format_name.startswith('mc_') and choices:
            return render_fn(question, choices, answer, "")
        else:
            return render_fn(question, answer, "")
    
    def extract_answer(self, response: str, format_name: str) -> Tuple[Optional[str], str]:
        """Extract answer from a response."""
        if format_name not in self.query_formats:
            format_name = 'qa_plain' if self.task_type == 'qa' else 'mc_plain'
        
        extract_fn = self.query_formats[format_name][1]
        return extract_fn(response)
    
    def select_next_renderer(self, exclude: Optional[List[str]] = None) -> str:
        """Select next prompt renderer using UCT."""
        exclude = exclude or []
        best_score = -float('inf')
        best_renderer = None
        
        for name, node in self.renderer_root.children.items():
            if name in exclude:
                continue
            score = node.ucb1_score(self.exploration_constant)
            if score > best_score:
                best_score = score
                best_renderer = name
        
        return best_renderer or 'plain'
    
    def select_next_query_format(self, exclude: Optional[List[str]] = None) -> str:
        """Select next query format using UCT."""
        exclude = exclude or []
        best_score = -float('inf')
        best_format = None
        
        for name, node in self.query_root.children.items():
            if name in exclude:
                continue
            score = node.ucb1_score(self.exploration_constant)
            if score > best_score:
                best_score = score
                best_format = name
        
        default = 'qa_plain' if self.task_type == 'qa' else 'mc_plain'
        return best_format or default
    
    def update_renderer(self, name: str, reward: float) -> None:
        """Update UCT score for a prompt renderer."""
        if name in self.renderer_root.children:
            self.renderer_root.children[name].update(reward)
    
    def update_query_format(self, name: str, reward: float) -> None:
        """Update UCT score for a query format."""
        if name in self.query_root.children:
            self.query_root.children[name].update(reward)
    
    def update(self, renderer_name: str, query_format_name: str, reward: float, metadata=None) -> None:
        """Update both pools with a reward."""
        self.update_renderer(renderer_name, reward)
        self.update_query_format(query_format_name, reward)
        
        entry = {
            'renderer': renderer_name,
            'query_format': query_format_name,
            'reward': reward,
            'metadata': metadata or {},
        }
        self.history.append(entry)
    
    def get_top_renderers(self, k: int = 3) -> List[Tuple[str, float]]:
        """Get top-k performing prompt renderers."""
        scores = []
        for name, node in self.renderer_root.children.items():
            if node.visits > 0:
                scores.append((name, node.average_reward))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
    
    def get_top_query_formats(self, k: int = 3) -> List[Tuple[str, float]]:
        """Get top-k performing query formats."""
        scores = []
        for name, node in self.query_root.children.items():
            if node.visits > 0:
                scores.append((name, node.average_reward))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
    
    def summary(self) -> str:
        """Get a summary of format pool performance."""
        lines = ["Format Search Pool Summary:"]
        lines.append("=" * 50)
        
        lines.append("\nPrompt Renderers:")
        lines.append("-" * 40)
        sorted_renderers = sorted(
            self.renderer_root.children.items(),
            key=lambda x: x[1].average_reward,
            reverse=True
        )
        for name, node in sorted_renderers:
            if node.visits > 0:
                lines.append(f"  {name}: avg={node.average_reward:.3f}, visits={node.visits}")
            else:
                lines.append(f"  {name}: not visited")
        
        lines.append("\nQuery Formats:")
        lines.append("-" * 40)
        sorted_queries = sorted(
            self.query_root.children.items(),
            key=lambda x: x[1].average_reward,
            reverse=True
        )
        for name, node in sorted_queries:
            if node.visits > 0:
                lines.append(f"  {name}: avg={node.average_reward:.3f}, visits={node.visits}")
            else:
                lines.append(f"  {name}: not visited")
        
        return "\n".join(lines)
