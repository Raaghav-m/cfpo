# Mutators package

# Import base mutator first
from .base import Mutator

# Import concrete mutators
from .case_diagnosis import CaseDiagnosisMutator
from .monte_carlo import MonteCarloMutator
from .format_mutator import FormatMutator
from .uct_mutator import UCTMutator, FormatSearchPool, UCTNode, FORMAT_POOL, FORMAT_DESCRIPTIONS

# Lazy accessors for format search pool (avoid circular imports at module level)
def get_format_search_pool_components():
    """Get format search pool components (lazy import)."""
    from .format_search_pool.prompt_renderer import (
        PROMPT_RENDERERS,
        PROMPT_RENDERER_DESCRIPTIONS,
        PromptRenderer,
        get_prompt_renderer,
    )
    from .format_search_pool.query_format import (
        QUERY_FORMATS,
        QUERY_FORMAT_DESCRIPTIONS,
        QueryFormat,
        get_query_format,
    )
    from .format_search_pool.format_generator import (
        LLMFormatGenerator,
        GeneratedFormat,
    )
    return {
        'PROMPT_RENDERERS': PROMPT_RENDERERS,
        'PROMPT_RENDERER_DESCRIPTIONS': PROMPT_RENDERER_DESCRIPTIONS,
        'PromptRenderer': PromptRenderer,
        'get_prompt_renderer': get_prompt_renderer,
        'QUERY_FORMATS': QUERY_FORMATS,
        'QUERY_FORMAT_DESCRIPTIONS': QUERY_FORMAT_DESCRIPTIONS,
        'QueryFormat': QueryFormat,
        'get_query_format': get_query_format,
        'LLMFormatGenerator': LLMFormatGenerator,
        'GeneratedFormat': GeneratedFormat,
    }

__all__ = [
    'Mutator',
    'CaseDiagnosisMutator',
    'MonteCarloMutator',
    'FormatMutator',
    'UCTMutator',
    'FormatSearchPool',
    'UCTNode',
    'FORMAT_POOL',
    'FORMAT_DESCRIPTIONS',
    'get_format_search_pool_components',
]
