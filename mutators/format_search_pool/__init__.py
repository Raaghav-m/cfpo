"""
Format Search Pool Module

Implements the two-level format system from CFPO:
1. Prompt Renderer - Overall prompt structure
2. Query Format - How questions/answers are formatted

Also includes LLM-guided format generation.
"""

from .prompt_renderer import (
    PROMPT_RENDERERS,
    PROMPT_RENDERER_DESCRIPTIONS,
    PromptRenderer,
    get_prompt_renderer,
)

from .query_format import (
    QUERY_FORMATS,
    QUERY_FORMAT_DESCRIPTIONS,
    QueryFormat,
    get_query_format,
)

from .format_generator import (
    LLMFormatGenerator,
    GeneratedFormat,
)

__all__ = [
    'PROMPT_RENDERERS',
    'PROMPT_RENDERER_DESCRIPTIONS',
    'PromptRenderer',
    'get_prompt_renderer',
    'QUERY_FORMATS',
    'QUERY_FORMAT_DESCRIPTIONS',
    'QueryFormat',
    'get_query_format',
    'LLMFormatGenerator',
    'GeneratedFormat',
]
