"""
Prompt Renderer Module

Defines various prompt rendering templates that control
the overall structure of how prompts are presented.

Each renderer has:
- render(prompt) -> str: Renders the full prompt
- extract(response) -> dict: Extracts components from a response
"""

import re
from typing import Callable, Dict, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class PromptRenderer:
    """A prompt renderer with render and extract functions."""
    name: str
    description: str
    render: Callable
    extract: Callable


# ============================================================================
# MARKDOWN RENDERER
# ============================================================================

def markdown_renderer(prompt) -> str:
    """Render prompt in Markdown format."""
    return f"""# Task
{prompt.task_instruction}

## Details
{prompt.task_detail}

## Output Format
{prompt.output_format}

## Examples
{prompt.examples}

{prompt.cot_hinter}"""


def markdown_extractor(text: str) -> Dict[str, str]:
    """Extract components from Markdown-formatted text."""
    components = {}
    patterns = {
        'task_instruction': r'# Task\n(.*?)(?=\n## |\Z)',
        'task_detail': r'## Details\n(.*?)(?=\n## |\Z)',
        'output_format': r'## Output Format\n(.*?)(?=\n## |\Z)',
        'examples': r'## Examples\n(.*?)(?=\n[^#]|\Z)',
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            components[key] = match.group(1).strip()
    return components


# ============================================================================
# XML RENDERER
# ============================================================================

def xml_renderer(prompt) -> str:
    """Render prompt in XML format."""
    return f"""<task>
<instruction>{prompt.task_instruction}</instruction>
<details>{prompt.task_detail}</details>
<output_format>{prompt.output_format}</output_format>
<examples>{prompt.examples}</examples>
</task>

{prompt.cot_hinter}"""


def xml_extractor(text: str) -> Dict[str, str]:
    """Extract components from XML-formatted text."""
    components = {}
    patterns = {
        'task_instruction': r'<instruction>(.*?)</instruction>',
        'task_detail': r'<details>(.*?)</details>',
        'output_format': r'<output_format>(.*?)</output_format>',
        'examples': r'<examples>(.*?)</examples>',
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            components[key] = match.group(1).strip()
    return components


# ============================================================================
# PLAIN RENDERER
# ============================================================================

def plain_renderer(prompt) -> str:
    """Render prompt in plain text format."""
    return f"""{prompt.task_instruction}

{prompt.task_detail}

{prompt.output_format}

{prompt.examples}

{prompt.cot_hinter}"""


def plain_extractor(text: str) -> Dict[str, str]:
    """Extract components from plain text (best effort)."""
    lines = text.strip().split('\n\n')
    components = {}
    if len(lines) >= 1:
        components['task_instruction'] = lines[0]
    if len(lines) >= 2:
        components['task_detail'] = lines[1]
    if len(lines) >= 3:
        components['output_format'] = lines[2]
    if len(lines) >= 4:
        components['examples'] = '\n\n'.join(lines[3:-1]) if len(lines) > 4 else lines[3]
    return components


# ============================================================================
# STRUCTURED RENDERER
# ============================================================================

def structured_renderer(prompt) -> str:
    """Render prompt with clear section markers."""
    return f"""=== TASK ===
{prompt.task_instruction}

=== INSTRUCTIONS ===
{prompt.task_detail}

=== OUTPUT REQUIREMENTS ===
{prompt.output_format}

=== EXAMPLES ===
{prompt.examples}

=== BEGIN ===
{prompt.cot_hinter}"""


def structured_extractor(text: str) -> Dict[str, str]:
    """Extract components from structured format."""
    components = {}
    patterns = {
        'task_instruction': r'=== TASK ===\n(.*?)(?=\n=== |\Z)',
        'task_detail': r'=== INSTRUCTIONS ===\n(.*?)(?=\n=== |\Z)',
        'output_format': r'=== OUTPUT REQUIREMENTS ===\n(.*?)(?=\n=== |\Z)',
        'examples': r'=== EXAMPLES ===\n(.*?)(?=\n=== |\Z)',
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            components[key] = match.group(1).strip()
    return components


# ============================================================================
# JSON RENDERER
# ============================================================================

def json_renderer(prompt) -> str:
    """Render prompt in JSON-like format."""
    return f"""{{
  "task": "{prompt.task_instruction}",
  "details": "{prompt.task_detail}",
  "output_format": "{prompt.output_format}",
  "examples": "{prompt.examples}"
}}

{prompt.cot_hinter}"""


def json_extractor(text: str) -> Dict[str, str]:
    """Extract components from JSON-like format."""
    components = {}
    patterns = {
        'task_instruction': r'"task":\s*"(.*?)"',
        'task_detail': r'"details":\s*"(.*?)"',
        'output_format': r'"output_format":\s*"(.*?)"',
        'examples': r'"examples":\s*"(.*?)"',
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            components[key] = match.group(1).strip()
    return components


# ============================================================================
# HTML RENDERER
# ============================================================================

def html_renderer(prompt) -> str:
    """Render prompt in HTML format."""
    return f"""<div class="prompt">
  <h1>Task</h1>
  <p>{prompt.task_instruction}</p>
  
  <h2>Details</h2>
  <p>{prompt.task_detail}</p>
  
  <h2>Output Format</h2>
  <p>{prompt.output_format}</p>
  
  <h2>Examples</h2>
  <pre>{prompt.examples}</pre>
</div>

{prompt.cot_hinter}"""


def html_extractor(text: str) -> Dict[str, str]:
    """Extract components from HTML format."""
    components = {}
    patterns = {
        'task_instruction': r'<h1>Task</h1>\s*<p>(.*?)</p>',
        'task_detail': r'<h2>Details</h2>\s*<p>(.*?)</p>',
        'output_format': r'<h2>Output Format</h2>\s*<p>(.*?)</p>',
        'examples': r'<pre>(.*?)</pre>',
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            components[key] = match.group(1).strip()
    return components


# ============================================================================
# LATEX RENDERER
# ============================================================================

def latex_renderer(prompt) -> str:
    """Render prompt in LaTeX format."""
    return f"""\\section{{Task}}
{prompt.task_instruction}

\\subsection{{Details}}
{prompt.task_detail}

\\subsection{{Output Format}}
{prompt.output_format}

\\subsection{{Examples}}
\\begin{{verbatim}}
{prompt.examples}
\\end{{verbatim}}

{prompt.cot_hinter}"""


def latex_extractor(text: str) -> Dict[str, str]:
    """Extract components from LaTeX format."""
    components = {}
    patterns = {
        'task_instruction': r'\\section\{Task\}\n(.*?)(?=\\subsection|\Z)',
        'task_detail': r'\\subsection\{Details\}\n(.*?)(?=\\subsection|\Z)',
        'output_format': r'\\subsection\{Output Format\}\n(.*?)(?=\\subsection|\Z)',
        'examples': r'\\begin\{verbatim\}\n(.*?)\\end\{verbatim\}',
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            components[key] = match.group(1).strip()
    return components


# ============================================================================
# CONVERSATIONAL RENDERER
# ============================================================================

def conversational_renderer(prompt) -> str:
    """Render prompt in conversational style."""
    return f"""I need your help with a task.

Here's what I need you to do: {prompt.task_instruction}

Some additional details: {prompt.task_detail}

Please format your answer like this: {prompt.output_format}

Here are some examples to guide you:
{prompt.examples}

{prompt.cot_hinter}"""


def conversational_extractor(text: str) -> Dict[str, str]:
    """Extract components from conversational format."""
    components = {}
    patterns = {
        'task_instruction': r"Here's what I need you to do: (.*?)(?=\n\nSome additional|\Z)",
        'task_detail': r'Some additional details: (.*?)(?=\n\nPlease format|\Z)',
        'output_format': r'Please format your answer like this: (.*?)(?=\n\nHere are|\Z)',
        'examples': r'Here are some examples to guide you:\n(.*?)(?=\n\n[A-Z]|\Z)',
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            components[key] = match.group(1).strip()
    return components


# ============================================================================
# NUMBERED RENDERER
# ============================================================================

def numbered_renderer(prompt) -> str:
    """Render prompt with numbered sections."""
    return f"""1. TASK: {prompt.task_instruction}

2. DETAILS: {prompt.task_detail}

3. OUTPUT FORMAT: {prompt.output_format}

4. EXAMPLES:
{prompt.examples}

5. NOW SOLVE:
{prompt.cot_hinter}"""


def numbered_extractor(text: str) -> Dict[str, str]:
    """Extract components from numbered format."""
    components = {}
    patterns = {
        'task_instruction': r'1\. TASK: (.*?)(?=\n\n2\.|\Z)',
        'task_detail': r'2\. DETAILS: (.*?)(?=\n\n3\.|\Z)',
        'output_format': r'3\. OUTPUT FORMAT: (.*?)(?=\n\n4\.|\Z)',
        'examples': r'4\. EXAMPLES:\n(.*?)(?=\n\n5\.|\Z)',
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            components[key] = match.group(1).strip()
    return components


# ============================================================================
# ACADEMIC RENDERER
# ============================================================================

def academic_renderer(prompt) -> str:
    """Render prompt in academic paper style."""
    return f"""PROBLEM STATEMENT:
{prompt.task_instruction}

METHODOLOGY:
{prompt.task_detail}

EXPECTED OUTPUT FORMAT:
{prompt.output_format}

WORKED EXAMPLES:
{prompt.examples}

SOLUTION:
{prompt.cot_hinter}"""


def academic_extractor(text: str) -> Dict[str, str]:
    """Extract components from academic format."""
    components = {}
    patterns = {
        'task_instruction': r'PROBLEM STATEMENT:\n(.*?)(?=\nMETHODOLOGY:|\Z)',
        'task_detail': r'METHODOLOGY:\n(.*?)(?=\nEXPECTED OUTPUT FORMAT:|\Z)',
        'output_format': r'EXPECTED OUTPUT FORMAT:\n(.*?)(?=\nWORKED EXAMPLES:|\Z)',
        'examples': r'WORKED EXAMPLES:\n(.*?)(?=\nSOLUTION:|\Z)',
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            components[key] = match.group(1).strip()
    return components


# ============================================================================
# CHAIN OF THOUGHT RENDERER
# ============================================================================

def chain_of_thought_renderer(prompt) -> str:
    """Render prompt emphasizing step-by-step reasoning."""
    return f"""{prompt.task_instruction}

Let me break this down step by step:
{prompt.task_detail}

Remember to format your answer as:
{prompt.output_format}

Here are some examples showing the reasoning:
{prompt.examples}

Now think through this problem step by step:
{prompt.cot_hinter}"""


def chain_of_thought_extractor(text: str) -> Dict[str, str]:
    """Extract components from chain-of-thought format."""
    components = {}
    patterns = {
        'task_instruction': r'^(.*?)(?=\n\nLet me break|\Z)',
        'task_detail': r'Let me break this down step by step:\n(.*?)(?=\n\nRemember to format|\Z)',
        'output_format': r'Remember to format your answer as:\n(.*?)(?=\n\nHere are some examples|\Z)',
        'examples': r'Here are some examples showing the reasoning:\n(.*?)(?=\n\nNow think|\Z)',
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            components[key] = match.group(1).strip()
    return components


# ============================================================================
# PROMPT RENDERER REGISTRY
# ============================================================================

PROMPT_RENDERERS: Dict[str, Tuple[Callable, Callable]] = {
    'markdown': (markdown_renderer, markdown_extractor),
    'xml': (xml_renderer, xml_extractor),
    'plain': (plain_renderer, plain_extractor),
    'structured': (structured_renderer, structured_extractor),
    'json': (json_renderer, json_extractor),
    'html': (html_renderer, html_extractor),
    'latex': (latex_renderer, latex_extractor),
    'conversational': (conversational_renderer, conversational_extractor),
    'numbered': (numbered_renderer, numbered_extractor),
    'academic': (academic_renderer, academic_extractor),
    'chain_of_thought': (chain_of_thought_renderer, chain_of_thought_extractor),
}

PROMPT_RENDERER_DESCRIPTIONS: Dict[str, str] = {
    'markdown': 'Uses Markdown headers (# ##) for structure',
    'xml': 'Uses XML-style tags for clear nesting',
    'plain': 'Simple plain text with paragraph breaks',
    'structured': 'Uses === markers for section boundaries',
    'json': 'JSON-like key-value structure',
    'html': 'HTML tags with semantic elements',
    'latex': 'LaTeX document structure',
    'conversational': 'Natural conversational tone',
    'numbered': 'Numbered list format (1. 2. 3.)',
    'academic': 'Formal academic paper structure',
    'chain_of_thought': 'Emphasizes step-by-step reasoning',
}


def get_prompt_renderer(name: str) -> PromptRenderer:
    """Get a PromptRenderer by name."""
    if name not in PROMPT_RENDERERS:
        name = 'plain'
    
    render_fn, extract_fn = PROMPT_RENDERERS[name]
    return PromptRenderer(
        name=name,
        description=PROMPT_RENDERER_DESCRIPTIONS.get(name, ''),
        render=render_fn,
        extract=extract_fn,
    )
