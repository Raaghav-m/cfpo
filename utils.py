"""
CFPO Utilities

Helper functions for prompt optimization.
"""

import re
import time
from typing import List, Dict, Any, Optional
import logging


def convert_seconds(seconds: float) -> str:
    """
    Convert seconds to human-readable time format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string like "1h 23m 45s"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    
    return " ".join(parts)


def stringify_dict(d: Dict, indent: int = 0) -> str:
    """
    Convert a dictionary to a formatted string.
    
    Args:
        d: Dictionary to format
        indent: Number of spaces for indentation
        
    Returns:
        Formatted string
    """
    lines = []
    prefix = " " * indent
    
    for key, value in d.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(stringify_dict(value, indent + 2))
        else:
            lines.append(f"{prefix}{key}: {value}")
    
    return "\n".join(lines)


def parse_tagged_text(
    text: str, 
    start_tag: str, 
    end_tag: str,
    logger: Optional[logging.Logger] = None
) -> List[str]:
    """
    Extract content between start and end tags.
    
    Args:
        text: Text to parse
        start_tag: Opening tag (e.g., "<START>")
        end_tag: Closing tag (e.g., "<END>")
        logger: Optional logger for debugging
        
    Returns:
        List of extracted contents
    """
    results = []
    
    # Escape special regex characters in tags
    start_escaped = re.escape(start_tag)
    end_escaped = re.escape(end_tag)
    
    # Find all matches
    pattern = f"{start_escaped}(.*?){end_escaped}"
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        content = match.strip()
        if content:
            results.append(content)
    
    if not results and logger:
        logger.debug(f"No content found between {start_tag} and {end_tag}")
    
    return results


def clean_response(text: str) -> str:
    """
    Clean up LLM response by removing quotes, extra whitespace, etc.
    
    Args:
        text: Raw LLM response
        
    Returns:
        Cleaned text
    """
    # Remove leading/trailing quotes and whitespace
    text = re.sub(r'^[\n"\' ]+|[\n"\' ]+$', '', text)
    
    # Remove markdown code block markers
    if text.startswith("```") and text.endswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else ""
    
    return text.strip()


def extract_number(text: str) -> Optional[str]:
    """
    Extract a number from text.
    
    Args:
        text: Text containing a number
        
    Returns:
        Extracted number as string, or None
    """
    # Try common patterns
    patterns = [
        r'[Tt]he answer is:?\s*\$?([0-9,.-]+)',
        r'####\s*\$?([0-9,.-]+)',
        r'=\s*\$?([0-9,.-]+)\s*$',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).replace(',', '').strip()
    
    # Fallback: last number in text
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return numbers[-1] if numbers else None


def normalize_number(num_str: str) -> float:
    """
    Normalize a number string for comparison.
    
    Args:
        num_str: Number as string
        
    Returns:
        Normalized float
    """
    try:
        # Remove currency symbols and commas
        cleaned = re.sub(r'[$,]', '', num_str)
        return float(cleaned)
    except (ValueError, TypeError):
        return float('nan')


def numbers_equal(num1: str, num2: str, tolerance: float = 1e-6) -> bool:
    """
    Check if two number strings are equal.
    
    Args:
        num1: First number
        num2: Second number
        tolerance: Floating point tolerance
        
    Returns:
        True if numbers are equal
    """
    try:
        n1 = normalize_number(num1)
        n2 = normalize_number(num2)
        return abs(n1 - n2) < tolerance
    except:
        return num1.strip() == num2.strip()


class Timer:
    """Simple timer context manager."""
    
    def __init__(self, name: str = "", logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        if self.logger:
            self.logger.info(f"{self.name}: {convert_seconds(elapsed)}")
        return False
    
    @property
    def elapsed(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return time.time() - self.start_time
        return 0.0


def batch_list(lst: List, batch_size: int) -> List[List]:
    """
    Split a list into batches.
    
    Args:
        lst: List to split
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]


def get_component_description(component_key: str) -> str:
    """
    Get a human-readable description of a prompt component.
    
    Args:
        component_key: Component key (e.g., "TASK_INSTRUCTION")
        
    Returns:
        Description string
    """
    descriptions = {
        'TASK_INSTRUCTION': 'The main instruction describing what task to perform',
        'TASK_DETAIL': 'Additional details, constraints, or guidelines for the task',
        'OUTPUT_FORMAT': 'How the response should be formatted',
        'EXAMPLE_HINTER': 'Introduction text for the examples section',
        'EXAMPLES': 'Few-shot examples demonstrating the task',
        'COT_HINTER': 'Chain-of-thought hint to encourage step-by-step reasoning',
        'PROMPT_FORMAT': 'The overall structural format of the prompt (markdown, xml, etc.)',
        'QUERY_FORMAT': 'How individual queries are formatted',
    }
    
    return descriptions.get(component_key, f"The {component_key} component")


# Prompt format templates (for programmatic use)
PROMPT_FORMATS = {
    'markdown': {
        'name': 'Markdown',
        'template': """# Task
{task_instruction}

## Details
{task_detail}

## Output Format
{output_format}

## Examples
{examples}

{cot_hinter}
"""
    },
    'xml': {
        'name': 'XML',
        'template': """<task>
<instruction>{task_instruction}</instruction>
<details>{task_detail}</details>
<output_format>{output_format}</output_format>
<examples>{examples}</examples>
</task>

{cot_hinter}
"""
    },
    'plain': {
        'name': 'Plain Text',
        'template': """{task_instruction}

{task_detail}

{output_format}

{examples}

{cot_hinter}
"""
    },
    'structured': {
        'name': 'Structured',
        'template': """=== TASK ===
{task_instruction}

=== INSTRUCTIONS ===
{task_detail}

=== OUTPUT FORMAT ===
{output_format}

=== EXAMPLES ===
{examples}

=== BEGIN ===
{cot_hinter}
"""
    }
}
