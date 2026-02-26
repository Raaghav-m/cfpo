"""
Query Format Module

Defines how individual questions and examples are formatted.
Separate from the overall prompt structure (handled by prompt_renderer).

Each query format has:
- render(question, answer, cot_hinter) -> str: Renders a Q&A pair
- extract(response) -> tuple: Extracts answer from response
"""

import re
from typing import Callable, Dict, Tuple, Optional, List, Any
from dataclasses import dataclass


@dataclass
class QueryFormat:
    """A query format with render and extract functions."""
    name: str
    description: str
    task_type: str  # 'qa', 'multiple_choice', 'math'
    render: Callable
    extract: Callable


# ============================================================================
# MATH/QA QUERY FORMATS
# ============================================================================

def qa_plain_renderer(question: str, answer: str, cot_hinter: str = "") -> str:
    """Plain Q&A format for math/reasoning."""
    return f"""Q: {question}
A: {answer}"""


def qa_plain_extractor(response: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract answer from plain Q&A response."""
    # Look for "The answer is: X" pattern
    patterns = [
        r'[Tt]he answer is[:\s]*([^\n.]+)',
        r'[Aa]nswer[:\s]*([^\n.]+)',
        r'= (\d+(?:\.\d+)?)\s*$',
        r'(\d+(?:\.\d+)?)\s*$',
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.MULTILINE)
        if match:
            return match.group(1).strip(), response
    return None, response


def qa_instruction_response_renderer(question: str, answer: str, cot_hinter: str = "") -> str:
    """Instruction-Response format."""
    return f"""Instruction: {question}
Response: {answer}"""


def qa_instruction_response_extractor(response: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract from Instruction-Response format."""
    match = re.search(r'Response:\s*(.*?)(?=\n\n|\Z)', response, re.DOTALL)
    if match:
        resp_text = match.group(1).strip()
        # Look for final answer
        ans_match = re.search(r'[Tt]he answer is[:\s]*([^\n.]+)', resp_text)
        if ans_match:
            return ans_match.group(1).strip(), resp_text
        return None, resp_text
    return None, response


def qa_problem_solution_renderer(question: str, answer: str, cot_hinter: str = "") -> str:
    """Problem-Solution format."""
    return f"""Problem: {question}

Solution: {answer}"""


def qa_problem_solution_extractor(response: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract from Problem-Solution format."""
    match = re.search(r'Solution:\s*(.*?)(?=\n\nProblem:|\Z)', response, re.DOTALL)
    if match:
        sol_text = match.group(1).strip()
        ans_match = re.search(r'[Tt]he answer is[:\s]*([^\n.]+)', sol_text)
        if ans_match:
            return ans_match.group(1).strip(), sol_text
        return None, sol_text
    return None, response


def qa_input_output_renderer(question: str, answer: str, cot_hinter: str = "") -> str:
    """Input-Output format."""
    return f"""Input: {question}
Output: {answer}"""


def qa_input_output_extractor(response: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract from Input-Output format."""
    match = re.search(r'Output:\s*(.*?)(?=\n\nInput:|\Z)', response, re.DOTALL)
    if match:
        out_text = match.group(1).strip()
        ans_match = re.search(r'[Tt]he answer is[:\s]*([^\n.]+)', out_text)
        if ans_match:
            return ans_match.group(1).strip(), out_text
        return None, out_text
    return None, response


def qa_step_by_step_renderer(question: str, answer: str, cot_hinter: str = "") -> str:
    """Step-by-step reasoning format."""
    return f"""Question: {question}

Step-by-step solution:
{answer}"""


def qa_step_by_step_extractor(response: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract from step-by-step format."""
    patterns = [
        r'[Ff]inal [Aa]nswer[:\s]*([^\n.]+)',
        r'[Tt]he answer is[:\s]*([^\n.]+)',
        r'[Tt]herefore[,:\s]*.*?(\d+(?:\.\d+)?)',
        r'= (\d+(?:\.\d+)?)\s*$',
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.MULTILINE)
        if match:
            return match.group(1).strip(), response
    return None, response


# ============================================================================
# MULTIPLE CHOICE QUERY FORMATS
# ============================================================================

def mc_plain_renderer(question: str, choices: List[str], answer: str, cot_hinter: str = "") -> str:
    """Plain multiple choice format."""
    choice_str = "\n".join([f"{chr(65+i)}) {c}" for i, c in enumerate(choices)])
    return f"""Question: {question}

{choice_str}

{answer}"""


def mc_plain_extractor(response: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract answer letter from MC response."""
    # Look for patterns like "Answer: A", "(A)", "The answer is A"
    patterns = [
        r'[Aa]nswer[:\s]*\(?([A-Da-d])\)?',
        r'[Tt]he answer is[:\s]*\(?([A-Da-d])\)?',
        r'\b\(?([A-Da-d])\)?\s*$',
        r'choose\s*\(?([A-Da-d])\)?',
    ]
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1).upper(), response
    return None, response


def mc_markdown_renderer(question: str, choices: List[str], answer: str, cot_hinter: str = "") -> str:
    """Markdown multiple choice format."""
    choice_str = "\n".join([f"- **{chr(65+i)})** {c}" for i, c in enumerate(choices)])
    return f"""### Question
{question}

### Options
{choice_str}

### Answer
{answer}"""


def mc_markdown_extractor(response: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract from Markdown MC format."""
    match = re.search(r'### Answer\s*(.*?)(?=\n###|\Z)', response, re.DOTALL)
    if match:
        ans_text = match.group(1).strip()
        letter_match = re.search(r'\(?([A-Da-d])\)?', ans_text)
        if letter_match:
            return letter_match.group(1).upper(), ans_text
    return mc_plain_extractor(response)


def mc_numbered_renderer(question: str, choices: List[str], answer: str, cot_hinter: str = "") -> str:
    """Numbered multiple choice format."""
    choice_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
    return f"""Q: {question}

Options:
{choice_str}

A: {answer}"""


def mc_numbered_extractor(response: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract from numbered MC format."""
    # Convert number to letter if needed
    match = re.search(r'A:\s*(.*?)(?=\n\nQ:|\Z)', response, re.DOTALL)
    if match:
        ans_text = match.group(1).strip()
        # Check for letter
        letter_match = re.search(r'\(?([A-Da-d])\)?', ans_text)
        if letter_match:
            return letter_match.group(1).upper(), ans_text
        # Check for number
        num_match = re.search(r'\b([1-4])\b', ans_text)
        if num_match:
            return chr(64 + int(num_match.group(1))), ans_text  # 1->A, 2->B, etc.
    return mc_plain_extractor(response)


def mc_bracket_renderer(question: str, choices: List[str], answer: str, cot_hinter: str = "") -> str:
    """Bracket-style multiple choice format."""
    choice_str = " | ".join([f"[{chr(65+i)}] {c}" for i, c in enumerate(choices)])
    return f"""{question}
Choices: {choice_str}
Answer: {answer}"""


def mc_bracket_extractor(response: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract from bracket MC format."""
    match = re.search(r'Answer:\s*\[?([A-Da-d])\]?', response)
    if match:
        return match.group(1).upper(), response
    return mc_plain_extractor(response)


def mc_parenthesis_renderer(question: str, choices: List[str], answer: str, cot_hinter: str = "") -> str:
    """Parenthesis-style multiple choice format."""
    choice_str = "  ".join([f"({chr(65+i)}) {c}" for i, c in enumerate(choices)])
    return f"""{question}
{choice_str}

The correct answer is: {answer}"""


def mc_parenthesis_extractor(response: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract from parenthesis MC format."""
    match = re.search(r'[Tt]he correct answer is[:\s]*\(?([A-Da-d])\)?', response)
    if match:
        return match.group(1).upper(), response
    return mc_plain_extractor(response)


# ============================================================================
# QUERY FORMAT REGISTRY
# ============================================================================

# Math/QA formats
QA_QUERY_FORMATS: Dict[str, Tuple[Callable, Callable]] = {
    'qa_plain': (qa_plain_renderer, qa_plain_extractor),
    'instruction_response': (qa_instruction_response_renderer, qa_instruction_response_extractor),
    'problem_solution': (qa_problem_solution_renderer, qa_problem_solution_extractor),
    'input_output': (qa_input_output_renderer, qa_input_output_extractor),
    'step_by_step': (qa_step_by_step_renderer, qa_step_by_step_extractor),
}

# Multiple choice formats
MC_QUERY_FORMATS: Dict[str, Tuple[Callable, Callable]] = {
    'mc_plain': (mc_plain_renderer, mc_plain_extractor),
    'mc_markdown': (mc_markdown_renderer, mc_markdown_extractor),
    'mc_numbered': (mc_numbered_renderer, mc_numbered_extractor),
    'mc_bracket': (mc_bracket_renderer, mc_bracket_extractor),
    'mc_parenthesis': (mc_parenthesis_renderer, mc_parenthesis_extractor),
}

# Combined registry
QUERY_FORMATS: Dict[str, Tuple[Callable, Callable]] = {
    **QA_QUERY_FORMATS,
    **MC_QUERY_FORMATS,
}

QUERY_FORMAT_DESCRIPTIONS: Dict[str, str] = {
    # QA formats
    'qa_plain': 'Simple Q: A: format',
    'instruction_response': 'Instruction: Response: format',
    'problem_solution': 'Problem: Solution: format',
    'input_output': 'Input: Output: format',
    'step_by_step': 'Question with step-by-step solution',
    # MC formats
    'mc_plain': 'Plain multiple choice with A) B) C) D)',
    'mc_markdown': 'Markdown-styled multiple choice',
    'mc_numbered': 'Numbered options (1. 2. 3. 4.)',
    'mc_bracket': 'Bracket-style [A] [B] [C] [D]',
    'mc_parenthesis': 'Parenthesis-style (A) (B) (C) (D)',
}


def get_query_format(name: str, task_type: str = 'qa') -> QueryFormat:
    """Get a QueryFormat by name."""
    if name not in QUERY_FORMATS:
        # Default based on task type
        name = 'qa_plain' if task_type == 'qa' else 'mc_plain'
    
    render_fn, extract_fn = QUERY_FORMATS[name]
    return QueryFormat(
        name=name,
        description=QUERY_FORMAT_DESCRIPTIONS.get(name, ''),
        task_type='multiple_choice' if name.startswith('mc_') else 'qa',
        render=render_fn,
        extract=extract_fn,
    )


def get_formats_for_task(task_type: str) -> Dict[str, Tuple[Callable, Callable]]:
    """Get all query formats suitable for a task type."""
    if task_type in ['multiple_choice', 'mc', 'bbh', 'arc', 'mmlu']:
        return MC_QUERY_FORMATS
    else:
        return QA_QUERY_FORMATS
