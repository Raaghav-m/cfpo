"""
Multiple Choice Task - For BBH, ARC, MMLU, and similar benchmarks

Supports multiple choice questions with A/B/C/D options.
"""

import os
import re
import json
import random
from typing import Optional, List, Dict, Any

from .base import Task


class MultipleChoiceTask(Task):
    """
    Multiple Choice Question benchmark.
    
    Supports various MC benchmarks:
    - BBH (BIG-Bench Hard)
    - ARC (AI2 Reasoning Challenge)
    - MMLU (Massive Multitask Language Understanding)
    - HellaSwag
    - WinoGrande
    
    Example:
        Q: What is the capital of France?
        A) London  B) Paris  C) Berlin  D) Madrid
        Answer: B
    """
    
    # Different benchmark types
    BENCHMARK_TYPES = ['bbh', 'arc', 'mmlu', 'hellaswag', 'winogrande', 'custom']
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        train_size: int = 10,
        valid_size: int = 10,
        test_size: int = 10,
        benchmark_type: str = 'bbh',
        subject: Optional[str] = None,  # For MMLU subjects
    ):
        """
        Initialize MultipleChoice task.
        
        Args:
            data_dir: Path to data files
            train_size: Number of training examples
            valid_size: Number of validation examples  
            test_size: Number of test examples
            benchmark_type: Type of benchmark (bbh, arc, mmlu, etc.)
            subject: Subject for MMLU (e.g., 'abstract_algebra', 'anatomy')
        """
        self.benchmark_type = benchmark_type.lower()
        self.subject = subject
        
        super().__init__(
            name=f"MultipleChoice-{benchmark_type.upper()}",
            data_dir=data_dir,
            train_size=train_size,
            valid_size=valid_size,
            test_size=test_size,
            answer_marker="Answer:",
        )
    
    def _load_data(self) -> None:
        """Load data from files or use built-in examples."""
        
        # Try loading from data_dir
        if self.data_dir and os.path.exists(self.data_dir):
            self._load_from_files()
            return
        
        # Use built-in sample data for demo
        self._load_sample_data()
    
    def _load_from_files(self) -> None:
        """Load from JSON/JSONL files based on benchmark type."""
        try:
            if self.benchmark_type == 'mmlu':
                self._load_mmlu_format()
            elif self.benchmark_type == 'arc':
                self._load_arc_format()
            elif self.benchmark_type == 'bbh':
                self._load_bbh_format()
            else:
                self._load_generic_format()
                
        except Exception as e:
            print(f"Error loading data: {e}, using sample data")
            self._load_sample_data()
    
    def _load_mmlu_format(self) -> None:
        """Load MMLU format (CSV/JSONL)."""
        subject = self.subject or 'abstract_algebra'
        data_file = os.path.join(self.data_dir, f'{subject}_test.csv')
        
        all_data = []
        if os.path.exists(data_file):
            import csv
            with open(data_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 6:  # question, A, B, C, D, answer
                        all_data.append({
                            'question': row[0],
                            'choices': [row[1], row[2], row[3], row[4]],
                            'answer': row[5],
                            'subject': subject
                        })
        
        self._split_data(all_data)
    
    def _load_arc_format(self) -> None:
        """Load ARC format (JSONL)."""
        data_file = os.path.join(self.data_dir, 'ARC-Challenge-Test.jsonl')
        
        all_data = []
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    choices = item.get('choices', {})
                    all_data.append({
                        'question': item.get('question', ''),
                        'choices': choices.get('text', []),
                        'answer': item.get('answerKey', ''),
                    })
        
        self._split_data(all_data)
    
    def _load_bbh_format(self) -> None:
        """Load BBH format (JSON)."""
        data_file = os.path.join(self.data_dir, 'data.json')
        
        all_data = []
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                data = json.load(f)
                for item in data.get('examples', []):
                    all_data.append({
                        'question': item.get('input', ''),
                        'choices': item.get('choices', []),
                        'answer': item.get('target', ''),
                    })
        
        self._split_data(all_data)
    
    def _load_generic_format(self) -> None:
        """Load generic JSONL format."""
        data_file = os.path.join(self.data_dir, 'data.jsonl')
        
        all_data = []
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    all_data.append({
                        'question': item.get('question', item.get('input', '')),
                        'choices': item.get('choices', item.get('options', [])),
                        'answer': item.get('answer', item.get('target', '')),
                    })
        
        self._split_data(all_data)
    
    def _split_data(self, all_data: List[Dict]) -> None:
        """Split data into train/valid/test."""
        if not all_data:
            self._load_sample_data()
            return
            
        random.shuffle(all_data)
        self.train_data = all_data[:self.train_size]
        self.valid_data = all_data[self.train_size:self.train_size + self.valid_size]
        self.test_data = all_data[self.train_size + self.valid_size:
                                   self.train_size + self.valid_size + self.test_size]
    
    def _load_sample_data(self) -> None:
        """Load built-in sample problems for demo."""
        sample_data = [
            # BBH-style logical deduction
            {
                "question": "The following paragraphs each describe a set of three objects arranged in a fixed order. The statements are logically consistent within each paragraph.\n\nOn a branch, there are three birds: a robin, a blue jay, and a crow. The blue jay is to the right of the crow. The robin is to the right of the blue jay.\n\nWhich of the following is true?",
                "choices": [
                    "A) The robin is the leftmost",
                    "B) The blue jay is the leftmost",
                    "C) The crow is the leftmost"
                ],
                "answer": "C"
            },
            # MMLU-style science question
            {
                "question": "In a double-slit experiment, if the wavelength of light is doubled while keeping the slit separation constant, what happens to the spacing between interference fringes?",
                "choices": [
                    "A) It doubles",
                    "B) It halves",
                    "C) It quadruples",
                    "D) It remains the same"
                ],
                "answer": "A"
            },
            # ARC-style reasoning
            {
                "question": "A student places a drop of pond water on a glass slide and observes it under a microscope. Which of the following would indicate that the sample contains living organisms?",
                "choices": [
                    "A) The presence of green color",
                    "B) Self-propelled movement",
                    "C) Large cell size",
                    "D) Irregular shapes"
                ],
                "answer": "B"
            },
            # BBH-style tracking shuffled objects
            {
                "question": "Alice, Bob, and Claire are playing a game. At the start of the game, they are each holding a ball: Alice has a blue ball, Bob has a red ball, and Claire has a green ball.\n\nAs the game progresses, pairs of players trade balls. First, Alice and Bob swap balls. Then, Bob and Claire swap balls. Finally, Alice and Claire swap balls.\n\nAt the end of the game, what ball does Bob have?",
                "choices": [
                    "A) Blue ball",
                    "B) Red ball", 
                    "C) Green ball"
                ],
                "answer": "C"
            },
            # MMLU-style history
            {
                "question": "The Treaty of Westphalia (1648) is often cited as establishing which principle in international relations?",
                "choices": [
                    "A) Collective security",
                    "B) State sovereignty",
                    "C) Free trade",
                    "D) Human rights"
                ],
                "answer": "B"
            },
            # Logic puzzle
            {
                "question": "If all Bloops are Razzies and all Razzies are Lazzies, then which statement must be true?",
                "choices": [
                    "A) All Lazzies are Bloops",
                    "B) All Bloops are Lazzies",
                    "C) Some Lazzies are Razzies",
                    "D) All Razzies are Bloops"
                ],
                "answer": "B"
            },
            # MMLU-style math
            {
                "question": "What is the derivative of f(x) = x³ - 3x² + 2x - 1?",
                "choices": [
                    "A) 3x² - 6x + 2",
                    "B) 3x² - 6x",
                    "C) x² - 3x + 2",
                    "D) 3x³ - 6x² + 2x"
                ],
                "answer": "A"
            },
            # BBH-style causal judgment
            {
                "question": "A machine produces widgets at a rate of 100 per hour. The machine breaks down and is repaired, after which it produces widgets at 80 per hour. What is the cause of the reduced production rate?",
                "choices": [
                    "A) The repair was incomplete",
                    "B) The machine is now less efficient",
                    "C) Both A and B are possible",
                    "D) The production rate should be the same"
                ],
                "answer": "C"
            },
            # ARC-style earth science
            {
                "question": "Which layer of Earth's atmosphere contains the ozone layer that protects us from ultraviolet radiation?",
                "choices": [
                    "A) Troposphere",
                    "B) Stratosphere",
                    "C) Mesosphere",
                    "D) Thermosphere"
                ],
                "answer": "B"
            },
            # BBH-style navigate
            {
                "question": "If you follow these instructions, where do you end up relative to where you started?\n\nTurn left. Take 3 steps. Turn around. Take 5 steps. Turn left. Take 2 steps.",
                "choices": [
                    "A) 2 steps to the left and 2 steps forward",
                    "B) 2 steps to the right and 2 steps forward",
                    "C) 2 steps forward and 2 steps to the left",
                    "D) 2 steps to the left and 2 steps back"
                ],
                "answer": "A"
            },
            # MMLU-style biology
            {
                "question": "Which organelle is responsible for producing ATP through cellular respiration?",
                "choices": [
                    "A) Nucleus",
                    "B) Ribosome",
                    "C) Mitochondria",
                    "D) Golgi apparatus"
                ],
                "answer": "C"
            },
            # ARC-style chemistry
            {
                "question": "When sodium (Na) reacts with chlorine (Cl₂), what type of bond is formed?",
                "choices": [
                    "A) Covalent bond",
                    "B) Ionic bond",
                    "C) Metallic bond",
                    "D) Hydrogen bond"
                ],
                "answer": "B"
            },
            # BBH-style temporal sequences
            {
                "question": "The following events occurred: Jane arrived before Mark. Sarah arrived after Mark but before Jane left. Tom arrived after Sarah left.\n\nWhich of the following is a possible order of arrivals?",
                "choices": [
                    "A) Jane, Mark, Sarah, Tom",
                    "B) Mark, Jane, Sarah, Tom",
                    "C) Jane, Sarah, Mark, Tom",
                    "D) Mark, Jane, Tom, Sarah"
                ],
                "answer": "A"
            },
            # MMLU-style economics
            {
                "question": "According to the law of demand, when the price of a good increases, what typically happens to the quantity demanded?",
                "choices": [
                    "A) It increases",
                    "B) It decreases",
                    "C) It remains the same",
                    "D) It becomes unpredictable"
                ],
                "answer": "B"
            },
            # BBH-style hyperbaton
            {
                "question": "Which sentence best describes a hyperbaton?",
                "choices": [
                    "A) An exaggerated statement",
                    "B) Words in an unusual order for emphasis",
                    "C) A comparison using 'like' or 'as'",
                    "D) A statement that contradicts itself"
                ],
                "answer": "B"
            },
            # ARC-style physics
            {
                "question": "A ball is thrown straight up into the air. At the highest point of its path, what is its velocity?",
                "choices": [
                    "A) Maximum",
                    "B) Zero",
                    "C) Equal to initial velocity",
                    "D) Cannot be determined"
                ],
                "answer": "B"
            },
        ]
        
        # Duplicate to meet size requirements
        multiplier = max(
            (self.train_size // len(sample_data)) + 1,
            (self.valid_size // len(sample_data)) + 1,
            (self.test_size // len(sample_data)) + 1,
        )
        extended_data = sample_data * multiplier
        random.shuffle(extended_data)
        
        self.train_data = extended_data[:self.train_size]
        self.valid_data = extended_data[self.train_size:self.train_size + self.valid_size]
        self.test_data = extended_data[self.train_size + self.valid_size:
                                        self.train_size + self.valid_size + self.test_size]
    
    def format_question(self, item: Dict) -> str:
        """
        Format a question with its choices for display.
        
        Args:
            item: Dict with 'question' and 'choices' keys
            
        Returns:
            Formatted question string
        """
        question = item['question']
        choices = item.get('choices', [])
        
        # Format choices if they don't already have labels
        formatted_choices = []
        for i, choice in enumerate(choices):
            if not choice.strip().startswith(('A)', 'B)', 'C)', 'D)', 'A.', 'B.', 'C.', 'D.')):
                label = chr(65 + i)  # A, B, C, D...
                formatted_choices.append(f"{label}) {choice}")
            else:
                formatted_choices.append(choice)
        
        return f"{question}\n\n" + "\n".join(formatted_choices)
    
    def extract_answer(self, text: str) -> Optional[str]:
        """
        Extract the answer letter from model output.
        
        Looks for patterns like:
        - "Answer: A"
        - "The answer is A"
        - "(A)"
        - Just "A" at the end
        """
        text = text.strip()
        
        # Pattern 1: "Answer: X" or "Answer is X"
        match = re.search(r'[Aa]nswer\s*(?:is|:)?\s*\(?([A-Da-d])\)?', text)
        if match:
            return match.group(1).upper()
        
        # Pattern 2: Parenthesized answer like "(A)" or "[A]"
        match = re.search(r'[\(\[]([A-Da-d])[\)\]]', text)
        if match:
            return match.group(1).upper()
        
        # Pattern 3: Just the letter at the end of the text
        match = re.search(r'\b([A-Da-d])\s*$', text)
        if match:
            return match.group(1).upper()
        
        # Pattern 4: "option A" or "choice A"
        match = re.search(r'(?:option|choice)\s+([A-Da-d])', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # Pattern 5: First standalone letter A-D in the last line
        last_line = text.split('\n')[-1]
        match = re.search(r'\b([A-Da-d])\b', last_line)
        if match:
            return match.group(1).upper()
        
        return None
    
    def evaluate(self, prediction: str, ground_truth: str) -> bool:
        """
        Check if prediction matches ground truth.
        
        Args:
            prediction: Model's extracted answer (e.g., "A")
            ground_truth: Correct answer (e.g., "A")
            
        Returns:
            True if correct, False otherwise
        """
        if not prediction or not ground_truth:
            return False
        
        # Normalize both to uppercase single letter
        pred = prediction.strip().upper()
        truth = ground_truth.strip().upper()
        
        # Handle cases where answer might be full text like "A) Paris"
        if pred and pred[0] in 'ABCD':
            pred = pred[0]
        if truth and truth[0] in 'ABCD':
            truth = truth[0]
        
        return pred == truth


class BBHTask(MultipleChoiceTask):
    """BIG-Bench Hard specific task."""
    
    def __init__(self, data_dir: Optional[str] = None, **kwargs):
        super().__init__(data_dir=data_dir, benchmark_type='bbh', **kwargs)
        self.name = "BBH"


class ARCTask(MultipleChoiceTask):
    """AI2 Reasoning Challenge specific task."""
    
    def __init__(self, data_dir: Optional[str] = None, **kwargs):
        super().__init__(data_dir=data_dir, benchmark_type='arc', **kwargs)
        self.name = "ARC"


class MMLUTask(MultipleChoiceTask):
    """MMLU specific task."""
    
    def __init__(self, data_dir: Optional[str] = None, subject: str = 'abstract_algebra', **kwargs):
        super().__init__(data_dir=data_dir, benchmark_type='mmlu', subject=subject, **kwargs)
        self.name = f"MMLU-{subject}"
