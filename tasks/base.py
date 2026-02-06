"""
Base Task Interface

Defines the interface for all benchmark tasks.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import random


class Task(ABC):
    """
    Abstract base class for all tasks/benchmarks.
    
    A task defines:
    - Data loading (train/validation/test splits)
    - Evaluation logic (how to check if answer is correct)
    - Answer extraction (how to parse model output)
    """
    
    def __init__(
        self,
        name: str = "base",
        data_dir: Optional[str] = None,
        train_size: int = 10,
        valid_size: int = 10,
        test_size: int = 10,
        answer_marker: str = "The answer is:",
    ):
        """
        Initialize the task.

        Args:
            name: Task name
            data_dir: Path to data files
            train_size: Number of training examples
            valid_size: Number of validation examples
            test_size: Number of test examples
            answer_marker: String that precedes the answer
        """
        self.name = name
        self.data_dir = data_dir
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.answer_marker = answer_marker
        
        # Data splits
        self.train_data: List[Dict[str, Any]] = []
        self.valid_data: List[Dict[str, Any]] = []
        self.test_data: List[Dict[str, Any]] = []
        
        # Load data
        self._load_data()
    
    @abstractmethod
    def _load_data(self) -> None:
        """Load data into train/valid/test splits."""
        pass
    
    @abstractmethod
    def evaluate(self, prediction: str, ground_truth: str) -> bool:
        """
        Check if prediction matches ground truth.
        
        Args:
            prediction: Model's output
            ground_truth: Correct answer
            
        Returns:
            True if correct, False otherwise
        """
        pass
    
    @abstractmethod
    def extract_answer(self, text: str) -> Optional[str]:
        """
        Extract the answer from model output.
        
        Args:
            text: Model's full output
            
        Returns:
            Extracted answer or None if not found
        """
        pass
    
    def get_train_batch(self, batch_size: Optional[int] = None) -> List[Dict]:
        """Get a random batch of training examples."""
        size = batch_size or len(self.train_data)
        if len(self.train_data) <= size:
            return self.train_data
        return random.sample(self.train_data, size)
    
    def get_valid_data(self) -> List[Dict]:
        """Get validation data."""
        return self.valid_data
    
    def get_test_data(self) -> List[Dict]:
        """Get test data."""
        return self.test_data
    
    def evaluate_batch(
        self,
        predictions: List[str],
        ground_truths: List[str]
    ) -> Tuple[float, List[bool]]:
        """
        Evaluate a batch of predictions.
        
        Args:
            predictions: List of model outputs
            ground_truths: List of correct answers
            
        Returns:
            (accuracy, list of correct/incorrect for each)
        """
        results = [
            self.evaluate(pred, truth)
            for pred, truth in zip(predictions, ground_truths)
        ]
        accuracy = sum(results) / len(results) if results else 0.0
        return accuracy, results
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"train={len(self.train_data)}, "
            f"valid={len(self.valid_data)}, "
            f"test={len(self.test_data)})"
        )
