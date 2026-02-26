"""
GSM8K Task - Grade School Math Word Problems

GSM8K is a dataset of 8.5K math word problems requiring
multi-step reasoning to solve.
"""

import os
import re
import json
from typing import Optional, List, Dict, Any

from .base import Task


class GSM8KTask(Task):
    """
    GSM8K math reasoning benchmark.
    
    Example:
        Q: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast
           and bakes 4 into muffins. She sells the rest for $2 each.
           How much does she make?
        A: 16 - 3 - 4 = 9 eggs. 9 * 2 = $18. The answer is: 18
    """
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        train_size: int = 10,
        valid_size: int = 10,
        test_size: int = 10,
    ):
        super().__init__(
            name="GSM8K",
            data_dir=data_dir,
            train_size=train_size,
            valid_size=valid_size,
            test_size=test_size,
            answer_marker="The answer is:",
        )
    
    def _load_data(self) -> None:
        """Load GSM8K data from files or use built-in examples."""
        
        # Try loading from data_dir
        if self.data_dir and os.path.exists(self.data_dir):
            self._load_from_files()
            return
        
        # Use built-in sample data for demo
        self._load_sample_data()
    
    def _load_from_files(self) -> None:
        """Load from JSONL files."""
        try:
            train_file = os.path.join(self.data_dir, 'train.jsonl')
            test_file = os.path.join(self.data_dir, 'test.jsonl')
            
            all_train = []
            if os.path.exists(train_file):
                with open(train_file, 'r') as f:
                    all_train = [json.loads(line) for line in f]
            
            all_test = []
            if os.path.exists(test_file):
                with open(test_file, 'r') as f:
                    all_test = [json.loads(line) for line in f]
            
            self.train_data = all_train[:self.train_size]
            self.valid_data = all_test[:self.valid_size]
            self.test_data = all_test[self.valid_size:self.valid_size + self.test_size]
            
        except Exception as e:
            print(f"Error loading data: {e}, using sample data")
            self._load_sample_data()
    
    def _load_sample_data(self) -> None:
        """Load built-in sample problems - TRICKY problems designed to challenge LLMs."""
        sample_data = [
            # ===== TRICKY PROBLEMS - Common LLM failure modes =====
            
            # 1. TRAP: Misleading "per" - many models miss the double conversion
            {
                "question": "A car travels at 60 miles per hour. How many feet does it travel in 30 seconds? (1 mile = 5280 feet)",
                "answer": "60 mph = 60 miles/hour = 60*5280 feet/hour = 316800 feet/hour. Per second: 316800/3600 = 88 feet/second. In 30 seconds: 88*30 = 2640 feet. The answer is: 2640"
            },
            # 2. TRAP: Order of operations with percentages
            {
                "question": "A shirt costs $80. It's marked down 25%, then that price is marked down another 20%. What's the final price?",
                "answer": "After 25% off: 80 * 0.75 = $60. After 20% off: 60 * 0.80 = $48. The answer is: 48"
            },
            # 3. TRAP: "How many more" vs "how many"
            {
                "question": "Tom has 15 apples. Jerry has 9 apples. How many more apples does Tom have than Jerry?",
                "answer": "Difference = 15 - 9 = 6 apples. The answer is: 6"
            },
            # 4. TRAP: Inclusive vs exclusive counting (fence post)
            {
                "question": "A fence is 100 meters long. Posts are placed every 5 meters starting from one end. How many posts are needed?",
                "answer": "Number of gaps = 100/5 = 20. Number of posts = 20 + 1 = 21 (including both ends). The answer is: 21"
            },
            # 5. TRAP: Reading comprehension - some numbers are distractors
            {
                "question": "A bookshelf has 5 shelves. The top shelf has 12 books, the second shelf has 15 books. The store owner adds 8 books to the top shelf and removes 3 books from the second shelf. How many books are now on the top two shelves combined?",
                "answer": "Top shelf: 12 + 8 = 20 books. Second shelf: 15 - 3 = 12 books. Combined: 20 + 12 = 32 books. The answer is: 32"
            },
            # 6. TRAP: Negative/decrease confusion
            {
                "question": "A stock was worth $100. It dropped 20% on Monday, then rose 20% on Tuesday. What is the stock worth now?",
                "answer": "After Monday (20% drop): 100 * 0.80 = $80. After Tuesday (20% rise): 80 * 1.20 = $96. The answer is: 96"
            },
            # 7. TRAP: Units conversion chain
            {
                "question": "A recipe needs 2.5 cups of flour. You only have a tablespoon measure. How many tablespoons do you need? (1 cup = 16 tablespoons)",
                "answer": "2.5 cups * 16 tablespoons/cup = 40 tablespoons. The answer is: 40"
            },
            # 8. TRAP: Average with different group sizes
            {
                "question": "In a class, 20 students scored an average of 75 on a test. 10 students scored an average of 90. What is the overall average for all 30 students?",
                "answer": "Total for first group: 20 * 75 = 1500. Total for second group: 10 * 90 = 900. Overall: (1500 + 900) / 30 = 2400 / 30 = 80. The answer is: 80"
            },
            # 9. TRAP: Working backwards
            {
                "question": "After giving away 1/3 of his marbles, Tim has 24 marbles left. How many marbles did Tim start with?",
                "answer": "24 marbles = 2/3 of original. Original = 24 / (2/3) = 24 * (3/2) = 36 marbles. The answer is: 36"
            },
            # 10. TRAP: Rate problem with meeting point
            {
                "question": "Two trains start 300 miles apart heading toward each other. Train A travels at 60 mph, Train B at 40 mph. How long until they meet?",
                "answer": "Combined speed = 60 + 40 = 100 mph. Time = 300 / 100 = 3 hours. The answer is: 3"
            },
            # 11. TRAP: Careful reading - who has what
            {
                "question": "Alice has 3 times as many stickers as Bob. Together they have 48 stickers. How many stickers does Alice have?",
                "answer": "Let Bob = x, then Alice = 3x. x + 3x = 48. 4x = 48. x = 12. Alice has 3 * 12 = 36 stickers. The answer is: 36"
            },
            # 12. TRAP: Remainder problem
            {
                "question": "A number when divided by 7 gives quotient 12 and remainder 5. What is the number?",
                "answer": "Number = 7 * 12 + 5 = 84 + 5 = 89. The answer is: 89"
            },
            # 13. TRAP: Profit percentage (on cost, not selling)
            {
                "question": "A merchant buys an item for $80 and sells it for $100. What is the profit percentage?",
                "answer": "Profit = 100 - 80 = $20. Profit percentage = (20/80) * 100 = 25%. The answer is: 25"
            },
            # 14. TRAP: Age problems - careful about "years ago"
            {
                "question": "Five years ago, a mother was 3 times as old as her daughter. Now the mother is 40 years old. How old is the daughter now?",
                "answer": "Five years ago, mother was 40 - 5 = 35. Five years ago, daughter was 35/3 = 11.67 years (but let's check: if 35 = 3 * daughter, daughter was 11.67). Now daughter is 11.67 + 5 = 16.67. Rounding: The answer is: 16.67"
            },
            # 15. TRAP: Ratio problems
            {
                "question": "The ratio of boys to girls in a class is 3:5. If there are 24 boys, how many students are in the class total?",
                "answer": "Boys/Girls = 3/5. If boys = 24, then 24/girls = 3/5. Girls = 24 * 5/3 = 40. Total = 24 + 40 = 64. The answer is: 64"
            },
            # 16. TRAP: Time zones
            {
                "question": "A flight leaves New York at 2:00 PM local time and arrives in Los Angeles at 5:00 PM local time. If New York is 3 hours ahead of LA, how long is the flight?",
                "answer": "When it's 2:00 PM in NY, it's 11:00 AM in LA. Flight arrives 5:00 PM LA time. Duration = 5:00 PM - 11:00 AM = 6 hours. The answer is: 6"
            },
            # 17. TRAP: Interest calculation
            {
                "question": "You invest $1000 at 10% simple interest per year. How much interest do you earn after 6 months?",
                "answer": "Simple interest = Principal * Rate * Time = 1000 * 0.10 * 0.5 = $50. The answer is: 50"
            },
            # 18. TRAP: Speed and distance with stops
            {
                "question": "A bus travels 120 km in 3 hours including a 30-minute rest stop. What was the bus's average moving speed?",
                "answer": "Moving time = 3 hours - 0.5 hours = 2.5 hours. Average moving speed = 120 / 2.5 = 48 km/h. The answer is: 48"
            },
            # 19. TRAP: Surface area vs volume
            {
                "question": "A cube has a surface area of 96 square cm. What is its volume?",
                "answer": "Surface area of cube = 6 * side². 96 = 6 * side². side² = 16. side = 4 cm. Volume = 4³ = 64 cubic cm. The answer is: 64"
            },
            # 20. TRAP: Probability common mistake
            {
                "question": "You flip a fair coin 3 times. What is the probability of getting exactly 2 heads? Express as a decimal.",
                "answer": "Total outcomes = 2³ = 8. Favorable (exactly 2 heads): HHT, HTH, THH = 3. Probability = 3/8 = 0.375. The answer is: 0.375"
            },
        ]
        
        # Shuffle to ensure varied difficulty across train/valid/test
        import random
        random.seed(42)  # Fixed seed for reproducibility
        shuffled_data = sample_data.copy()
        random.shuffle(shuffled_data)
        
        # Duplicate to meet size requirements
        multiplier = max(
            (self.train_size // len(shuffled_data)) + 1,
            (self.valid_size // len(shuffled_data)) + 1,
            (self.test_size // len(shuffled_data)) + 1,
        )
        extended_data = shuffled_data * multiplier
        
        self.train_data = extended_data[:self.train_size]
        self.valid_data = extended_data[self.train_size:self.train_size + self.valid_size]
        self.test_data = extended_data[self.train_size + self.valid_size:self.train_size + self.valid_size + self.test_size]
    
    def extract_answer(self, text: str) -> Optional[str]:
        """
        Extract numerical answer from model output.
        
        Looks for patterns like:
        - "The answer is: 42"
        - "The answer is 42"
        - "#### 42"
        """
        # Pattern 1: "The answer is: X"
        match = re.search(r'[Tt]he answer is:?\s*\$?([0-9,.-]+)', text)
        if match:
            return match.group(1).replace(',', '').strip()
        
        # Pattern 2: "#### X" (GSM8K format)
        match = re.search(r'####\s*\$?([0-9,.-]+)', text)
        if match:
            return match.group(1).replace(',', '').strip()
        
        # Pattern 3: Last number in text
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return numbers[-1]
        
        return None
    
    def evaluate(self, prediction: str, ground_truth: str) -> bool:
        """
        Check if predicted answer matches ground truth.
        
        Handles:
        - Floating point comparison
        - Different formats (42, 42.0, 42.00)
        """
        pred_answer = self.extract_answer(prediction)
        true_answer = self.extract_answer(ground_truth)
        
        if pred_answer is None or true_answer is None:
            return False
        
        try:
            # Compare as floats with tolerance
            pred_num = float(pred_answer)
            true_num = float(true_answer)
            return abs(pred_num - true_num) < 1e-5
        except ValueError:
            # Fall back to string comparison
            return pred_answer.strip() == true_answer.strip()
