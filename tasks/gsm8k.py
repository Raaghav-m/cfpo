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
        """Load built-in sample problems for demo - Grade 8 level complexity."""
        sample_data = [
            # Compound interest and exponential growth
            {
                "question": "Maria invests $5,000 in a savings account that earns 6% compound interest annually. After 3 years, she withdraws half the total amount and reinvests the rest at 8% for 2 more years. How much money does she have at the end of 5 years? Round to the nearest dollar.",
                "answer": "After 3 years: 5000 * (1.06)^3 = 5000 * 1.191016 = $5955.08. She withdraws half: 5955.08 / 2 = $2977.54. Remaining reinvested: 2977.54 * (1.08)^2 = 2977.54 * 1.1664 = $3472.60. The answer is: 3473"
            },
            # Systems of equations word problem
            {
                "question": "A movie theater sold 350 tickets for a show. Adult tickets cost $12 and child tickets cost $7. If the total revenue was $3,400, how many adult tickets were sold?",
                "answer": "Let a = adult tickets, c = child tickets. a + c = 350, so c = 350 - a. Revenue: 12a + 7c = 3400. Substitute: 12a + 7(350 - a) = 3400. 12a + 2450 - 7a = 3400. 5a = 950. a = 190. The answer is: 190"
            },
            # Quadratic application
            {
                "question": "A ball is thrown upward from a height of 4 feet with an initial velocity of 64 feet per second. Its height h after t seconds is given by h = -16t² + 64t + 4. What is the maximum height the ball reaches?",
                "answer": "Maximum occurs at t = -b/(2a) = -64/(2*-16) = 64/32 = 2 seconds. Height at t=2: h = -16(4) + 64(2) + 4 = -64 + 128 + 4 = 68 feet. The answer is: 68"
            },
            # Probability with combinations
            {
                "question": "A bag contains 5 red marbles, 4 blue marbles, and 3 green marbles. If you draw 2 marbles without replacement, what is the probability that both are red? Express as a percentage rounded to one decimal place.",
                "answer": "Total marbles = 12. P(first red) = 5/12. After drawing one red, P(second red) = 4/11. P(both red) = (5/12) * (4/11) = 20/132 = 5/33 = 0.1515... = 15.2%. The answer is: 15.2"
            },
            # Similar triangles and proportions
            {
                "question": "A 6-foot tall person stands 15 feet away from a streetlight. If the person's shadow is 9 feet long, how tall is the streetlight in feet?",
                "answer": "Using similar triangles: height/shadow = streetlight/(shadow + distance). 6/9 = h/(9+15). 6/9 = h/24. h = 6 * 24 / 9 = 144/9 = 16 feet. The answer is: 16"
            },
            # Rate and work problems
            {
                "question": "Pipe A can fill a tank in 6 hours. Pipe B can fill the same tank in 4 hours. Pipe C can drain the full tank in 8 hours. If all three pipes are open, how many hours will it take to fill the empty tank?",
                "answer": "Rate A = 1/6, Rate B = 1/4, Rate C = -1/8 (draining). Combined rate = 1/6 + 1/4 - 1/8 = 4/24 + 6/24 - 3/24 = 7/24. Time = 1 / (7/24) = 24/7 = 3.43 hours. The answer is: 3.43"
            },
            # Percentage change and markup
            {
                "question": "A store buys a jacket for $45 and marks it up by 80%. During a sale, the jacket is discounted by 25%. What is the sale price and what is the store's profit per jacket?",
                "answer": "Markup price = 45 * 1.80 = $81. Sale discount = 81 * 0.75 = $60.75. Profit = 60.75 - 45 = $15.75. The answer is: 15.75"
            },
            # Distance, rate, time with multiple legs
            {
                "question": "Sarah drives from City A to City B at 60 mph. She then drives from City B to City C at 40 mph. The total distance is 280 miles and the total time is 5.5 hours. What is the distance from City A to City B?",
                "answer": "Let d = distance A to B. Time for A to B = d/60. Time for B to C = (280-d)/40. Total: d/60 + (280-d)/40 = 5.5. Multiply by 120: 2d + 3(280-d) = 660. 2d + 840 - 3d = 660. -d = -180. d = 180 miles. The answer is: 180"
            },
            # Geometry - volume and surface area
            {
                "question": "A cylindrical water tank has a radius of 3 meters and a height of 8 meters. If 1 cubic meter of water weighs 1000 kg, how many metric tons of water can the tank hold when full? Use π = 3.14159.",
                "answer": "Volume = πr²h = 3.14159 * 9 * 8 = 226.19 cubic meters. Weight = 226.19 * 1000 = 226,190 kg = 226.19 metric tons. The answer is: 226.19"
            },
            # Sequences and series
            {
                "question": "The first term of a geometric sequence is 3 and the common ratio is 2. What is the sum of the first 8 terms?",
                "answer": "Sum of geometric series: S = a(r^n - 1)/(r - 1) = 3(2^8 - 1)/(2 - 1) = 3(256 - 1)/1 = 3 * 255 = 765. The answer is: 765"
            },
            # Multi-step algebra
            {
                "question": "Three consecutive even integers have a sum of 78. What is the product of the largest and smallest of these integers?",
                "answer": "Let the integers be n, n+2, n+4. Sum: n + (n+2) + (n+4) = 78. 3n + 6 = 78. 3n = 72. n = 24. Integers are 24, 26, 28. Product = 24 * 28 = 672. The answer is: 672"
            },
            # Trigonometry application
            {
                "question": "A ladder 13 meters long leans against a wall. The base of the ladder is 5 meters from the wall. How high up the wall does the ladder reach?",
                "answer": "Using Pythagorean theorem: a² + b² = c². 5² + h² = 13². 25 + h² = 169. h² = 144. h = 12 meters. The answer is: 12"
            },
            # Statistics - weighted average
            {
                "question": "In a class, 15 students scored an average of 72 on a test, and 25 students scored an average of 84. What is the average score for the entire class?",
                "answer": "Total points = 15*72 + 25*84 = 1080 + 2100 = 3180. Total students = 15 + 25 = 40. Average = 3180/40 = 79.5. The answer is: 79.5"
            },
            # Mixture problems
            {
                "question": "A chemist has 100 mL of a 40% acid solution. How many mL of pure acid must be added to create a 60% acid solution?",
                "answer": "Current acid = 0.40 * 100 = 40 mL. Let x = mL of pure acid added. (40 + x)/(100 + x) = 0.60. 40 + x = 0.60(100 + x). 40 + x = 60 + 0.60x. 0.40x = 20. x = 50 mL. The answer is: 50"
            },
            # Ratios and proportions complex
            {
                "question": "The ratio of boys to girls in a school is 3:4. If 24 more boys join the school, the ratio becomes 5:4. How many students were in the school originally?",
                "answer": "Let boys = 3x, girls = 4x. After: (3x + 24)/4x = 5/4. 4(3x + 24) = 5(4x). 12x + 96 = 20x. 96 = 8x. x = 12. Original: boys = 36, girls = 48. Total = 84. The answer is: 84"
            },
            # Function composition
            {
                "question": "If f(x) = 2x + 3 and g(x) = x² - 1, what is the value of f(g(4))?",
                "answer": "First find g(4) = 4² - 1 = 16 - 1 = 15. Then f(g(4)) = f(15) = 2(15) + 3 = 30 + 3 = 33. The answer is: 33"
            },
        ]
        
        # Duplicate to meet size requirements
        multiplier = max(
            (self.train_size // len(sample_data)) + 1,
            (self.valid_size // len(sample_data)) + 1,
            (self.test_size // len(sample_data)) + 1,
        )
        extended_data = sample_data * multiplier
        
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
