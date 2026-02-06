# 📚 CFPO - Controllable Feedback-based Prompt Optimization

A comprehensive, automated prompt optimization framework that uses **beam search** and **LLM-powered mutations** to iteratively improve prompts for better task performance.

---

## 🎯 Overview

CFPO takes an initial prompt and automatically improves it through multiple rounds of:

1. **Mutation** - Generate prompt variations using different strategies
2. **Evaluation** - Test each variant on real examples
3. **Selection** - Keep the best-performing prompts (beam search)

The result is an optimized prompt that achieves higher accuracy on your target task.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              MAIN.PY                                      │
│                         (Entry Point)                                     │
│  • Parses CLI arguments                                                   │
│  • Sets up logging                                                        │
│  • Creates model, task, mutators                                          │
│  • Runs optimizer                                                         │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                            OPTIMIZER.PY                                   │
│                      (Core Beam Search Loop)                              │
│  • Maintains "beam" of best prompts                                       │
│  • Each round: generate → evaluate → select top-K                         │
│  • Saves checkpoints and final results                                    │
└──────────────────────────────────────────────────────────────────────────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌────────────────┐    ┌────────────────┐    ┌────────────────────────┐
│   MUTATORS/    │    │    MODELS/     │    │        TASKS/          │
│ (Generate new  │    │ (LLM backends) │    │   (Benchmark data)     │
│   prompts)     │    │                │    │                        │
├────────────────┤    ├────────────────┤    ├────────────────────────┤
│• MonteCarlo    │    │• OllamaModel   │    │• GSM8KTask             │
│• FormatMutator │    │• HuggingFace   │    │  (Math word problems)  │
│• CaseDiagnosis │    │                │    │                        │
└────────────────┘    └────────────────┘    └────────────────────────┘
                                    │
                                    ▼
                      ┌────────────────────────┐
                      │       PROMPTS/         │
                      │  (Prompt structure)    │
                      ├────────────────────────┤
                      │• Prompt class          │
                      │• PromptHistory         │
                      └────────────────────────┘
```

---

## 📁 Project Structure

```
cfpo_simple/
├── run.sh              # 🚀 One-click run script
├── main.py             # Entry point & CLI argument parsing
├── optimizer.py        # Core beam search optimization loop
├── requirements.txt    # Python dependencies
│
├── models/             # LLM backends
│   ├── base.py         # Abstract LLM interface
│   ├── ollama.py       # Ollama (local, free)
│   └── huggingface.py  # HuggingFace API (free tier)
│
├── prompts/            # Prompt management
│   └── prompt.py       # Prompt class + PromptHistory
│
├── tasks/              # Benchmark tasks
│   ├── base.py         # Abstract Task interface
│   └── gsm8k.py        # GSM8K math word problems
│
├── mutators/           # Mutation strategies
│   ├── base.py         # Abstract Mutator interface
│   ├── monte_carlo.py  # Random exploration
│   ├── format_mutator.py   # Format/structure changes
│   └── case_diagnosis.py   # Error-based learning
│
└── results/            # Output directory
    └── run_TIMESTAMP/  # Each run's results
        ├── log.txt
        ├── checkpoint_round_N.json
        ├── final_results.json
        └── best_prompt.txt
```

---

## 📖 Detailed Component Documentation

### 1. `main.py` - Entry Point

**Purpose:** Command-line interface and orchestration

**What it does:**

- Parses CLI arguments (model type, rounds, beam size, data sizes)
- Sets up dual logging (console + file)
- Creates the initial structured prompt
- Initializes LLM model, task, and mutators
- Runs the optimization loop
- Displays and saves final results

**CLI Arguments:**

| Argument       | Default                   | Description                              |
| -------------- | ------------------------- | ---------------------------------------- |
| `--model`      | `ollama`                  | LLM backend (`ollama` or `huggingface`)  |
| `--model-name` | `phi` / `Llama-3.1-8B`    | Specific model to use                    |
| `--rounds`     | `3`                       | Number of optimization rounds            |
| `--beam-size`  | `2`                       | Number of best prompts to keep per round |
| `--train-size` | `5`                       | Training examples per round              |
| `--valid-size` | `5`                       | Validation examples for scoring          |
| `--output-dir` | `./results/run_TIMESTAMP` | Where to save results                    |

**Initial Prompt Structure:**

```python
Prompt(
    task_instruction="Solve the following math word problem step by step.",
    task_detail="Read the problem carefully. Identify the key numbers...",
    output_format="End your response with 'The answer is: [NUMBER]'...",
    example_hinter="Here is an example:",
    examples="Q: John has 5 apples...\nA: John starts with 5...",
    cot_hinter="Let's solve this step by step:",
)
```

---

### 2. `optimizer.py` - Core Optimization Engine

**Purpose:** Implements beam search over the prompt space

**Algorithm:**

```
1. START with initial prompt
2. EVALUATE initial prompt on validation set → get baseline score

3. FOR each round (1 to num_rounds):
   │
   ├── GENERATE: For each prompt in beam:
   │   └── Apply each mutator to create variations
   │
   ├── COLLECT: Gather all candidates (new + current beam)
   │
   ├── DEDUPLICATE: Remove identical prompts
   │
   ├── EVALUATE: Score each candidate on validation set
   │   └── accuracy = correct_predictions / total_examples
   │
   ├── SELECT: Keep top-K prompts (beam search)
   │
   └── CHECKPOINT: Save current state to JSON

4. RETURN best prompt found across all rounds
```

**Key Methods:**

| Method                                   | Purpose                                                       |
| ---------------------------------------- | ------------------------------------------------------------- |
| `run(init_prompt)`                       | Main optimization loop, returns (best_prompt, score, history) |
| `_evaluate_prompt(prompt)`               | Tests prompt on validation data, returns accuracy (0.0-1.0)   |
| `_save_checkpoint(round, beam, history)` | Saves round state to JSON file                                |
| `_save_results(prompt, score, history)`  | Saves final results and best prompt                           |

**Evaluation Process:**

```python
for each validation example:
    1. full_prompt = prompt.render(question)  # Combine prompt + question
    2. prediction = llm.generate(full_prompt)  # Get model response
    3. is_correct = task.evaluate(prediction, ground_truth)  # Check answer

accuracy = correct_count / total_count
```

---

### 3. `prompts/prompt.py` - Prompt Structure

**Purpose:** Represents and manages structured prompts

#### The `Prompt` Class

A prompt is composed of **6 modular components**:

| Component          | Purpose                         | Example                               |
| ------------------ | ------------------------------- | ------------------------------------- |
| `task_instruction` | Main task description           | "Solve the math problem step by step" |
| `task_detail`      | Additional constraints/guidance | "Read carefully, show your work"      |
| `output_format`    | How to format the answer        | "End with 'The answer is: [NUMBER]'"  |
| `example_hinter`   | Introduces few-shot examples    | "Here is an example:"                 |
| `examples`         | Demonstration Q&A pairs         | "Q: John has 5 apples..."             |
| `cot_hinter`       | Chain-of-thought trigger        | "Let's solve this step by step:"      |

**Key Methods:**

| Method                                  | Purpose                                          |
| --------------------------------------- | ------------------------------------------------ |
| `render(question="")`                   | Combines all components into final prompt string |
| `get_component(key)`                    | Gets a component by uppercase key name           |
| `set_component(key, value)`             | Sets a component value                           |
| `generate(round, keys, values, action)` | Creates modified copy of prompt                  |
| `to_dict()` / `from_dict()`             | Serialization for saving/loading                 |

**Rendered Prompt Example:**

```
# Task
Solve the following math word problem step by step.

# Details
Read the problem carefully. Identify the key numbers and operations needed.

# Output Format
End your response with 'The answer is: [NUMBER]'

Here is an example:
Q: John has 5 apples. He buys 3 more. How many does he have?
A: Total = 5 + 3 = 8. The answer is: 8

Let's solve this step by step:

# Question
[USER'S QUESTION HERE]
```

#### The `PromptHistory` Class

**Purpose:** Tracks all prompts tried during optimization

**Tracks:**

- List of (round, prompt, score) tuples
- Current best prompt and score
- Automatically updates best when new high score found

**Methods:**

- `add(prompt, score, round)` - Record a new prompt evaluation
- `get_best()` - Returns (best_prompt, best_score)
- `summary()` - Returns formatted history string

---

### 4. Mutators - Prompt Variation Strategies

Mutators are **search operators** that explore the prompt space by generating variations.

#### 4a. `mutators/base.py` - Abstract Interface

**Base Class:** All mutators inherit from `Mutator`

**Required Method:**

```python
def mutate(prompt, num_mutations, temperature, round) -> List[Prompt]
```

**Modifiable Components:**

```python
COMPONENT_KEYS = [
    'TASK_INSTRUCTION',
    'TASK_DETAIL',
    'OUTPUT_FORMAT',
    'EXAMPLES',
    'COT_HINTER',
]
```

**Helper Method:**

- `_get_meta_prompt_header(prompt)` - Creates context for LLM when asking it to improve prompts

---

#### 4b. `mutators/monte_carlo.py` - Random Exploration

**Strategy:** Randomly select and modify prompt components using LLM-generated variations

**Algorithm:**

```
FOR each mutation to generate:
    1. Randomly select 1-2 components to modify
    2. FOR each selected component:
       - Ask LLM: "Create a DIFFERENT version of this"
       - Use the generated variation
    3. Create new prompt with modifications
    4. Return if different from original
```

**Two Variation Types:**

| Method              | Purpose                                             |
| ------------------- | --------------------------------------------------- |
| `_vary_component()` | Rephrases any component (instruction, format, etc.) |
| `_vary_examples()`  | Adds, removes, or modifies few-shot examples        |

**Example LLM Prompt for Variation:**

```
You are an expert prompt engineer...

Please create a DIFFERENT version of the TASK_INSTRUCTION component.
Keep the same meaning but vary the wording, structure, or style.

Current TASK_INSTRUCTION:
"""Solve the following math word problem step by step."""

Your varied TASK_INSTRUCTION (just the content, no explanation):
```

---

#### 4c. `mutators/format_mutator.py` - Structural Changes

**Strategy:** Changes HOW the prompt is formatted, not WHAT it says

**Available Formats:**

| Format           | Style              | Example                                       |
| ---------------- | ------------------ | --------------------------------------------- |
| `markdown`       | Uses `#` headers   | `# Task\nSolve the problem...`                |
| `xml`            | Uses XML tags      | `<task><instruction>...</instruction></task>` |
| `plain`          | No formatting      | Just plain text                               |
| `structured`     | Section delimiters | `=== TASK ===\n...`                           |
| `conversational` | Natural language   | "I need your help with a task..."             |

**Why This Matters:** Different LLMs respond better to different prompt formats!

**Algorithm:**

```
1. Get list of available formats (excluding current)
2. For each format variation requested:
   - Apply format template to prompt content
   - Store format function for rendering
3. Return list of reformatted prompts
```

---

#### 4d. `mutators/case_diagnosis.py` - Error-Based Learning

**Strategy:** Learn from mistakes to improve the prompt

**Algorithm:**

```
1. Run current prompt on training examples
2. Identify FAILED cases (incorrect predictions)
3. Build diagnosis prompt with:
   - Current prompt structure
   - Failed examples with expected vs. actual
   - Some correct examples for contrast
4. Ask LLM: "Why did these fail? How should we fix the prompt?"
5. Parse suggested improvements
6. Apply improvements to create new prompt
```

**Key Insight:** This is **feedback-driven optimization** - the prompt learns from its own errors!

**Diagnosis Prompt Template:**

```
You are an expert prompt engineer...

Current prompt:
- Task Instruction: ...
- Output Format: ...

Error Cases:
- Question: Janet's ducks lay 16 eggs...
- Expected: 18
- Got: 14

Why might the prompt have caused this error?
What specific changes would fix it?
```

---

### 5. Models - LLM Backends

#### 5a. `models/base.py` - Abstract Interface

```python
class LLMModel(ABC):
    def generate(prompt: str, temperature: float) -> str
    def generate_batch(prompts: List[str], temperature: float) -> List[str]
```

**Parameters:**

- `model_name` - Model identifier
- `max_tokens` - Maximum response length (default: 2048)
- `temperature` - Creativity level (0=deterministic, 1=creative)

---

#### 5b. `models/ollama.py` - Local LLM (Free)

**What it is:** Connects to Ollama server running locally

**Features:**

- Uses REST API at `localhost:11434`
- Supports any Ollama model: phi, llama2, mistral, codellama, etc.
- 5-minute timeout for CPU inference
- Connection check on startup

**Usage:**

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull phi

# Start server
ollama serve

# Run CFPO
python main.py --model ollama --model-name phi
```

---

#### 5c. `models/huggingface.py` - Cloud LLM (Free API)

**What it is:** Uses HuggingFace Inference Providers API

**Features:**

- Multiple backend providers (Together AI, Fireworks, Groq, etc.)
- Automatic rate limiting and retry logic
- Model loading wait handling (503 responses)
- Chat completion format

**Supported Models:**

- `meta-llama/Llama-3.1-8B-Instruct` (default)
- `Qwen/Qwen2.5-7B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`

**Usage:**

```bash
# Get free token at: https://huggingface.co/settings/tokens
export HF_API_TOKEN=your_token_here

# Run CFPO
python main.py --model huggingface
```

---

### 6. Tasks - Benchmark Definitions

#### 6a. `tasks/base.py` - Abstract Interface

```python
class Task(ABC):
    def _load_data() -> None           # Load train/valid/test splits
    def evaluate(pred, truth) -> bool  # Check if prediction is correct
    def extract_answer(text) -> str    # Parse answer from model output
```

**Data Splits:**

- `train_data` - Used by CaseDiagnosisMutator to find errors
- `valid_data` - Used to score prompts during optimization
- `test_data` - Final evaluation (optional)

---

#### 6b. `tasks/gsm8k.py` - Math Word Problems

**GSM8K** = Grade School Math 8K - A dataset of 8,500 multi-step math word problems

**Sample Problem:**

```
Q: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast
   and bakes 4 into muffins. She sells the rest for $2 each.
   How much does she make?

A: 16 - 3 - 4 = 9 eggs. 9 * 2 = $18. The answer is: 18
```

**Answer Extraction Patterns:**

1. `"The answer is: 42"` or `"The answer is 42"`
2. `"#### 42"` (GSM8K native format)
3. Last number in text (fallback)

**Evaluation:**

- Compares extracted numbers as floats
- Tolerance: `abs(pred - truth) < 1e-5`
- Handles formatting differences (42, 42.0, 42.00)

**Built-in Sample Data:**

- 8 example problems included for demo purposes
- Automatically duplicates to meet requested size

---

## 🔄 How the Optimization Works

### Example Run (From Log)

```
Configuration:
├── Model: HuggingFace (Llama-3.1-8B-Instruct)
├── Rounds: 3
├── Beam Size: 2
├── Validation Set: 3 examples
└── Mutators: MonteCarloMutator, FormatMutator

Optimization:
├── Initial prompt score: 100.00%
│
├── ROUND 1/3
│   ├── MonteCarloMutator: 2 variations
│   │   ├── Variation 1: modified COT_HINTER
│   │   └── Variation 2: modified TASK_INSTRUCTION
│   ├── FormatMutator: 2 variations (xml, plain)
│   ├── Total unique candidates: 3 (after deduplication)
│   ├── Evaluations:
│   │   ├── Candidate 1: 100.00% (monte_carlo)
│   │   ├── Candidate 2: 100.00% (monte_carlo)
│   │   └── Candidate 3: 100.00% (format_xml)
│   └── Beam: ['100.00%', '100.00%']
│
├── ROUND 2/3
│   ├── (2 beam prompts × 2 mutators × 2 variations each)
│   ├── Total unique candidates: 6
│   ├── All scored 100.00%
│   └── Beam: ['100.00%', '100.00%']
│
├── ROUND 3/3
│   ├── Similar pattern
│   ├── Total unique candidates: 6
│   └── All scored 100.00%
│
└── FINAL RESULT: Best score 100.00%
```

**Note:** In this run, the initial prompt was already excellent for the small validation set, so no improvement was needed.

---

## 📊 Output Files

Each run creates a timestamped directory with:

| File                      | Contents                                 |
| ------------------------- | ---------------------------------------- |
| `log.txt`                 | Complete execution log with timestamps   |
| `checkpoint_round_N.json` | Beam state after each round              |
| `final_results.json`      | Best score, prompt dict, history summary |
| `best_prompt.txt`         | The winning prompt as plain text         |

**Example `final_results.json`:**

```json
{
  "best_score": 1.0,
  "best_prompt": {
    "TASK_INSTRUCTION": "Solve the following math word problem step by step.",
    "TASK_DETAIL": "Read the problem carefully...",
    "OUTPUT_FORMAT": "End your response with 'The answer is: [NUMBER]'",
    "EXAMPLES": "Q: John has 5 apples...",
    "COT_HINTER": "Let's solve this step by step:",
    "round": 2,
    "score": 1.0,
    "action": "monte_carlo"
  },
  "history_summary": "...",
  "num_prompts_tried": 15
}
```

---

## 🚀 Quick Start

### Option 1: Using Ollama (Local, Free)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull phi

# Start server (in separate terminal)
ollama serve

# Run optimization
cd cfpo_simple
pip install -r requirements.txt
python main.py --model ollama --model-name phi --rounds 3
```

### Option 2: Using HuggingFace (Cloud, Free)

```bash
# Set API token
export HF_API_TOKEN=your_token_here  # Get at: https://huggingface.co/settings/tokens

# Run optimization
cd cfpo_simple
pip install -r requirements.txt
python main.py --model huggingface --rounds 5 --beam-size 4
```

### Using the Shell Script

```bash
chmod +x run.sh
./run.sh              # Uses Ollama by default
./run.sh --hf         # Uses HuggingFace
```

---

## 🔧 Customization

### Adding a New Task

1. Create `tasks/my_task.py`:

```python
from .base import Task

class MyTask(Task):
    def _load_data(self):
        # Load your data into self.train_data, self.valid_data, self.test_data
        pass

    def extract_answer(self, text):
        # Parse answer from model output
        pass

    def evaluate(self, prediction, ground_truth):
        # Return True if correct
        pass
```

2. Import in `tasks/__init__.py`
3. Use in `main.py`

### Adding a New Mutator

1. Create `mutators/my_mutator.py`:

```python
from .base import Mutator

class MyMutator(Mutator):
    def mutate(self, prompt, num_mutations, temperature, round):
        # Generate and return list of modified prompts
        pass
```

2. Import in `mutators/__init__.py`
3. Add to mutators list in `main.py`

---

## 📈 Tips for Better Results

1. **Increase validation set size** - More examples = more reliable scores
2. **Use more rounds** - More exploration = better prompts
3. **Increase beam size** - Keep more candidates = avoid local optima
4. **Add CaseDiagnosisMutator** - Learn from errors for targeted improvement
5. **Try different models** - Some models respond better to certain prompt styles

---

## 🎓 Key Concepts

| Term                       | Definition                                                  |
| -------------------------- | ----------------------------------------------------------- |
| **Beam Search**            | Keep top-K candidates at each step instead of just the best |
| **Mutator**                | Strategy for generating prompt variations                   |
| **Chain-of-Thought (CoT)** | Prompting technique that encourages step-by-step reasoning  |
| **Few-shot Learning**      | Including examples in the prompt to guide behavior          |
| **Meta-prompting**         | Using an LLM to improve prompts for an LLM                  |

---

## 📝 License

This is a simplified implementation of the CFPO framework for educational and research purposes.

---

## 🔗 References

- [GSM8K Dataset](https://github.com/openai/grade-school-math)
- [Ollama](https://ollama.com/)
- [HuggingFace Inference API](https://huggingface.co/docs/api-inference)
