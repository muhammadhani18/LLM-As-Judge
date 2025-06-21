# 🧠 LLM-as-Judge: Multi-LLM Evaluation Framework

This project evaluates code and text generation capabilities of multiple LLMs using a *judge LLM* that acts as a referee. It supports both single-prompt and batch evaluations, provides scoring on correctness, code quality, security, and style, and outputs detailed JSON results.

---

## 📐 Architecture Overview

![Alt text]("images/flow_diagram.png")


### CLI Interface
Users interact through a CLI and provide:
- A prompt or a batch of prompts
- Models to compare (e.g., `gpt-4`, `gpt-3.5-turbo`)
- Evaluation type: `code` or `text`
- Save flag to store results as `.json`

### Code Evaluation Pipeline
1. **Model Invocation**: Prompts are sent to selected models.
2. **Code Cleaning**: Removes Markdown formatting or extra text.
3. **Static Analysis**:
   - Syntax check using `py_compile`
   - Security analysis via `bandit`
4. **Style Analysis**:
   - Linting score via `pylint`
   - Cyclomatic complexity via `radon`
   - Style score is normalized for ranking
5. **Test Generation**: Judge LLM creates edge case test functions.
6. **Test Execution**: Pytest-style runner executes the tests.
7. **Score Aggregation**: Functionality + style are combined.

### Text Evaluation Pipeline
1. **Model Invocation**: Each model generates text for the prompt.
2. **Prompt Construction**: The judge model compares outputs.
3. **Judge LLM Scoring**: Assigns scores based on correctness and relevance.
4. **Score Parsing**: JSON format is parsed into results.

### Output Layer
- Results are saved as JSON in `evaluations/` if `--save` is used.
- Otherwise, results are printed to console.

---

## 📦 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure you have:
- Python 3.8+
- `bandit`, `pylint`, and `radon` installed

### 2. Set Environment Variables

Create a `.env` file with your OpenAI API key:

```env
OPENAI_API_KEY=sk-...
```

### 3. Single Prompt Evaluation

**Code Example:**
```bash
python -m llm_judge.cli "Write a function to validate emails" -m gpt-4 -m gpt-3.5-turbo -t code --save
```

**Text Example:**
```bash
python -m llm_judge.cli "Explain quantum computing in simple terms" -m gpt-4 -m gpt-3.5-turbo -t text --save
```

### 4. Batch Evaluation

Create a JSON file with multiple prompts:
```json
["Prompt 1", "Prompt 2", "Prompt 3"]
```

Run:
```bash
python -m llm_judge.cli dummy -m gpt-4 -m gpt-3.5-turbo -t code --batch-file path/to/batch.json --save
```

---

## 📊 Scoring Logic

| Component       | Description                                  | Weight |
|----------------|----------------------------------------------|--------|
| Test Pass Rate | Ratio of passing unit tests                  | 70%    |
| Security       | Bandit-reported issues penalize the score    | -10% per issue |
| Style Score    | Based on `pylint` + `radon` complexity        | 30%    |

Final Score = 0.7 × (Pass Rate − Penalty) + 0.3 × Style Score

---

## 📁 Output Format

Each evaluation result includes:

```json
{
  "prompt": "Write a Python function to...",
  "results": {
    "gpt-4": {
      "status": "evaluated",
      "score": 0.82,
      "tests": {"passed": 9, "failed": 1},
      "lint": {...},
      "style": {
        "pylint_score": 9.2,
        "average_complexity": 3.5,
        "style_score": 0.78
      }
    }
  },
  "winner": "gpt-4"
}
```

---

## 📌 Folder Structure

```
llm_judge/
├── cli.py
├── code_pipeline.py
├── text_pipeline.py
├── evaluator.py
├── evaluations/        # Stores JSON results
├── batch_code_prompts.json
├── batch_text_prompts.json
```

---

## 📄 License

MIT License