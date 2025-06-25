# Migration Guide: From Monolithic to Modular Architecture

This guide helps you understand the changes from the old monolithic structure to the new modular architecture in LLM Judge.

## 🔄 What Changed

### Old Structure (Monolithic)
```
llm_judge/
├── cli.py                    # CLI + all logic
├── code_pipeline.py          # 597 lines - everything mixed together
├── text_pipeline.py          # 245 lines - everything mixed together
├── evaluator.py              # Abstract base class
└── batch_*.json              # Batch prompts
```

### New Structure (Modular)
```
llm_judge/
├── config.py                 # Centralized configuration
├── evaluator.py              # Abstract base class (unchanged)
├── main.py                   # New main entry point
├── cli.py                    # Legacy CLI (still works)
├── utils/                    # Reusable utilities
│   ├── cost_tracker.py       # Cost tracking logic
│   ├── code_analysis.py      # Code analysis logic
│   ├── test_runner.py        # Test generation/execution
│   └── judge_analyzer.py     # Judge reasoning logic
├── pipelines/                # Clean pipeline classes
│   ├── base_pipeline.py      # Common functionality
│   ├── code_pipeline.py      # Code evaluation (simplified)
│   └── text_pipeline.py      # Text evaluation (simplified)
└── batch_prompts/            # Organized batch files
```

## 🚀 Benefits of the New Architecture

### 1. **Separation of Concerns**
- **Before**: All logic mixed in large files
- **After**: Each component has a single responsibility

### 2. **Reusability**
- **Before**: Code duplicated between pipelines
- **After**: Common utilities shared across components

### 3. **Maintainability**
- **Before**: Hard to find and fix issues
- **After**: Clear structure makes debugging easier

### 4. **Extensibility**
- **Before**: Adding features required modifying large files
- **After**: Easy to add new utilities or evaluation types

## 📋 Migration Steps

### Step 1: Update Imports

**Old way:**
```python
from llm_judge.code_pipeline import CodePipeline
from llm_judge.text_pipeline import TextPipeline
```

**New way:**
```python
from llm_judge.pipelines.code_pipeline import CodePipeline
from llm_judge.pipelines.text_pipeline import TextPipeline
```

### Step 2: Update CLI Usage

**Old way:**
```bash
python -m llm_judge.cli "prompt" -m gpt-4 -m gpt-3.5-turbo -t code
```

**New way:**
```bash
python -m llm_judge.main "prompt" -m gpt-4 -m gpt-3.5-turbo -t code
```

**Note**: The old CLI still works for backward compatibility.

### Step 3: Programmatic Usage

**Old way:**
```python
from llm_judge.code_pipeline import CodePipeline

pipeline = CodePipeline(models=["gpt-4", "gpt-3.5-turbo"])
result = pipeline.evaluate("Write a function...")
```

**New way:**
```python
from llm_judge.pipelines.code_pipeline import CodePipeline

pipeline = CodePipeline(models=["gpt-4", "gpt-3.5-turbo"])
result = pipeline.evaluate("Write a function...")
```

The API remains the same, but the import path changes.

## 🔧 New Features Available

### 1. **Cost Tracking**
```python
# Cost tracking is now automatic
result = pipeline.evaluate(prompt)
print(f"Total cost: ${result['cost_breakdown']['total']:.4f}")
```

### 2. **Custom Judge Models**
```python
# Specify a different judge model
pipeline = CodePipeline(
    models=["gpt-4", "gpt-3.5-turbo"],
    judge_model="gpt-4-turbo"
)
```

### 3. **Detailed Reasoning**
```python
# Get detailed analysis from judge
result = pipeline.evaluate(prompt)
judge_analysis = result['judge_analysis']
print(f"Winner reasoning: {judge_analysis['winner_reasoning']}")
```

### 4. **Configuration Management**
```python
from llm_judge.config import CODE_EVALUATION_WEIGHTS, COST_PER_1K_TOKENS

# Easily modify evaluation weights
print(f"Functionality weight: {CODE_EVALUATION_WEIGHTS['functionality']}")
```

## 🧪 Testing the Migration

### 1. **Run the Example Script**
```bash
python example_usage.py
```

This will test all the new modular components.

### 2. **Compare Results**
Run the same evaluation with both old and new versions to ensure results are identical.

### 3. **Check Cost Tracking**
Verify that cost tracking is working correctly:
```bash
python -m llm_judge.main "test prompt" -m gpt-4 -t text
```

## 🔍 What's Backward Compatible

### ✅ **Still Works**
- CLI interface (`python -m llm_judge.cli`)
- Basic pipeline usage
- Evaluation results format
- Batch processing
- Save functionality

### ⚠️ **Changed**
- Import paths for pipelines
- Internal implementation details
- Additional features available

### ❌ **Removed**
- Nothing removed - all functionality preserved

## 🐛 Troubleshooting

### Issue: Import Errors
**Error**: `ModuleNotFoundError: No module named 'llm_judge.pipelines'`

**Solution**: Make sure you're using the updated import paths:
```python
# ✅ Correct
from llm_judge.pipelines.code_pipeline import CodePipeline

# ❌ Incorrect
from llm_judge.code_pipeline import CodePipeline
```

### Issue: Missing Dependencies
**Error**: `ModuleNotFoundError: No module named 'radon'`

**Solution**: Install additional dependencies:
```bash
pip install bandit pylint radon python-dotenv
```

### Issue: Configuration Not Found
**Error**: `NameError: name 'COST_PER_1K_TOKENS' is not defined`

**Solution**: Import from the config module:
```python
from llm_judge.config import COST_PER_1K_TOKENS
```

## 📊 Performance Comparison

| Aspect | Old Version | New Version | Improvement |
|--------|-------------|-------------|-------------|
| Code Lines | 842 total | 600+ total | Better organized |
| File Size | 2 large files | 8 focused files | Easier to navigate |
| Maintainability | Low | High | Clear separation |
| Extensibility | Difficult | Easy | Modular design |
| Testing | Hard | Easy | Isolated components |

## 🎯 Next Steps

1. **Update your imports** to use the new modular structure
2. **Test your existing code** with the new architecture
3. **Explore new features** like cost tracking and detailed reasoning
4. **Consider contributing** to the modular codebase

## 🤝 Need Help?

If you encounter issues during migration:

1. Check this migration guide
2. Review the updated README.md
3. Run the example script to verify functionality
4. Open an issue with specific error details

The new modular architecture makes the codebase more maintainable, extensible, and easier to understand while preserving all existing functionality. 