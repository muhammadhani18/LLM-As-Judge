#!/usr/bin/env python3
"""
Example usage of the LLM Judge framework with the new modular architecture.
"""

from llm_judge.pipelines.code_pipeline import CodePipeline
from llm_judge.pipelines.text_pipeline import TextPipeline
from llm_judge.config import DEFAULT_MODELS

def example_code_evaluation():
    """Example of code evaluation using the modular pipeline."""
    print("🚀 Example: Code Evaluation")
    print("=" * 50)
    
    # Initialize the code pipeline
    pipeline = CodePipeline(models=DEFAULT_MODELS)
    
    # Define a prompt
    prompt = "Write a Python function to check if a string is a palindrome"
    
    # Run evaluation
    result = pipeline.evaluate(prompt)
    
    # Display results
    print(f"\n📊 Evaluation Results:")
    print(f"Winner: {result['winner']}")
    print(f"Cost Breakdown: {result['cost_breakdown']}")
    
    return result

def example_text_evaluation():
    """Example of text evaluation using the modular pipeline."""
    print("\n🚀 Example: Text Evaluation")
    print("=" * 50)
    
    # Initialize the text pipeline
    pipeline = TextPipeline(models=DEFAULT_MODELS)
    
    # Define a prompt
    prompt = "Explain the concept of machine learning in simple terms"
    
    # Run evaluation
    result = pipeline.evaluate(prompt)
    
    # Display results
    print(f"\n📊 Evaluation Results:")
    print(f"Winner: {result['winner']}")
    print(f"Cost Breakdown: {result['cost_breakdown']}")
    
    return result

def example_custom_models():
    """Example using custom models and judge."""
    print("\n🚀 Example: Custom Models")
    print("=" * 50)
    
    # Use different models
    models = ["gpt-4", "gpt-3.5-turbo"]
    judge_model = "gpt-4-turbo"
    
    # Initialize pipeline with custom judge
    pipeline = CodePipeline(models=models, judge_model=judge_model)
    
    # Define a prompt
    prompt = "Write a function to sort a list of numbers"
    
    # Run evaluation
    result = pipeline.evaluate(prompt)
    
    print(f"\n📊 Custom Evaluation Results:")
    print(f"Models used: {models}")
    print(f"Judge model: {judge_model}")
    print(f"Winner: {result['winner']}")
    
    return result

def example_cost_tracking():
    """Example demonstrating cost tracking features."""
    print("\n🚀 Example: Cost Tracking")
    print("=" * 50)
    
    pipeline = TextPipeline(models=["gpt-4", "gpt-3.5-turbo"])
    
    # Reset cost tracker
    pipeline.reset_cost_tracker()
    
    # Run multiple evaluations
    prompts = [
        "What is artificial intelligence?",
        "Explain blockchain technology",
        "Describe the benefits of renewable energy"
    ]
    
    total_cost = 0
    for i, prompt in enumerate(prompts, 1):
        print(f"\n📝 Evaluation {i}: {prompt[:50]}...")
        result = pipeline.evaluate(prompt)
        cost = result['cost_breakdown']['total']
        total_cost += cost
        print(f"   Cost: ${cost:.4f}")
    
    # Display total cost
    print(f"\n💰 Total Cost for {len(prompts)} evaluations: ${total_cost:.4f}")
    
    return total_cost

if __name__ == "__main__":
    print("🧠 LLM Judge Framework - Modular Architecture Examples")
    print("=" * 60)
    
    try:
        # Run examples
        code_result = example_code_evaluation()
        text_result = example_text_evaluation()
        custom_result = example_custom_models()
        total_cost = example_cost_tracking()
        
        print("\n✅ All examples completed successfully!")
        print(f"Total cost across all examples: ${total_cost:.4f}")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Make sure you have set your OPENAI_API_KEY environment variable.") 