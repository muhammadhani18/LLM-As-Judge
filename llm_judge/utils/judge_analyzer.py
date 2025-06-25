"""
Judge analysis utilities for detailed reasoning and evaluation.
"""
import re
import json
from typing import Dict, Any
from openai import OpenAI
from ..config import DEFAULT_JUDGE_MODEL

client = OpenAI()

class JudgeAnalyzer:
    """Handles detailed analysis and reasoning from the judge model."""
    
    def __init__(self, judge_model: str = DEFAULT_JUDGE_MODEL):
        self.judge_model = judge_model
    
    def analyze_code_quality(self, prompt: str, generated: Dict[str, str], scores: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed analysis from judge LLM about code quality and reasoning."""
        # Build analysis prompt
        analysis_prompt = f"""
You are an expert code evaluator. Given the following prompt:

PROMPT:
{prompt}

And the following model responses with their evaluation results:

"""
        
        for name, code in generated.items():
            score_info = scores.get(name, {})
            analysis_prompt += f"""
{name}:
CODE:
{code}

EVALUATION RESULTS:
- Status: {score_info.get('status', 'unknown')}
- Final Score: {score_info.get('score', 0)}
- Tests: {score_info.get('tests', {})}
- Style Score: {score_info.get('style', {}).get('style_score', 0)}
- Pylint Score: {score_info.get('style', {}).get('pylint_score', 0)}
- Complexity: {score_info.get('style', {}).get('average_complexity', 0)}
- Security Issues: {score_info.get('lint', {}).get('security_issues', 0)}

"""

        analysis_prompt += """
Analyze each code submission and provide detailed reasoning about:
1. Code quality and readability
2. Test performance and edge case handling
3. Security considerations
4. Style and maintainability
5. Overall strengths and weaknesses

Return your analysis in this exact JSON format:

{
  "gpt-4": {
    "code_quality": "Detailed analysis of code structure and readability",
    "test_performance": "Analysis of how well the code handles edge cases",
    "security_analysis": "Security considerations and potential issues",
    "style_assessment": "Code style, maintainability, and best practices",
    "strengths": "List of specific strengths",
    "weaknesses": "List of specific weaknesses"
  },
  "gpt-3.5-turbo": {
    "code_quality": "Detailed analysis of code structure and readability",
    "test_performance": "Analysis of how well the code handles edge cases", 
    "security_analysis": "Security considerations and potential issues",
    "style_assessment": "Code style, maintainability, and best practices",
    "strengths": "List of specific strengths",
    "weaknesses": "List of specific weaknesses"
  },
  "winner": "model_name",
  "winner_reasoning": "Detailed explanation of why this code is the best",
  "loser_reasoning": "Detailed explanation of why the losing code fell short"
}

Only return valid JSON.
"""

        try:
            response = client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0
            )
            
            judgment = response.choices[0].message.content
            print("\n🧠 Judge Analysis Output:\n", judgment)
            
            return self._extract_json(judgment)
        except Exception as e:
            print(f"Error getting judge analysis: {e}")
            return {}
    
    def analyze_text_quality(self, prompt: str, responses: Dict[str, str]) -> str:
        """Build judge prompt for text evaluation."""
        instruction = f"""
            You are an expert evaluator. Given the following prompt:

            PROMPT:
            {prompt}

            And the following model responses:

            """
        for name, response in responses.items():
            instruction += f"{name}:\n{response}\n\n"

        instruction += """
            Score each response on a scale of 1–5 for:
            - Correctness: How accurate and factually correct is the response?
            - Relevance: How well does the response address the prompt?

            Then select the best response and provide detailed reasoning.

            Return the output in **this exact JSON format**:

            {
            "gpt-4": { 
                "correctness": 5, 
                "relevance": 5,
                "reasoning": "Detailed explanation of strengths and weaknesses"
            },
            "gpt-3.5-turbo": { 
                "correctness": 3, 
                "relevance": 4,
                "reasoning": "Detailed explanation of strengths and weaknesses"
            },
            "winner": "gpt-4",
            "winner_reasoning": "Detailed explanation of why this response is the best",
            "loser_reasoning": "Detailed explanation of why the losing response fell short"
            }

            For the reasoning fields:
            - Be specific about what makes each response good or bad
            - Point out specific strengths and weaknesses
            - Explain how well each response addresses the prompt
            - Highlight any factual errors, logical gaps, or missed points

            Only return valid JSON.
        """
        return instruction.strip()
    
    def display_code_reasoning(self, scores: Dict[str, Any], judge_analysis: Dict[str, Any]):
        """Display detailed reasoning from the judge in a readable format."""
        print("\n" + "="*60)
        print("🏆 JUDGE'S DETAILED CODE ANALYSIS")
        print("="*60)
        
        winner = judge_analysis.get("winner")
        winner_reasoning = judge_analysis.get("winner_reasoning", "")
        loser_reasoning = judge_analysis.get("loser_reasoning", "")
        
        # Display individual model analysis
        for model_name, model_data in judge_analysis.items():
            if model_name in ["winner", "winner_reasoning", "loser_reasoning"]:
                continue
                
            print(f"\n📊 {model_name.upper()} ANALYSIS:")
            
            # Show evaluation metrics
            score_info = scores.get(model_name, {})
            if score_info:
                print(f"   Final Score: {score_info.get('score', 'N/A')}")
                tests = score_info.get('tests', {})
                print(f"   Tests: {tests.get('passed', 0)} passed, {tests.get('failed', 0)} failed")
                style = score_info.get('style', {})
                print(f"   Style Score: {style.get('style_score', 'N/A')}")
                print(f"   Pylint: {style.get('pylint_score', 'N/A')}/10")
                print(f"   Complexity: {style.get('average_complexity', 'N/A')}")
                print(f"   Security Issues: {score_info.get('lint', {}).get('security_issues', 'N/A')}")
            
            # Show detailed analysis
            if isinstance(model_data, dict):
                for category, analysis in model_data.items():
                    if analysis and category in ['code_quality', 'test_performance', 'security_analysis', 'style_assessment', 'strengths', 'weaknesses']:
                        print(f"   {category.replace('_', ' ').title()}: {analysis}")
        
        # Display winner/loser reasoning
        if winner and winner_reasoning:
            print(f"\n🥇 WINNER ({winner}):")
            print(f"   {winner_reasoning}")
        
        if loser_reasoning:
            # Find the loser (the model that's not the winner)
            loser = None
            for model_name in scores.keys():
                if model_name != winner:
                    loser = model_name
                    break
            
            if loser:
                print(f"\n🥈 RUNNER-UP ({loser}):")
                print(f"   {loser_reasoning}")
        
        print("\n" + "="*60)
    
    def display_text_reasoning(self, parsed: Dict[str, Any], responses: Dict[str, str]):
        """Display detailed reasoning from the judge in a readable format."""
        print("\n" + "="*60)
        print("🏆 JUDGE'S DETAILED ANALYSIS")
        print("="*60)
        
        winner = parsed.get("winner")
        winner_reasoning = parsed.get("winner_reasoning", "")
        loser_reasoning = parsed.get("loser_reasoning", "")
        
        # Display individual model analysis
        for model_name, model_data in parsed.items():
            if model_name in ["winner", "winner_reasoning", "loser_reasoning"]:
                continue
                
            print(f"\n📊 {model_name.upper()} ANALYSIS:")
            print(f"   Correctness: {model_data.get('correctness', 'N/A')}/5")
            print(f"   Relevance: {model_data.get('relevance', 'N/A')}/5")
            
            reasoning = model_data.get('reasoning', '')
            if reasoning:
                print(f"   Reasoning: {reasoning}")
        
        # Display winner/loser reasoning
        if winner and winner_reasoning:
            print(f"\n🥇 WINNER ({winner}):")
            print(f"   {winner_reasoning}")
        
        if loser_reasoning:
            # Find the loser (the model that's not the winner)
            loser = None
            for model_name in responses.keys():
                if model_name != winner:
                    loser = model_name
                    break
            
            if loser:
                print(f"\n🥈 RUNNER-UP ({loser}):")
                print(f"   {loser_reasoning}")
        
        print("\n" + "="*60)
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text response."""
        import json, re
        # Try to extract the first valid JSON block
        match = re.search(r"{.*}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError("No valid JSON in judge output") 