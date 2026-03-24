"""LLM-powered coding agent — uses GitHub Models to solve problems.

Requires: OPENAI_API_KEY (or gh auth token) and OPENAI_BASE_URL set.

Usage:
    export OPENAI_API_KEY=$(gh auth token)
    export OPENAI_BASE_URL=https://models.inference.ai.azure.com
    python run.py local coding agents/examples/llm_solver.py
"""

import os

# Only import openai if available — gracefully degrade
try:
    from openai import OpenAI
    _client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://models.inference.ai.azure.com"),
    )
    _model = os.environ.get("OPENAI_MODEL", "gpt-4o")
    _HAS_LLM = bool(os.environ.get("OPENAI_API_KEY"))
except ImportError:
    _HAS_LLM = False


def _ask_llm(problem_description: str) -> str:
    """Ask the LLM to generate code for a problem."""
    response = _client.chat.completions.create(
        model=_model,
        messages=[
            {"role": "system", "content": (
                "You are an expert Python programmer. "
                "Output ONLY executable Python function definitions. "
                "No markdown fences, no explanations, no imports unless needed."
            )},
            {"role": "user", "content": problem_description},
        ],
        temperature=0.2,
    )
    code = response.choices[0].message.content.strip()
    if code.startswith("```python"):
        code = code[len("```python"):].strip()
    if code.startswith("```"):
        code = code[3:].strip()
    if code.endswith("```"):
        code = code[:-3].strip()
    return code


# Pre-solve all known problems at import time if LLM is available
_SOLUTIONS = {}

if _HAS_LLM:
    _PROBLEMS = {
        "fib": "Write a function fib(n) that returns the nth Fibonacci number (0-indexed). fib(0)=0, fib(1)=1.",
        "fizzbuzz": "Write a function fizzbuzz(n) that returns a list of strings from 1 to n. Multiples of 3 → 'Fizz', 5 → 'Buzz', both → 'FizzBuzz', else str(i).",
        "is_palindrome": "Write a function is_palindrome(s) that returns True if s is a palindrome, case-insensitive, ignoring non-alphanumeric characters.",
        "two_sum": "Write a function two_sum(nums, target) that returns a list of two indices whose values add to target.",
        "reverse_words": "Write a function reverse_words(s) that reverses word order in a string, trimming extra spaces.",
    }
    for name, desc in _PROBLEMS.items():
        try:
            code = _ask_llm(desc)
            exec(code, globals())
            _SOLUTIONS[name] = code
        except Exception as e:
            print(f"LLM failed for {name}: {e}")
else:
    # Fallback: hardcoded solutions when no LLM is available
    def fib(n):
        if n <= 1: return n
        a, b = 0, 1
        for _ in range(2, n + 1): a, b = b, a + b
        return b

    def fizzbuzz(n):
        return [("FizzBuzz" if i%15==0 else "Fizz" if i%3==0 else "Buzz" if i%5==0 else str(i)) for i in range(1,n+1)]

    def is_palindrome(s):
        c = "".join(ch.lower() for ch in s if ch.isalnum())
        return c == c[::-1]

    def two_sum(nums, target):
        seen = {}
        for i, n in enumerate(nums):
            if target-n in seen: return [seen[target-n], i]
            seen[n] = i

    def reverse_words(s):
        return " ".join(s.split()[::-1])
