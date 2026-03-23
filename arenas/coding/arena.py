"""Coding arena — evaluate code solutions against test suites."""

import json

from arenas.base import Arena


SAMPLE_PROBLEMS = [
    {
        "id": "fibonacci",
        "description": "Write a function `fib(n)` that returns the nth Fibonacci number (0-indexed).",
        "tests": [
            "assert fib(0) == 0",
            "assert fib(1) == 1",
            "assert fib(5) == 5",
            "assert fib(10) == 55",
            "assert fib(20) == 6765",
        ],
    },
    {
        "id": "fizzbuzz",
        "description": "Write a function `fizzbuzz(n)` that returns a list of strings from 1 to n with FizzBuzz rules.",
        "tests": [
            "assert fizzbuzz(1) == ['1']",
            "assert fizzbuzz(3) == ['1', '2', 'Fizz']",
            "assert fizzbuzz(5) == ['1', '2', 'Fizz', '4', 'Buzz']",
            "assert fizzbuzz(15)[-1] == 'FizzBuzz'",
            "assert len(fizzbuzz(100)) == 100",
        ],
    },
    {
        "id": "palindrome",
        "description": "Write a function `is_palindrome(s)` that returns True if the string is a palindrome (case-insensitive, ignoring non-alphanumeric).",
        "tests": [
            "assert is_palindrome('racecar') == True",
            "assert is_palindrome('hello') == False",
            "assert is_palindrome('A man a plan a canal Panama') == True",
            "assert is_palindrome('') == True",
            "assert is_palindrome('ab') == False",
        ],
    },
    {
        "id": "two_sum",
        "description": "Write a function `two_sum(nums, target)` that returns indices of two numbers that add to target.",
        "tests": [
            "assert sorted(two_sum([2, 7, 11, 15], 9)) == [0, 1]",
            "assert sorted(two_sum([3, 2, 4], 6)) == [1, 2]",
            "assert sorted(two_sum([3, 3], 6)) == [0, 1]",
        ],
    },
    {
        "id": "reverse_words",
        "description": "Write a function `reverse_words(s)` that reverses the order of words in a string.",
        "tests": [
            "assert reverse_words('hello world') == 'world hello'",
            "assert reverse_words('  the sky is blue  ') == 'blue is sky the'",
            "assert reverse_words('a') == 'a'",
        ],
    },
]


class CodingArena(Arena):
    name = "coding"

    def scenarios(self) -> list[dict]:
        return SAMPLE_PROBLEMS

    def eval_script(self, agent_code: str, scenario: dict) -> str:
        tests_str = "\n    ".join(scenario["tests"])
        problem_id = scenario["id"]

        return f'''
import json, traceback

# --- Agent code (untrusted) ---
{agent_code}
# --- End agent code ---

tests = {json.dumps(scenario["tests"])}
passed = 0
failed = 0
errors = []

for test in tests:
    try:
        exec(test)
        passed += 1
    except AssertionError:
        failed += 1
        errors.append(f"FAIL: {{test}}")
    except Exception as e:
        failed += 1
        errors.append(f"ERROR in {{test}}: {{e}}")

total = passed + failed
result = {{
    "score": round(passed / total * 100, 1) if total > 0 else 0,
    "passed": failed == 0,
    "details": {{
        "problem": "{problem_id}",
        "tests_passed": passed,
        "tests_failed": failed,
        "total_tests": total,
        "errors": errors[:5],
    }}
}}
print(json.dumps(result))
'''
