"""Hard coding arena — problems that LLMs typically fail on first attempt.

These require tricky edge cases, performance constraints, or non-obvious algorithms.
Designed to test the multi-turn feedback loop.
"""

import json

from arenas.base import Arena


HARD_PROBLEMS = [
    {
        "id": "longest_substring",
        "description": "Write a function `longest_substring(s)` that returns the length of the longest substring without repeating characters.",
        "tests": [
            "assert longest_substring('abcabcbb') == 3",
            "assert longest_substring('bbbbb') == 1",
            "assert longest_substring('pwwkew') == 3",
            "assert longest_substring('') == 0",
            "assert longest_substring('au') == 2",
            "assert longest_substring('dvdf') == 3",
            "assert longest_substring('abba') == 2",
        ],
    },
    {
        "id": "eval_rpn",
        "description": "Write a function `eval_rpn(tokens)` that evaluates a Reverse Polish Notation expression. Tokens is a list of strings. Supported operators: +, -, *, /. Division truncates toward zero.",
        "tests": [
            "assert eval_rpn(['2', '1', '+', '3', '*']) == 9",
            "assert eval_rpn(['4', '13', '5', '/', '+']) == 6",
            "assert eval_rpn(['10', '6', '9', '3', '+', '-11', '*', '/', '*', '17', '+', '5', '+']) == 22",
            "assert eval_rpn(['3', '4', '-']) == -1",
        ],
    },
    {
        "id": "spiral_matrix",
        "description": "Write a function `spiral_order(matrix)` that returns elements of a 2D matrix in spiral order (clockwise from top-left). Return [] for empty matrix.",
        "tests": [
            "assert spiral_order([[1,2,3],[4,5,6],[7,8,9]]) == [1,2,3,6,9,8,7,4,5]",
            "assert spiral_order([[1,2,3,4],[5,6,7,8],[9,10,11,12]]) == [1,2,3,4,8,12,11,10,9,5,6,7]",
            "assert spiral_order([[1]]) == [1]",
            "assert spiral_order([[1,2],[3,4]]) == [1,2,4,3]",
            "assert spiral_order([]) == []",
        ],
    },
    {
        "id": "min_window_substring",
        "description": "Write a function `min_window(s, t)` that returns the minimum window substring of s that contains all characters of t (including duplicates). If no such window exists, return empty string ''.",
        "tests": [
            "assert min_window('ADOBECODEBANC', 'ABC') == 'BANC'",
            "assert min_window('a', 'a') == 'a'",
            "assert min_window('a', 'aa') == ''",
            "assert min_window('aa', 'aa') == 'aa'",
            "assert min_window('bba', 'ab') == 'ba'",
        ],
    },
    {
        "id": "lru_cache",
        "description": (
            "Implement an LRU Cache class.\n"
            "class LRUCache:\n"
            "    def __init__(self, capacity: int) — initialize with positive capacity\n"
            "    def get(self, key: int) -> int — return value if key exists, else -1\n"
            "    def put(self, key: int, value: int) — insert/update; if capacity exceeded, evict least recently used\n"
            "Both get and put must run in O(1) average time."
        ),
        "tests": [
            "c = LRUCache(2); c.put(1, 1); c.put(2, 2); assert c.get(1) == 1",
            "c = LRUCache(2); c.put(1, 1); c.put(2, 2); c.put(3, 3); assert c.get(2) == -1",
            "c = LRUCache(2); c.put(1, 1); c.put(2, 2); c.get(1); c.put(3, 3); assert c.get(2) == -1; assert c.get(1) == 1",
            "c = LRUCache(1); c.put(1, 1); c.put(2, 2); assert c.get(1) == -1; assert c.get(2) == 2",
        ],
    },
]


class HardCodingArena(Arena):
    name = "coding_hard"

    def scenarios(self) -> list[dict]:
        return HARD_PROBLEMS

    def eval_script(self, agent_code: str, scenario: dict) -> str:
        tests_json = json.dumps(scenario["tests"])
        problem_id = scenario["id"]

        return f'''
import json, traceback

# --- Agent code (untrusted) ---
{agent_code}
# --- End agent code ---

tests = {tests_json}
passed = 0
failed = 0
errors = []

for test in tests:
    try:
        exec(test)
        passed += 1
    except AssertionError:
        failed += 1
        errors.append(f"FAIL: {{test[:100]}}")
    except Exception as e:
        failed += 1
        errors.append(f"ERROR: {{type(e).__name__}}: {{e}}")

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
