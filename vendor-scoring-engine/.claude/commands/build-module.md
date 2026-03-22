Implement the evaluation module: $ARGUMENTS

1. Read the existing plan (if any) or check CLAUDE.md for module spec
2. Read src/modules/__init__.py and src/scoring/results.py for interfaces
3. Implement the module following these steps:
   a. Create the module file in src/modules/
   b. Implement the main evaluation function returning ModuleResult
   c. Add diagnostic plot generation
   d. Add narrative summary generation
   e. Create corresponding test file in tests/modules/
   f. Generate synthetic test data if needed in tests/fixtures/
4. Run the tests to verify: pytest tests/modules/test_{module_name}.py -v
5. Run ruff check and ruff format on the new files
6. Summarize what was built and any open questions
