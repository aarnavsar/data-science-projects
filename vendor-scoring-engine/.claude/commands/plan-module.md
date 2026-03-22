Plan a new evaluation module for the vendor scoring engine: $ARGUMENTS

1. Read CLAUDE.md to understand project architecture and data contracts
2. Check existing modules in src/modules/ for patterns and consistency
3. Review the ModuleResult dataclass in src/scoring/results.py
4. Create a detailed implementation plan:
   - What statistical tests / metrics does this module compute?
   - What reference data does it need?
   - What are the edge cases and failure modes?
   - What synthetic test data should be generated?
5. Present the plan for review before implementing
