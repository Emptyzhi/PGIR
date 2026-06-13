# Development Workflow

1. Create a branch for each method or experiment-code change.
2. Keep benchmark data, API keys, logs, and generated results out of Git.
3. Run the structural smoke before committing:

   ```powershell
   python disasterbench/smoke_frontier_structural.py
   ```

4. Commit code before launching an experiment.
5. Record the commit hash alongside every reported experiment result.
6. Never use benchmark gold plans or official scorers inside runtime diagnosis,
   repair, operator routing, or candidate selection.
