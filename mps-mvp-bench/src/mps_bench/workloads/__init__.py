"""In-container workloads. These modules import torch lazily.

Never import this package from the host orchestrator path (cli/runner) at module
scope -- the host must work without torch installed.
"""
