# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pneuma is a Python ML training project (v0.1.0, requires Python >=3.11) managed with `uv`. The training script runs on a remote PC via SSH.

## Commands

```bash
# Install dependencies
uv sync

# Run locally
uv run python main.py

# Deploy and run training on remote PC (WSL2 at 192.168.1.11)
bash train.sh
```

## Architecture

- [main.py](main.py) — entry point; contains the `main()` function
- [train.sh](train.sh) — SSH deployment script that pulls latest `main` branch on the remote PC (`ralts@192.168.1.11`) and launches `main.py` in the background, logging to `logs/train_<timestamp>.log`
- [pyproject.toml](pyproject.toml) — project metadata and dependencies (managed by `uv`)

## Workflow

Code is developed locally, then `train.sh` is used to `git pull` and execute on the remote GPU machine. Logs are stored in `logs/` on the remote machine.
