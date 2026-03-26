# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pneuma is a Python Ollama inference client (v0.1.0, requires Python >=3.11) managed with `uv`. It connects to an Ollama instance running on a remote machine (`192.168.1.11:11434`) via the OpenAI-compatible API and streams responses from a local LLM.

## Commands

```bash
# Install dependencies
uv sync

# Run
uv run python main.py
```

## Architecture

`main.py` is the sole entry point. It uses the `openai` SDK pointed at a local Ollama server, streams a chat completion from `qwen3.5:9b`, and prints reasoning tokens (under `delta.reasoning`) and content tokens separately — labelled `【思考过程】` and `【回答】` respectively.

Key constants at the top of `main.py`:
- `OLLAMA_BASE_URL` — the remote Ollama endpoint
- `MODEL` — the model tag to query

## 工作方式
- 在实现前先说明方法。
- 若需求有歧义、风险较高或影响较大，先澄清并等待批准，再开始写代码。
- Plan 只写方案，不写代码。
- 坚持 Spec Coding，不做 Vibe Coding。
- 优先迭代，使用 `/loop`。
- 完成后执行 `/simplify`。
- 你是统筹者，先指定一个 Claude 产出 plan。
- 基于 plan 将任务拆分后分配给不同的 Claude 并行或串行执行。
- 所有子任务应保持边界清晰、职责明确、便于独立验证。
- 完成后再指定一个 Claude 汇总结果并输出最终报告给我。

## 编码规则
- 代码中只允许使用英文。
- Spec 不依赖行号定位代码。
- 注释中不要写开发过程式说明。
- 优先用概念性描述定位代码，不用“文件路径 + 行号”。

## 拆分与范围控制
- 将任务拆分为低耦合、可独立验证的子任务，必要时使用 `/batch`。
- 重复出现 3 次的流程应沉淀为 Skill。
- 任务分配时优先控制单个 Claude 的上下文范围，避免把过多背景一次性注入到同一个上下文中。
- 只向负责该子任务的 Claude 提供完成任务所必需的最小上下文。
- 跨任务共享信息时，优先传递经过整理的结论、约束和接口，而不是完整过程性上下文。

## 质量要求
- 项目早期只保留最小必要质量标准：可运行、可验证、可回滚。
- 优先保证关键路径和高风险改动可验证。
- 处理 bug 时，先复现，再修复并验证。

## 纠错与协作
- 被纠正时，识别原因并改进做法；对重复性问题，沉淀为明确规则。
- 实现与审查分离：先完成方案或代码，再独立复核.。
- 统筹者负责跟踪各 Claude 的输入、输出、依赖关系和验收结果，避免遗漏与重复劳动。
- 汇总报告应至少包含：任务目标、各子任务结果、验证结论、遗留风险、后续建议。

## 禁止事项
- 永远不要使用 `/init`。
- `CLAUDE.md` 应按项目实际需求编写，不要套用空泛模板。
- Avoid terms to describe development progress (`FIXED`, `Step`, `Week`, `Section`, `Phase`, `AC-x`, etc) in code comments or commit message or PR body.
- Avoid AI tools name (like Codex, Claude, Grok, Gemini, ...) in code comments or git commit message (including authorship) or PR body.