<div align="center">

<img src="./assets/arc-logo.png" alt="ARC Logo" width="180" />

# ARC

### Action & Reasoning Core

**From thought to action**

Lightweight autonomous agent runtime built for local LLMs, structured reasoning, and real execution.

</div>

> [!WARNING]
> ## ARC is currently under active development
>
> Large parts of the runtime, APIs, CLI, SDK, and documentation are still evolving.
>
> Some examples shown in this repository are architectural previews and may not yet be fully implemented.
>
> In particular:
>
> - `pip install arc-runtime` is not yet available
> - CLI commands are currently placeholders
> - APIs may change without notice
> - Internal architecture is still being refined
> - Documentation may be incomplete or outdated
>
> ARC is currently focused on building the core runtime, reasoning pipeline, memory systems, and orchestration layers before stabilizing the public API.
>
> Expect breaking changes during development.

## What is ARC?

ARC is a local-first runtime for autonomous AI agents.

It combines reasoning, planning, memory, tool orchestration, and execution into one lightweight framework optimized for small models and low-end hardware.

ARC focuses on turning language models into structured problem-solving systems rather than passive chat interfaces.

# Features

- Lightweight autonomous agent runtime
- Multi-step planning and execution
- Structured working memory system
- Tool and function orchestration
- Local-first model support
- Small-model optimization (1.5B–7B)
- Async and streaming inference support
- Extensible SDK architecture
- Deterministic structured extraction
- Built on llama.cpp
- Context-efficient execution loops
- Runtime-managed model loading and unloading

# Architecture

```text
User Input
    ↓
Task Extraction
    ↓
Working Memory
    ↓
Planner
    ↓
Action Executor
    ↓
Validation Loop
    ↓
Goal Completion
````

# Philosophy

ARC bridges reasoning and action.

The goal is not to simulate intelligence through conversation, but to build systems capable of structured autonomous execution.

ARC is designed for:

* Small local models
* Low-end hardware
* Efficient context usage
* Long-running autonomous workflows
* Structured reasoning systems
* Modular extensibility

# Core Components

| Component   | Description                           |
| ----------- | ------------------------------------- |
| ARC Runtime | Managed llama.cpp runtime layer       |
| ARC Agent   | Planning and execution system         |
| ARC Memory  | Working memory and context management |
| ARC Planner | Step planning and orchestration       |
| ARC Tools   | Tool execution framework              |
| ARC SDK     | Extensible developer APIs             |

# Example

```python
from arc import Agent

agent = Agent(runtime)

result = agent.run(
    "Build a REST API with JWT authentication"
)

print(result)
```

# Why ARC?

Most agent systems are designed around cloud-scale models and massive context windows.

ARC takes a different approach.

It is designed around:

* Efficient execution loops
* Lightweight state handling
* Structured prompts
* Local inference
* Fast deterministic extraction
* Minimal runtime overhead

The result is an autonomous runtime capable of running effectively on consumer hardware with small local models.

# Installation

```bash
pip install arc-runtime
```

# CLI

```bash
arc run
arc chat
arc inspect
arc models
arc memory
arc tools
```

# Design Goals

* Local-first execution
* Deterministic behavior
* Modular architecture
* Small-model optimization
* Efficient context usage
* Long-running task execution
* Minimal dependencies
* Runtime-level orchestration

# Status

ARC is currently under active development.

The architecture and APIs are evolving rapidly as the runtime and agent systems mature.

---

<div align="center">

### ARC

**Action & Reasoning Core**

Structured intelligence for autonomous systems.

</div>