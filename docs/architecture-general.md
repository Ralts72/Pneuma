# 通用可扩展 Agent：整体架构

---

## 目录

1. [完整架构图](#1-完整架构图)
2. [Agent 清单与职责](#2-agent-清单与职责)
3. [意图路由层](#3-意图路由层)
4. [LLM 后端抽象](#4-llm-后端抽象)
5. [State 设计](#5-state-设计)
6. [工具系统](#6-工具系统)
7. [记忆系统](#7-记忆系统)
8. [LangGraph 编排骨架](#8-langgraph-编排骨架)
9. [Skill Agent 扩展协议](#9-skill-agent-扩展协议)
10. [前端架构](#10-前端架构)
11. [分阶段实现路线](#11-分阶段实现路线)

---

## 1. 完整架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                           用户界面层                                  │
│              CLI（第一阶段）·  Web / API（后续）                       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ 用户输入（文字 + 可选文件）
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         输入处理层                                    │
│  ┌──────────────┐  ┌───────────────────────────────────────────┐   │
│  │  输入解析     │  │              上下文组装                    │   │
│  │  文字 → msg  │  │  会话历史 + 用户档案 RAG + 知识库 RAG        │   │
│  │  文件 → 引用  │  │  （第三阶段启用）                           │   │
│  └──────────────┘  └───────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                层一：意图路由层                                        │
│                                                                     │
│  Intent Router                                                      │
│    输出 intent_type + effort_level                                   │
│                                                                     │
│  ├─ task       →  Executor（单次工具调用）                           │
│  ├─ query      →  Dialogue Agent（直接回答，无工具）                  │
│  ├─ refine     →  复用上一轮上下文，Executor 重新执行                 │
│  ├─ multi_step →  Task Planner 拆解 → 串行/并行 Executor             │
│  └─ meta       →  配置处理（切换模型、启用技能等）                    │
│                                                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                层二：编排层                                           │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  Task Planner                                                  │ │
│  │  将复杂任务分解为步骤列表 + 路由到对应 Skill                     │ │
│  └───────────────────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  Executor                                                      │ │
│  │  执行计划步骤：LLM 推理 + 工具调用循环                           │ │
│  └───────────────────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  Dialogue Agent                                               │ │
│  │  对话处理、结果展示、反馈收集                                    │ │
│  └───────────────────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  Memory Extractor（异步，不在关键路径）                         │ │
│  │  从 accept/reject 信号提炼用户偏好（第三阶段）                   │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ skill_type 路由
          ┌────────────────────┼──────────────────┐
          ▼                    ▼                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                层三：Skill Agent 层（可插拔）                          │
│                                                                     │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────────┐  │
│  │  Code Agent     │  │  File Agent    │  │  Search Agent        │  │
│  │  代码生成/调试   │  │  文件操作       │  │  网络/本地搜索        │  │
│  └────────────────┘  └────────────────┘  └──────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  General Agent（兜底，处理未路由的任务）                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  扩展方式：实现 SkillAgent 接口 + 在路由表注册，不改已有 Agent          │
│                                                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
          ┌────────────────────┴───────────────────┐
          ▼                                        ▼
┌──────────────────────────────┐  ┌───────────────────────────────────┐
│  模型层（各 Agent 直接调用）   │  │  工具层（确定性操作）               │
│                              │  │                                   │
│  · LLM（规划 / 对话）         │  │  · Shell（命令执行）                │
│  · Embedding（文本）          │  │  · 文件读写                        │
│  · 领域模型（后续扩展）        │  │  · 数据库（向量 DB / 关系 DB）      │
│                              │  │  · 外部服务（搜索 / API）           │
└──────────────────────────────┘  └───────────────────────────────────┘
          │                                        │
          └────────────────────┬───────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  记忆系统（四层 + 共享知识）                                            │
│  执行内存 · 会话上下文 · 项目记忆 · 用户档案 · 领域知识库               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Agent 清单与职责

```
层一：意图路由层
  · Intent Router         识别 intent_type（5 种）+ 评估 effort_level（3 档）

层二：编排层
  · Task Planner          任务分解 → 步骤列表 + skill_type 路由
  · Executor              执行步骤：LLM 推理 + 工具调用循环
  · Dialogue Agent        对话处理、结果展示
  · Memory Extractor      异步偏好提炼（第三阶段）

层三：Skill Agent 层（按需扩展）
  · General Agent         通用兜底（必须实现）
  · Code Agent            代码生成、调试、解释（第二阶段）
  · File Agent            文件操作与搜索（第二阶段）
  · Search Agent          网络/本地搜索（第二阶段）
  · （领域 Agent）         图像编辑、数据分析等（后续）

模型层（各 Agent 直接调用）
  · LLM                   Task Planner、Executor、Dialogue Agent 的推理基座
  · Embedding 模型         文本向量化（第三阶段 RAG 使用）

工具层（确定性操作）
  · Shell                 命令执行
  · 文件 I/O              文件读写
  · 数据库                向量 DB、关系 DB
  · 外部服务              搜索、API 调用

注：新增 Skill Agent 只需三步，不改动任何现有 Agent：
  1. 实现 SkillAgent 接口（run 方法 + TOOLS 声明 + SYSTEM_PROMPT）
  2. 在 Task Planner 路由表注册新 skill_type
  3. 在 LangGraph 图中 add_node + add_conditional_edge
```

---

## 3. 意图路由层

### Intent Router 输入输出

```
输入：用户当前消息 + 最近 2 轮对话（仅此，更多会干扰分类）

输出：
  {
    "intent_type":  "task | query | refine | multi_step | meta",
    "effort_level": "light | medium | heavy",
    "reason":       "一句话说明判断依据"
  }
```

### intent_type 路由语义

```
task        →  用户想完成单个可操作的请求（执行命令、编辑文件、调用 API）
               路由：Executor → Dialogue Agent

query       →  信息咨询，无需工具，无副作用
               路由：Dialogue Agent 直接回复

refine      →  在上一轮结果基础上调整（"简短一点"、"加上错误处理"）
               路由：复用上下文，Executor 重新执行，跳过重新规划

multi_step  →  需要分解的复杂请求
               路由：Task Planner 拆解 → 串行/并行 Executor

meta        →  配置 Agent 自身（切换模型、启用技能）
               路由：配置处理器，无需 LLM 推理
```

### effort_level 语义与资源分配

| effort_level | 触发条件 | 资源分配 |
|---|---|---|
| `light` | 单次 LLM 调用或单次工具调用 | 直接执行，不启动并行路径 |
| `medium` | 一个 Skill 内多次顺序工具调用 | 单 Skill Agent，顺序执行 |
| `heavy` | 需要任务分解、多 Skill 协作 | Task Planner 拆解，可并行执行 |

---

## 4. LLM 后端抽象

### 设计原则

Ollama、vLLM、LiteLLM 均暴露 OpenAI 兼容 API。使用单一 OpenAI SDK 客户端，通过不同 `base_url` 配置支持多后端，无需为每个提供商编写独立适配类。

```python
class ModelConfig:
    provider: str          # "ollama" | "openai_compat" | "local"
    model_id: str          # "qwen3.5:9b"、"gpt-4o" 等
    base_url: str | None   # 提供商端点
    api_key: str | None
    capabilities: list[str]  # ["chat", "tool_use", "reasoning", "embedding"]
    context_window: int    # 最大 token 数

class ModelRegistry:
    """管理可用模型，按能力选择最优模型"""

    def get(self, capability: str) -> ModelConfig | None:
        """按能力返回最优可用模型"""

    def register(self, name: str, config: ModelConfig) -> None:
        """注册新模型"""
```

### 多后端配置示例

```python
registry = ModelRegistry()

# 本地 Ollama
registry.register("qwen", ModelConfig(
    provider="ollama",
    model_id="qwen3.5:9b",
    base_url="http://192.168.1.11:11434/v1",
    api_key="ollama",
    capabilities=["chat", "tool_use", "reasoning"],
))

# OpenAI 兼容协议（vLLM 本地部署 / OpenAI 云端）
registry.register("gpt4o", ModelConfig(
    provider="openai_compat",
    model_id="gpt-4o",
    base_url=None,           # 默认 api.openai.com
    api_key="sk-...",
    capabilities=["chat", "tool_use", "reasoning", "embedding"],
))

# 本地训练模型（vLLM 部署）
registry.register("custom", ModelConfig(
    provider="local",
    model_id="my-finetuned-model",
    base_url="http://localhost:8000/v1",
    api_key="no-key",
    capabilities=["chat"],
))
```

---

## 5. State 设计

```python
from typing import Annotated, TypedDict
import operator


class AgentState(TypedDict):
    # ── 执行内存（各 Agent 在 Pipeline 内写入）────────────────────────
    user_input:    str
    intent_type:   str
    effort_level:  str
    task_plan:     dict | None        # Task Planner 输出：步骤 + 路由
    tool_results:  Annotated[list[dict], operator.add]  # 累积工具结果
    response:      str | None         # 最终响应

    # ── 会话上下文（Pipeline 启动前注入，只读）──────────────────────────
    session_id:             str
    conversation_history:   list[dict]    # 最近若干条消息

    # ── 长期记忆（RAG 注入，只读）── 第三阶段 ─────────────────────────
    user_preferences:    list[str]    # Profile Memory top-k
    relevant_knowledge:  list[str]    # Knowledge Base top-k

    # ── 控制流字段 ───────────────────────────────────────────────────
    current_skill: str | None
    retry_count:   int
    error_info:    str | None
```

**设计要点**：执行内存字段由各 Agent 在 Pipeline 内写入；会话上下文和长期记忆字段在 Pipeline 启动前注入，所有 Agent 只读。两类字段职责不交叉。

---

## 6. 工具系统

### 工具描述规范

工具描述质量直接影响 Agent 的调用准确率。每个工具必须声明五要素：

```python
class ToolSpec:
    name: str           # 动词_名词格式
    summary: str        # 一句话说清"解决什么问题"（最重要）
    input_schema: dict  # 每个参数的类型、含义、约束
    output_schema: dict # 成功 + 失败两种输出结构
    boundary: str       # 明确不能做什么（防止 Agent 误用）

# 示例
run_shell = ToolSpec(
    name="run_shell",
    summary="执行 Shell 命令并返回 stdout/stderr，适用于系统命令、文件列举、进程管理",
    input_schema={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "要执行的 Shell 命令"}
        },
        "required": ["command"],
    },
    output_schema={
        "success": {"stdout": "str", "return_code": "int"},
        "failure": {"stderr": "str", "return_code": "int"},
    },
    boundary="不执行交互式命令；不使用 sudo；超时 30 秒；不修改系统配置",
)
```

### 工具注册表

```python
class ToolRegistry:
    def register(self, spec: ToolSpec, handler: Callable) -> None: ...
    def get_openai_tools(self) -> list[dict]: ...     # 供 LLM 调用
    def execute(self, name: str, args: dict) -> Any: ...

    # 第二阶段：Tool RAG（工具数量增多后启用）
    def search(self, query: str, top_k: int = 5) -> list[ToolSpec]: ...
    # 当前任务描述 → 向量化 → 相似度检索 → 传 top-k 给 LLM，避免工具过多导致混淆
```

---

## 7. 记忆系统

四层记忆模型，从图像编辑 Agent 泛化而来：

```
热 ◄──────────────────────────────────────────────► 冷

┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  执行内存    │  │  会话上下文  │  │  项目记忆    │  │  用户档案    │
│  Working    │  │  Session    │  │  Project    │  │  Profile    │
│  Memory     │  │  Context    │  │  Memory     │  │  Memory     │
├─────────────┤  ├─────────────┤  ├─────────────┤  ├─────────────┤
│ 生命周期     │  │ 生命周期     │  │ 生命周期     │  │ 生命周期     │
│ 单次操作     │  │ 单次会话     │  │ 项目存续期   │  │ 账号存续期   │
│ 秒级         │  │ 分钟～小时   │  │ 天～周       │  │ 月～年       │
├─────────────┤  ├─────────────┤  ├─────────────┤  ├─────────────┤
│ 存储         │  │ 存储         │  │ 存储         │  │ 存储         │
│ LangGraph   │  │ 进程内存     │  │ SQLite/PG   │  │ 向量 DB      │
│ State       │  │ → Redis     │  │ + 向量 DB    │  │              │
├─────────────┤  ├─────────────┤  ├─────────────┤  ├─────────────┤
│ 内容         │  │ 内容         │  │ 内容         │  │ 内容         │
│ 当前操作的   │  │ 对话历史     │  │ 任务历史     │  │ 跨项目沉淀   │
│ 中间产物     │  │ + 工具调用   │  │ + 决策记录   │  │ 的偏好与     │
│              │  │ 结果         │  │ + 执行结果   │  │ 工作习惯     │
├─────────────┤  ├─────────────┤  ├─────────────┤  ├─────────────┤
│ 实现阶段     │  │ 第一阶段     │  │ 第二阶段     │  │ 第三阶段     │
│ 第一阶段     │  │（进程内）    │  │              │  │              │
│              │  │ 第二阶段     │  │              │  │              │
│              │  │（Redis）    │  │              │  │              │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
```

### 与图像编辑 Agent 记忆系统的对比

| | 图像编辑 Agent | 通用 Agent |
|---|---|---|
| 执行内存 | `ImageState`（图像路径、Vision 结果等） | `AgentState`（通用任务字段） |
| 偏好维度 | crop / color / text / layout / style | task / communication / tool_preference / workflow |
| 项目粒度 | 围绕素材文件的迭代项目 | 围绕目标任务的持续交互 |
| 知识库 | Design KB（优质编辑方案） | Domain KB（领域规则 + 优质解决方案） |
| 专项扩展 | Compositing / Style Transfer Agent | 任意 Skill Agent（Code / File / 图像编辑等） |

---

## 8. LangGraph 编排骨架

图的**拓扑结构由工程师固定**，LLM 只负责各节点内部的决策。结构固定保证可靠性和可调试性，节点内动态决策保证灵活性。

```python
from langgraph.graph import StateGraph, END
from pneuma.state import AgentState

graph = StateGraph(AgentState)

# ── 层一：路由 ────────────────────────────────────────────────────────
graph.add_node("intent_router",   run_intent_router)

# ── 层二：编排 ────────────────────────────────────────────────────────
graph.add_node("task_planner",    run_task_planner)     # 第二阶段启用
graph.add_node("executor",        run_executor)
graph.add_node("dialogue_agent",  run_dialogue_agent)

# ── 层三：Skill Agent ──────────────────────────────────────────────────
graph.add_node("general_agent",   run_general_agent)    # 兜底
# 后续：graph.add_node("code_agent",  run_code_agent)
# 后续：graph.add_node("file_agent",  run_file_agent)

# ── 边：固定的拓扑结构 ─────────────────────────────────────────────────
graph.set_entry_point("intent_router")
graph.add_conditional_edges("intent_router", route_by_intent, {
    "task":       "executor",        # 直接执行
    "query":      "dialogue_agent",  # 直接回答
    "refine":     "executor",        # 复用上下文，重新执行
    "multi_step": "task_planner",    # 先分解再执行（第二阶段）
    "meta":       "dialogue_agent",  # 配置变更
})
graph.add_edge("task_planner",   "executor")
graph.add_edge("executor",       "dialogue_agent")
graph.add_edge("dialogue_agent", END)

app = graph.compile()
```

### 第一阶段简化方案

`multi_step` 在第一阶段暂时直接路由到 `executor`（跳过 Task Planner），待第二阶段 Task Planner 上线后恢复完整路由。

---

## 9. Skill Agent 扩展协议

新增一个 Skill Agent 只需三步，**不修改任何现有 Agent**：

```
1. 创建 Skill 模块，实现接口：
   · run(state: AgentState) -> dict
   · TOOLS: list[ToolSpec]       领域专属工具
   · SYSTEM_PROMPT: str          领域知识

2. 在 Task Planner 路由表注册新类型：
   SKILL_ROUTES["image_edit"] = "image_edit_agent"

3. 在 LangGraph 图中注册：
   graph.add_node("image_edit_agent", run_image_edit_agent)
   # 在 task_planner 的条件边中加入新路由
```

### 示例：将图像编辑 Agent 接入为 Skill

图像编辑 Agent（现有系统）可直接作为一个 Skill Agent 接入通用 Agent，无需改造内部实现：

```
通用 Agent 接收 "帮我编辑这张图"
  → Intent Router: intent_type = task, effort_level = medium
  → Task Planner: skill_type = "image_edit"
  → 路由到 image_edit_agent（调用现有图像编辑系统）
  → Dialogue Agent 展示结果
```

---

## 10. 前端架构

### 10.1 技术栈

```
前端：React 19 + TypeScript + Vite
样式：Tailwind CSS
状态：useState / useContext（第一阶段），Zustand（后续）
后端 API：FastAPI（SSE 流式输出）
通信协议：Server-Sent Events（SSE）
```

**选择 SSE 而非 WebSocket 的原因**：Chat 场景是单向流（服务端 → 客户端），SSE 更简单，HTTP 原生支持，无需额外握手协议。WebSocket 适合双向实时通信，对此场景是过度设计。

---

### 10.2 整体前后端关系

```
┌───────────────────────────────────────────────────────────────┐
│  前端（React + TypeScript）                                     │
│                                                               │
│  ┌─────────────┐  ┌──────────────────┐  ┌─────────────────┐  │
│  │  ChatPanel   │  │  ToolCallPanel   │  │  DebugPanel     │  │
│  │  主对话区     │  │  工具调用可视化   │  │  路由/调试信息   │  │
│  └─────────────┘  └──────────────────┘  └─────────────────┘  │
│                          │ HTTP + SSE                         │
└──────────────────────────┼────────────────────────────────────┘
                           │
┌──────────────────────────┼────────────────────────────────────┐
│  后端 API 层（FastAPI）   │                                     │
│                          ▼                                    │
│  POST /api/chat  →  SSE Stream                                │
│  GET  /api/sessions/{id}                                      │
│  DELETE /api/sessions/{id}                                    │
│                          │                                    │
│                          ▼                                    │
│              LangGraph Agent Pipeline                         │
└───────────────────────────────────────────────────────────────┘
```

---

### 10.3 SSE 事件协议

后端在 `/api/chat` 以 SSE 推送以下事件，前端按事件类型分发渲染：

```
event: intent
data: {"intent_type": "task", "effort_level": "light", "reason": "..."}

event: delta
data: {"content": "我来帮你执行这个命令"}

event: tool_call
data: {"id": "call_1", "name": "run_shell", "args": {"command": "ls -la"}}

event: tool_result
data: {"id": "call_1", "result": {"stdout": "total 48\n...", "return_code": 0}, "duration_ms": 120}

event: delta
data: {"content": "执行完成，文件列表如下："}

event: done
data: {}

event: error
data: {"message": "模型调用超时", "code": "TIMEOUT"}
```

**设计要点**：
- `intent` 事件在 delta 之前推送，前端立即更新调试面板
- `tool_call` 和 `tool_result` 成对出现，前端可在 `tool_call` 时显示加载状态
- `delta` 是增量文本，前端追加到当前 assistant 消息

---

### 10.4 组件树

```
App
├── Layout
│   ├── Sidebar（会话列表，后续）
│   └── MainArea
│       ├── ChatPanel（主区域，左侧 ~70%）
│       │   ├── MessageList
│       │   │   ├── UserMessage
│       │   │   └── AssistantMessage
│       │   │       ├── TextContent（流式渲染）
│       │   │       └── ToolCallCard（内联，每次工具调用一个）
│       │   │           ├── ToolCallHeader（工具名 + 参数摘要）
│       │   │           ├── ToolCallArgs（可折叠，完整参数）
│       │   │           └── ToolCallResult（输出 / 错误）
│       │   └── InputBar
│       │       ├── TextArea（支持 Shift+Enter 换行）
│       │       └── SendButton
│       └── DebugPanel（右侧 ~30%，可折叠）
│           ├── IntentBadge（当前轮 intent_type + effort_level）
│           ├── AgentTrace（路由决策链路）
│           └── ToolCallLog（本轮所有工具调用记录）
```

---

### 10.5 核心数据结构（TypeScript）

```typescript
// 一条消息
interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;          // 文本内容（流式追加）
  toolCalls: ToolCall[];    // 该消息中的工具调用
  isStreaming: boolean;
  createdAt: number;
}

// 单次工具调用
interface ToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
  result: ToolResult | null;   // null = 调用中
  durationMs: number | null;
}

interface ToolResult {
  success: boolean;
  data: Record<string, unknown>;
}

// 调试面板数据
interface DebugInfo {
  intentType: string;
  effortLevel: string;
  reason: string;
  agentTrace: AgentTraceEntry[];
}

interface AgentTraceEntry {
  agent: string;
  decision: string;
  durationMs: number;
}

// 会话状态
interface ChatState {
  sessionId: string;
  messages: Message[];
  isLoading: boolean;
  debug: DebugInfo | null;
}
```

---

### 10.6 流式消息处理（核心逻辑）

```typescript
async function sendMessage(input: string, state: ChatState, setState: ...) {
  const userMsg: Message = { id: uuid(), role: "user", content: input, ... };
  const assistantMsg: Message = { id: uuid(), role: "assistant", content: "", toolCalls: [], isStreaming: true, ... };

  setState(s => ({ ...s, messages: [...s.messages, userMsg, assistantMsg], isLoading: true }));

  const response = await fetch("/api/chat", {
    method: "POST",
    body: JSON.stringify({ session_id: state.sessionId, message: input }),
  });

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    // 解析 SSE 行
    const lines = decoder.decode(value).split("\n");
    for (const line of lines) {
      if (!line.startsWith("data:")) continue;
      const event = parseSSEEvent(line);  // { event, data }

      switch (event.event) {
        case "intent":
          setState(s => ({ ...s, debug: { intentType: event.data.intent_type, ... } }));
          break;

        case "delta":
          setState(s => ({
            ...s,
            messages: s.messages.map(m =>
              m.id === assistantMsg.id
                ? { ...m, content: m.content + event.data.content }
                : m
            ),
          }));
          break;

        case "tool_call":
          // 在 assistant 消息中插入 pending 工具调用
          setState(s => ({
            ...s,
            messages: s.messages.map(m =>
              m.id === assistantMsg.id
                ? { ...m, toolCalls: [...m.toolCalls, { id: event.data.id, name: event.data.name, args: event.data.args, result: null, durationMs: null }] }
                : m
            ),
          }));
          break;

        case "tool_result":
          // 更新对应工具调用的结果
          setState(s => ({
            ...s,
            messages: s.messages.map(m =>
              m.id === assistantMsg.id
                ? { ...m, toolCalls: m.toolCalls.map(tc =>
                    tc.id === event.data.id
                      ? { ...tc, result: event.data.result, durationMs: event.data.duration_ms }
                      : tc
                  )}
                : m
            ),
          }));
          break;

        case "done":
          setState(s => ({
            ...s,
            isLoading: false,
            messages: s.messages.map(m =>
              m.id === assistantMsg.id ? { ...m, isStreaming: false } : m
            ),
          }));
          break;
      }
    }
  }
}
```

---

### 10.7 FastAPI 后端接口（配套）

前端所需的后端 API 层（独立于 LangGraph Agent，作为 HTTP 适配层）：

```python
# api.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json, asyncio

app = FastAPI()

@app.post("/api/chat")
async def chat(req: ChatRequest):
    async def event_stream():
        async for event in run_agent_pipeline(req.session_id, req.message):
            yield f"event: {event['type']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    return load_session(session_id)

@app.delete("/api/sessions/{session_id}")
async def clear_session(session_id: str):
    delete_session(session_id)
    return {"ok": True}
```

`run_agent_pipeline` 在 LangGraph 执行过程中 `yield` SSE 事件，将 LangGraph 节点的输出实时推送到前端。

---

### 10.8 目录结构

```
frontend/
├── src/
│   ├── components/
│   │   ├── chat/
│   │   │   ├── ChatPanel.tsx
│   │   │   ├── MessageList.tsx
│   │   │   ├── UserMessage.tsx
│   │   │   ├── AssistantMessage.tsx
│   │   │   ├── ToolCallCard.tsx
│   │   │   └── InputBar.tsx
│   │   └── debug/
│   │       ├── DebugPanel.tsx
│   │       ├── IntentBadge.tsx
│   │       └── ToolCallLog.tsx
│   ├── hooks/
│   │   ├── useChat.ts        # 核心 SSE 消息处理逻辑
│   │   └── useSession.ts     # 会话管理
│   ├── types/
│   │   └── index.ts          # Message, ToolCall, DebugInfo 等类型
│   ├── api/
│   │   └── client.ts         # fetch 封装 + SSE 解析
│   ├── App.tsx
│   └── main.tsx
├── index.html
├── vite.config.ts
├── tailwind.config.ts
└── package.json

api.py                        # FastAPI 入口（与 pneuma/ 并列）
```

---

### 10.9 UI 布局示意

```
┌─────────────────────────────────────────────────────────────┐
│  Pneuma                                          [调试 ▼]    │
├─────────────────────────────────┬───────────────────────────┤
│                                 │  调试面板                   │
│  You                            │  ┌─────────────────────┐  │
│  ┌──────────────────────────┐   │  │ intent: task        │  │
│  │ 列出当前目录的文件         │   │  │ effort: light       │  │
│  └──────────────────────────┘   │  └─────────────────────┘  │
│                                 │                            │
│  Pneuma                         │  工具调用                   │
│  ┌──────────────────────────┐   │  ┌─────────────────────┐  │
│  │ 好的，执行中…             │   │  │ run_shell           │  │
│  │                          │   │  │ ls -la              │  │
│  │ ┌────────────────────┐   │   │  │ ✓ 120ms             │  │
│  │ │ run_shell          │   │   │  └─────────────────────┘  │
│  │ │ $ ls -la           │   │   │                            │
│  │ │ total 48           │   │   │  路由链路                   │
│  │ │ drwxr-xr-x  ...    │   │   │  intent_router             │
│  │ └────────────────────┘   │   │   → executor               │
│  │                          │   │   → dialogue_agent          │
│  │ 文件列表如上所示。         │   │                            │
│  └──────────────────────────┘   │                            │
│                                 │                            │
├─────────────────────────────────┴───────────────────────────┤
│  ┌─────────────────────────────────────────────┐  [发送]    │
│  │ 输入消息…                                    │            │
│  └─────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

---

## 11. 分阶段实现路线

```
第一阶段：MVP（当前计划）
  · LLM 后端：Ollama via OpenAI SDK + ModelRegistry
  · 流式多轮对话（CLI）
  · Intent Router：5 种意图 + 3 档 effort
  · Executor：工具调用循环（LLM 决策 → 执行 → 返回）
  · 工具系统：ToolSpec + ToolRegistry + Shell 工具
  · 会话上下文：进程内存对话历史
  · LangGraph 固定拓扑编排

第二阶段：工具扩展 + Skill 体系
  · 更多内置工具：文件读写、网络搜索
  · Task Planner：多步骤拆解
  · 第一个 Skill Agent（Code Agent）
  · 会话上下文 → Redis 持久化
  · 多后端：OpenAI 兼容 API 配置化
  · heavy 任务并行执行

第三阶段：记忆 + 学习
  · 项目记忆（SQLite + Embedding）
  · 用户档案（向量 DB + 偏好提炼）
  · Memory Extractor（异步）
  · 上下文组装引入 RAG 注入
  · 领域知识库（Domain KB）

第四阶段：完整扩展
  · Skill 插件加载器（动态注册）
  · Web / API 接口
  · 可观测性（结构化 Trace 日志）
  · 容错与检查点（会话级检查点 + Skill 降级链路）
```

---

> **参考文档**
>
> 本架构设计参考了同项目图像编辑 Agent 的架构，将其中的分层结构、记忆系统、并行设计、工具规范等核心模式泛化为领域无关的通用框架。
>
> - 图像编辑 Agent 整体架构 → [Architecture.md](./Architecture.md)
> - 图像编辑 Agent 记忆系统 → [Memory.md](./Memory.md)
