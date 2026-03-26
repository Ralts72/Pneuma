# 图像编辑 Agent：整体架构

> **与当前仓库代码一致**的实现分层、三套 LangGraph、SSE 数据流见 → [architecture-implementation.md](./architecture-implementation.md)

> Memory 系统的完整设计（记忆层次、各 Agent 上下文、写入协议、可观测性、容错、技术栈、实现路线）见 → [memory.md](./memory.md)

---

## 目录

1. [完整架构图](#1-完整架构图)
2. [Agent 清单与职责](#2-agent-清单与职责)
3. [意图路由层](#3-意图路由层)
4. [编排层](#4-编排层)
5. [专项 Agent 层](#5-专项-agent-层)
6. [模型层](#6-模型层)
7. [工具层](#7-工具层)
8. [并行执行设计](#8-并行执行设计)

---

## 1. 完整架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                           用户界面层                                  │
│    Web / App / CLI    （流式输出 · Session 管理 · 图像上传/下载）        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ 用户输入（文字指令 + 图像素材）
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         输入处理层                                    │
│  ┌──────────────┐  ┌───────────────────────────────────────────┐   │
│  │  多模态解析   │  │              上下文组装                    │   │
│  │  图像 → URL   │  │  Session Context 注入 + 长期记忆 RAG 注入  │   │
│  │  文件 → 路径  │  │  （详见 memory.md §2、§4）                │   │
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
│  ├─ new_edit  →  完整编排流程（Vision Analyst + Edit Planner）       │
│  ├─ refine    →  跳过 Vision Analyst，复用 vision_result             │
│  ├─ compare   →  Edit Planner 并行生成多方向 Plan                    │
│  ├─ accept    →  触发 Memory Extractor（异步）                       │
│  ├─ undo      →  从 Session Context 版本列表回滚                     │
│  └─ query     →  直接交 Dialogue Agent 回复（不触发 Planner）         │
│                                                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                层二：编排层                                           │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  Vision Analyst（并行触发）                                     │ │
│  │  无状态 VLM 调用 → vision_result                               │ │
│  └───────────────────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  Asset Retrieval Agent（按需并行触发）                          │ │
│  │  Edit Plan 含素材需求时触发 → retrieved_assets                  │ │
│  └───────────────────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  Edit Planner（核心编排，五层上下文）                            │ │
│  │  vision_result + 用户指令 + 记忆 → Edit Plan + task_type 路由   │ │
│  └───────────────────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  Dialogue Agent                                               │ │
│  │  向用户展示结果、收集反馈、处理 query/accept/undo              │ │
│  └───────────────────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  Memory Extractor（异步，不在关键路径）                         │ │
│  │  accept/reject 触发 → 提炼 VisualPreference → Profile Memory   │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ Edit Plan + task_type 路由
          ┌────────────────────┼──────────────────┬────────────────┐
          ▼                    ▼                  ▼                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                层三：专项 Agent 层                                    │
│                                                                     │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────────┐  │
│  │ Compositing    │  │ Style Transfer │  │  Text Overlay Agent  │  │
│  │ Agent          │  │ Agent          │  │  （文字排版）          │  │
│  │ （多图拼接）    │  │ （风格迁移）    │  │                      │  │
│  └────────────────┘  └────────────────┘  └──────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  General Edit Agent（通用兜底，task_type = general 或未知）   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  各专项 Agent 均直接调用模型层，通过工具层完成确定性辅助操作          │
│                                                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
          ┌────────────────────┴───────────────────┐
          ▼                                        ▼
┌──────────────────────────────┐  ┌───────────────────────────────────┐
│  模型层（各 Agent 直接调用）   │  │  工具层（确定性操作）               │
│                              │  │                                   │
│  · LLM（规划 / 对话）         │  │  · 文件与存储（图像读写 / 云存储）   │
│  · VLM（视觉理解）            │  │  · 数据库（向量DB / Redis / 关系DB）│
│  · Embedding（CLIP / 文本）   │  │  · 外部服务（素材库 / 字体 / 色彩） │
│  · 图像编辑 / 生成模型         │  │  · 图像处理工具（格式转换 / 裁剪）   │
│  · 图像分割 / 检测模型         │  │                                   │
└──────────────────────────────┘  └───────────────────────────────────┘
          │                                        │
          └────────────────────┬───────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  记忆系统（四层 + 全局知识）                                           │
│  执行内存 · 会话上下文 · 项目记忆 · 用户档案 · 领域知识库 · 外部素材库   │
│  → 详见 memory.md §2                                                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Agent 清单与职责

```
层一：意图路由层
  Intent Router         识别 intent_type（6 种）+ 评估 effort_level（3 档）

层二：编排层
  Vision Analyst        无状态 VLM 调用，客观描述图像内容，输出 vision_result
  Asset Retrieval Agent 按 Edit Plan 的素材需求检索外部资产，输出 retrieved_assets
  Edit Planner          核心编排：五层上下文 → Edit Plan + task_type 路由
  Dialogue Agent        与用户沟通：展示结果、解释操作、处理 query/accept/undo
  Memory Extractor      异步提炼 VisualPreference，写入 Profile / Project Memory

层三：专项 Agent 层（按 task_type 路由）
  Compositing Agent     多图拼接（task_type = "compositing"）
  Style Transfer Agent  风格迁移（task_type = "style_transfer"）
  Text Overlay Agent    文字排版（task_type = "text_overlay"）
  General Edit Agent    通用兜底（task_type = "general" 或未知）

注：新增专项 Agent 只需实现 Agent 逻辑 + 在 Edit Planner 路由表注册新 task_type，
    不改动其他任何 Agent。
```

各 Agent 的上下文组成、Prompt 结构、Token 预算见 → [memory.md §4](./memory.md#4-各-agent-的上下文组成)

---

## 3. 意图路由层

### Intent Router 输出 Schema

```
输入：用户当前消息 + 最近 2 轮对话（仅此，更多会干扰分类）

输出：
  {
    "intent_type":  "new_edit | refine | compare | accept | undo | query",
    "effort_level": "light | medium | heavy",
    "reason":       "一句话说明判断依据"
  }
```

### intent_type 路由语义

```
new_edit  → 用户发起新一轮编辑（换图 / 全新指令）
             完整流程：Vision Analyst → Edit Planner → 专项 Agent

refine    → 在当前版本基础上微调（调颜色、改文字等）
             复用 vision_result，跳过 Vision Analyst 重跑

compare   → "帮我出几个方案对比一下"
             Edit Planner 并行生成 N 个方向 → 并行执行

accept    → 用户确认当前版本
             异步触发 Memory Extractor → 写入 Profile / Project Memory

undo      → 回退到上一个版本
             从 Session Context 版本列表直接回滚，不触发 Planner

query     → 纯对话咨询（"这个颜色叫什么？"）
             直接由 Dialogue Agent 回复，不触发 Planner / Executor
```

### effort_level 语义与资源分配

| effort_level | 触发条件 | Planner 的处理方式 |
|---|---|---|
| `light` | 微调操作（调颜色、改文字、accept/undo） | 复用 Session Context 中已有 vision_result，跳过 Vision Analyst 重跑 |
| `medium` | 重新规划构图、更换整体风格 | 完整流程，Vision Analyst 重跑，单方向 Plan |
| `heavy` | 多方案对比（compare 意图） | Vision Analyst 重跑 + Edit Planner 并行生成 N 个方向 |

---

## 4. 编排层

### LangGraph 图骨架（推荐方案）

图的**拓扑结构由工程师固定**，LLM 只负责各节点内部的决策。结构固定保证可靠性和可调试性，节点内动态决策保证灵活性。

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class ImageState(TypedDict):
    # 执行内存字段（Pipeline 内写入）
    input_images:     list[str]
    vision_result:    dict
    edit_plan:        dict
    retrieved_assets: list[dict]
    output_image:     str | None
    # 注入字段（只读，Pipeline 启动前装载）
    session_id:       str
    edit_history:     list[dict]
    profile_memories: list[str]
    knowledge_examples: list[str]
    # 控制流
    intent_type:      str
    effort_level:     str
    task_type:        str
    error_info:       str | None
    version_snapshots: Annotated[list, operator.add]


graph = StateGraph(ImageState)

# ── 层一：路由 ────────────────────────────────────────────────────────
graph.add_node("intent_router",         run_intent_router)

# ── 层二：编排 ────────────────────────────────────────────────────────
graph.add_node("vision_analyst",        run_vision_analyst)
graph.add_node("asset_retrieval",       run_asset_retrieval)
graph.add_node("edit_planner",          run_edit_planner)
graph.add_node("dialogue_agent",        run_dialogue_agent)
graph.add_node("memory_extractor",      run_memory_extractor)   # 异步节点

# ── 层三：专项 Agent ──────────────────────────────────────────────────
graph.add_node("compositing_agent",     run_compositing_agent)
graph.add_node("style_transfer_agent",  run_style_transfer_agent)
graph.add_node("text_overlay_agent",    run_text_overlay_agent)
graph.add_node("general_edit_agent",    run_general_edit_agent)

# ── 边：固定的拓扑结构 ─────────────────────────────────────────────────
graph.set_entry_point("intent_router")
graph.add_conditional_edges("intent_router", route_by_intent, {
    "new_edit":  "vision_analyst",
    "refine":    "edit_planner",      # 跳过 vision_analyst，复用缓存
    "compare":   "vision_analyst",
    "accept":    "memory_extractor",
    "undo":      "dialogue_agent",
    "query":     "dialogue_agent",
})
graph.add_edge("vision_analyst", "edit_planner")

# asset_retrieval 与 vision_analyst 并行：由 edit_planner 决定是否需要素材
graph.add_conditional_edges("edit_planner", route_by_task_type, {
    "compositing":    "compositing_agent",
    "style_transfer": "style_transfer_agent",
    "text_overlay":   "text_overlay_agent",
    "general":        "general_edit_agent",
    "needs_assets":   "asset_retrieval",    # 含素材需求时先检索
})
graph.add_edge("asset_retrieval",      "general_edit_agent")   # 检索完再执行
graph.add_edge("compositing_agent",    "dialogue_agent")
graph.add_edge("style_transfer_agent", "dialogue_agent")
graph.add_edge("text_overlay_agent",   "dialogue_agent")
graph.add_edge("general_edit_agent",   "dialogue_agent")
graph.add_edge("memory_extractor",     "dialogue_agent")
graph.add_edge("dialogue_agent",       END)

app = graph.compile()
```

### Edit Planner 的核心职责

Edit Planner 是编排层的核心，职责是**规划做什么**，不负责执行：

```
输入：
  · vision_result（Vision Analyst 输出）
  · 用户当前指令
  · 五层上下文（固定层 + 用户档案 + 项目记忆 + 知识库 + 编辑历史）

输出（Edit Plan）：
  {
    "task_type": "compositing | style_transfer | text_overlay | general",
    "steps": [...],                     // 高层意图描述，供专项 Agent 使用
    "material_required": [...],         // 是否需要外部素材（触发 Asset Retrieval）
    "layout": {...},                    // 布局参数（Compositing Agent 使用）
  }
```

五层上下文的组装方式见 → [memory.md §4 Edit Planner](./memory.md#edit-planner)

---

## 5. 专项 Agent 层

### 路由规则

```
Edit Plan.task_type
    │
    ├─ "compositing"      → Compositing Agent
    ├─ "style_transfer"   → Style Transfer Agent
    ├─ "text_overlay"     → Text Overlay Agent
    └─ "general" / 未知   → General Edit Agent（兜底）
```

### 各专项 Agent 的差异点

所有专项 Agent 的结构相同：接收 Edit Plan → 调用自己的领域模型 → 调用工具层保存结果。差异只在：

| Agent | 领域知识（固定层） | 直接调用的模型 | 特有工具 |
|---|---|---|---|
| Compositing | 构图法则、出血、对齐、边缘融合 | 图像分割模型 + 图像合成模型 | transform_image() |
| Style Transfer | 色彩空间、风格权重、内容保留 | 风格迁移模型 | — |
| Text Overlay | 排版规则、字体层级、可读性对比度 | 字体渲染 | get_font(), layout_text() |
| General Edit | 通用编辑 | 图像编辑模型 | transform_image() |

各专项 Agent 的完整上下文结构见 → [memory.md §4](./memory.md#4-各-agent-的上下文组成)

### 专项 Agent 扩展方式

新增一个专项 Agent 只需三步，**不修改任何现有 Agent**：

```
1. 实现 run_<new>_agent(state: ImageState) -> dict
2. 在 Edit Planner 的 task_type 枚举中注册新类型
3. 在 LangGraph 图中 add_node + add_conditional_edge
```

---

## 6. 模型层

模型层由各 Agent **直接调用**，不经过工具层。模型与工具的根本区别：

```
模型：需要推理，输出是概率性的"理解"或"生成"结果
工具：确定性执行，相同输入必然产生相同输出（幂等）
```

### 本系统使用的模型

```
┌──────────────────────────────┬────────────────────────────────────┐
│  模型类型                     │  用途                               │
├──────────────────────────────┼────────────────────────────────────┤
│  LLM                         │  Edit Planner 规划、Dialogue Agent  │
│  VLM                         │  Vision Analyst 视觉分析             │
│  Embedding（text）            │  Profile Memory / KB RAG 检索       │
│  Embedding（CLIP）            │  图像相似度检索（项目续作判断）        │
│  图像编辑 / 生成模型           │  General Edit Agent 核心执行         │
│  图像分割 / 检测模型           │  Compositing Agent 主体抠图          │
│  风格迁移模型                  │  Style Transfer Agent               │
└──────────────────────────────┴────────────────────────────────────┘
```

---

## 7. 工具层

工具层只处理**确定性的辅助操作**，不包含任何模型推理。

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
transform_image = ToolSpec(
    name="transform_image",
    summary="对图像做确定性几何变换（裁剪/缩放/旋转），不识别内容，不修改风格",
    input_schema={
        "image_path": "str，本地绝对路径",
        "ops": "list[CropOp|ScaleOp|RotateOp]，按顺序执行的变换列表"
    },
    output_schema={
        "success": {"output_path": "str", "original_size": "[W,H]", "output_size": "[W,H]"},
        "failure": {"error": "FileNotFoundError | ValueError", "message": "str"}
    },
    boundary="只做几何变换；不分析图像内容；不修改像素颜色；不生成新内容"
)
```

### 本系统的四类工具组

```
┌─────────────────────┬──────────────────────────────────────────────┐
│  工具组              │  工具示例                                      │
├─────────────────────┼──────────────────────────────────────────────┤
│  文件与存储          │  read_image() / save_output() / upload_cdn()  │
├─────────────────────┼──────────────────────────────────────────────┤
│  数据库操作          │  query_vector_db() / write_redis()            │
│                     │  save_project_version()                       │
├─────────────────────┼──────────────────────────────────────────────┤
│  外部服务            │  search_asset_library() / get_font()          │
│                     │  get_color_palette()                          │
├─────────────────────┼──────────────────────────────────────────────┤
│  图像处理工具         │  transform_image() / convert_format()         │
│  （无模型推理）       │  read_exif() / convert_color_space()          │
└─────────────────────┴──────────────────────────────────────────────┘
```

工具数量较多时，Agent 通过 **Tool RAG** 动态检索，只把 top-k 个工具描述传给 LLM，避免工具过多导致混淆：

```
当前任务描述 → 向量化 → 在工具库中相似度检索 → 取 top-5 → 传给 LLM 选择
```

---

## 8. 并行执行设计

并行是多 Agent 架构提供价值的核心来源，不是后期优化手段。

### 三层并行机会

**层次一：编排层并行（最高收益）**

新项目启动时，三路互不依赖，必须并行：

```
用户上传图像
    │
    ├─── Vision Analyst 分析图像        ─┐
    ├─── Profile Memory RAG 检索偏好    ─┤ 三路并行
    └─── Design KB RAG 检索优质方案     ─┘
                    │
                    ▼（三路结果汇入 Edit Planner）
```

**层次二：heavy 任务多版本并行**

`intent_type = compare` 或 `effort_level = heavy` 时：

```
Edit Planner → Plan A / Plan B / Plan C（并行生成）
                    │
Executor A / B / C（并行执行）
                    │
              汇聚点：Dialogue Agent 统一向用户展示对比结果
```

**层次三：专项 Agent 内部工具并行**

Compositing Agent 处理 N 张输入图时，各图的 `transform_image()` 调用互不依赖，并发执行，不串行：

```python
# 并行处理多张输入图
results = await asyncio.gather(*[
    transform_image(img, ops) for img in input_images
])
```

### 并行的陷阱

```
· light 任务 → 单 Agent 串行即可，拆分反而增加开销
· 只有 Orchestrator（Edit Planner）决定是否并行，不让专项 Agent 自行决定
· 并行分支必须有明确的汇聚点，汇聚后才能继续下游
```

---

> 以下内容见 memory.md：
>
> - **记忆系统详细设计**（四层记忆模型、各层 Schema、RAG 来源对比）→ [memory.md §2](./memory.md#2-记忆层次模型)
> - **各 Agent 上下文组成**（每个 Agent 的五层 Prompt 结构）→ [memory.md §4](./memory.md#4-各-agent-的上下文组成)
> - **全局数据流**（新项目启动 + 用户迭代完整流程）→ [memory.md §5](./memory.md#5-全局数据流)
> - **记忆写入协议**（accept/reject/Session 结束触发逻辑）→ [memory.md §6](./memory.md#6-记忆写入协议)
> - **可观测性**（AgentTrace Schema + 存储策略）→ [memory.md §7](./memory.md#7-可观测性)
> - **容错与检查点**（SessionCheckpoint + 专项 Agent 降级链路）→ [memory.md §8](./memory.md#8-容错与检查点)
> - **技术栈选型**→ [memory.md §9](./memory.md#9-技术栈)
> - **分阶段实现路线**→ [memory.md §10](./memory.md#10-分阶段实现路线)