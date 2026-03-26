# 通用图像编辑 Agent：Memory 与上下文架构

---

## 目录

1. [图像编辑 Agent 的核心特殊性](#1-图像编辑-agent-的核心特殊性)
2. [记忆层次模型](#2-记忆层次模型)
3. [Agent 分层架构](#3-agent-分层架构)
4. [各 Agent 的上下文组成](#4-各-agent-的上下文组成)
5. [全局数据流](#5-全局数据流)
6. [记忆写入协议](#6-记忆写入协议)
7. [可观测性](#7-可观测性)
8. [容错与检查点](#8-容错与检查点)
9. [技术栈](#9-技术栈)
10. [分阶段实现路线](#10-分阶段实现路线)

---

## 1. 图像编辑 Agent 的核心特殊性

图像编辑 Agent 与通用文本 Agent 在 Memory 设计上有三个根本差异，驱动了后续所有设计决策：

**差异一：反馈信号天然结构化**
每次"接受 / 拒绝 / 微调"都直接对应一个具体的视觉决策，是比文字描述更精准的偏好标注，应作为一等公民写入记忆。

**差异二：偏好难以言说，只能从行为中学习**
用户无法描述"我想要的色调的 HSL 参数"，系统必须从"接受了什么、拒绝了什么"中反向推断审美偏好。Memory 写入不能依赖用户主动描述，必须从行为自动提炼。

**差异三：任务天然形成"项目"**
用户不是一次性提问，而是围绕一套素材持续迭代，同一项目可能跨越多个 Session。记忆需要"项目"这个中间粒度，介于 Session 和用户账号之间。

---

## 2. 记忆层次模型

### 2.1 四层私有记忆

```
热 ◄──────────────────────────────────────────────► 冷
快 ◄──────────────────────────────────────────────► 慢

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
│ 进程内存     │  │ Redis        │  │ 关系DB       │  │ 向量DB       │
│              │  │              │  │ + 向量DB     │  │              │
├─────────────┤  ├─────────────┤  ├─────────────┤  ├─────────────┤
│ 内容         │  │ 内容         │  │ 内容         │  │ 内容         │
│ 当前操作的   │  │ 本次会话的   │  │ 该项目所有   │  │ 跨项目沉淀   │
│ 中间产物     │  │ 完整状态     │  │ 历史版本     │  │ 的审美偏好   │
│              │  │ + 编辑历史   │  │ + 偏好轨迹   │  │ + 工作习惯   │
├─────────────┤  ├─────────────┤  ├─────────────┤  ├─────────────┤
│ 访问模式     │  │ 访问模式     │  │ 访问模式     │  │ 访问模式     │
│ 共享 State  │  │ Orchestrator │  │ RAG 检索     │  │ RAG 检索     │
│              │  │ 显式注入     │  │              │  │              │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
```

四层记忆对应 LangGraph `ImageState` 中不同来源的字段：

```python
from typing import TypedDict, Annotated
import operator

class ImageState(TypedDict):
    # ── 执行内存（进程内，LangGraph State 管理，生命周期 = 单次 Pipeline）──────
    input_images:      list[str]          # 原始素材路径（不可修改，见 INV-001）
    vision_result:     dict               # Vision Analyst 分析结果
    edit_plan:         dict               # Edit Planner 生成的方案（含 task_type）
    retrieved_assets:  list[dict]         # Asset Retrieval 检索到的素材
    candidate_images:  dict[str, str]     # 并行执行的多版本输出 {model_name: path}

    # ── 会话上下文（Pipeline 启动前从 Redis 注入，只读）────────────────────────
    session_id:        str
    vision_cache:      dict | None        # 缓存的 vision_result（refine 时复用）
    version_snapshots: Annotated[list, operator.add]  # 历史版本（追加）
    edit_history:      list[dict]         # 近期编辑记录 {instruction, plan_summary, accepted}

    # ── 长期记忆（Pipeline 启动前 RAG 检索后注入，只读）───────────────────────
    profile_memories:     list[str]       # Profile Memory top-3，用户偏好描述
    knowledge_examples:   list[str]       # Design KB top-2，优质 Edit Plan 样例

    # ── 控制流字段 ─────────────────────────────────────────────────────────────
    intent_type:   str                    # intent_router 输出
    effort_level:  str                    # "light" | "medium" | "heavy"
    task_type:     str                    # edit_planner 输出，专项 Agent 路由依据
    retry_count:   int                    # 重试计数（容错用）
    error_info:    str | None
```

**设计要点**：执行内存字段由各 Agent 在 Pipeline 内写入；Session Context 和长期记忆字段在 Pipeline 启动前注入，所有 Agent 只读。两类字段职责不交叉。

### 2.2 全局共享知识来源

系统中存在三类全局共享的知识来源，RAG 检索的对象和用途各不相同，不能混淆：

```
┌──────────────────────────────────────────────────────────────────────┐
│  三类 RAG 来源对比                                                     │
├──────────────┬─────────────────┬──────────────────┬───────────────── ┤
│              │ 领域知识库        │ 外部素材库          │ 用户档案           │
│              │ Design KB        │ Asset Library      │ Profile Memory   │
├──────────────┼─────────────────┼──────────────────┼─────────────────┤
│ 存什么        │ 优质 Edit Plan   │ 花字/贴纸/字体/     │ 用户的审美偏好     │
│              │ 样例（匿名化）    │ 色彩方案等可复用     │ 和操作习惯         │
│              │ + 编辑规范规则    │ 设计资产            │                  │
├──────────────┼─────────────────┼──────────────────┼─────────────────┤
│ 谁写入        │ 系统从多用户      │ 设计师/运营人工      │ 系统从用户         │
│              │ 接受的方案中      │ 维护，离线索引       │ accept/reject    │
│              │ 匿名化沉淀        │                    │ 行为自动提炼       │
├──────────────┼─────────────────┼──────────────────┼─────────────────┤
│ 检索 query   │ 当前图像的        │ Edit Plan 中描述     │ 当前任务意图       │
│              │ CLIP embedding   │ 的素材需求关键词      │ 文本 embedding   │
├──────────────┼─────────────────┼──────────────────┼─────────────────┤
│ 注入对象      │ Edit Planner     │ 专项 Agent          │ Edit Planner     │
│              │ context（文本）   │ context（文件引用）   │ context（文本）   │
├──────────────┼─────────────────┼──────────────────┼─────────────────┤
│ 产出类型      │ 规划参考文本      │ 可直接用于执行的      │ 偏好描述文本       │
│              │                  │ 素材文件/URL         │                  │
└──────────────┴─────────────────┴──────────────────┴─────────────────┘
```

**关键区分**：领域知识库和用户档案的 RAG 结果是**文本上下文**，帮助 Edit Planner 做更好的决策；外部素材库的 RAG 结果是**可执行资产**（文件/URL），直接被专项 Agent 用于编辑执行。两类用途不同，不能放入同一个字段。

```
┌──────────────────────────────────────────────────────────────┐
│  领域知识库（Design Knowledge Base）                           │
├──────────────────────────────────────────────────────────────┤
│  内容：与特定用户无关的领域沉淀                                  │
│    · 经多用户验证的优质编辑方案（匿名化后沉淀）                    │
│    · 风格样本图的视觉 embedding 索引（CLIP）                    │
│    · 编辑规范与约束规则                                        │
│  写入：被多个不同用户接受的方案，匿名化后沉淀                      │
│  访问模式：RAG 检索注入 Edit Planner（文本上下文）               │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  外部素材库（Asset Library）                                   │
├──────────────────────────────────────────────────────────────┤
│  内容：可复用的设计执行资产                                      │
│    · 花字、艺术字样式包                                        │
│    · 贴纸、装饰元素                                            │
│    · 字体文件、色彩方案                                        │
│  写入：设计师 / 运营人工维护，离线批量索引                        │
│  访问模式：Asset Retrieval Agent 按需检索，                    │
│            结果作为文件引用注入专项 Agent context              │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Agent 分层架构

### 3.1 层级结构

```
┌──────────────────────────────────────────────────────────────────┐
│  层一：路由层                                                      │
│  Intent Router                                                   │
│  · 输出 intent_type + effort_level（light / medium / heavy）     │
└──────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  层二：编排层                                                      │
│  Vision Analyst  ·  Edit Planner  ·  Dialogue Agent              │
│  Asset Retrieval Agent（按需触发，与 Vision Analyst 并行）          │
│  Memory Extractor（异步，不在关键路径）                             │
└──────────────────────────────────────────────────────────────────┘
                           │ task_type 路由
          ┌────────────────┼──────────────────┐
          ▼                ▼                  ▼
┌──────────────────────────────────────────────────────────────────┐
│  层三：专项 Agent 层                                               │
│  Compositing Agent · Style Transfer Agent · Text Overlay Agent   │
│  General Edit Agent（兜底，必须实现）                              │
└──────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  模型层（各 Agent 直接调用）                                        │
│  LLM / VLM / Embedding 模型 / 图像生成编辑模型 / 分割检测模型        │
└──────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────┐
│  层四：工具层（确定性操作）                                         │
│  文件读写 · 数据库操作 · 外部服务 · 图像格式处理                      │
└──────────────────────────────────────────────────────────────────┘
```

工具层按功能分为四组：

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  文件与存储      │  │  数据库操作      │  │  外部服务        │
│ · 图像文件读写   │  │ · 向量DB读写     │  │ · 素材库搜索     │
│ · 云存储上传     │  │ · Redis读写      │  │ · 字体服务       │
│ · CDN 分发       │  │ · 关系DB读写     │  │ · 色彩服务       │
└─────────────────┘  └─────────────────┘  └─────────────────┘
┌─────────────────┐
│  图像处理工具     │ 
│ · 格式转换       │
│ · 尺寸裁剪/缩放  │
│ · 元数据读写     │
│ · 色彩空间转换   │
└─────────────────┘
```

**工具描述规范**：每个工具必须包含五要素——`name`（动词_名词格式）、`summary`（一句话说清解决什么问题）、`input_schema`、`output_schema`（含失败结构）、`boundary`（明确不能做什么）。

### 3.2 并行执行设计

```
层次一：编排层并行（影响最大）

  新项目启动时，三路互不依赖，必须并行：
  ┌─────────────────────┐
  │  Vision Analyst 分析 │
  │  Profile Memory 检索 │  → 三路完成后汇入 Edit Planner
  │  Knowledge Base 检索 │
  └─────────────────────┘

  Edit Planner 输出 Edit Plan 后，如含素材需求：
  ┌──────────────────────────┐
  │  Asset Retrieval Agent   │  → retrieved_assets → 注入专项 Agent（条件触发）
  └──────────────────────────┘

  compare 意图（heavy）时，多方向并行：
  Edit Planner → Plan A / B / C → Image Executor A / B / C（并行）

层次二：专项 Agent 内部工具并行
  多张输入图的独立工具调用并发执行（如 Compositing Agent 批量变换）

层次三：并行约束
  · light effort → 单 Agent 串行，不启动并行路径
  · 并行分支必须有明确汇聚点，由 Orchestrator 负责汇聚
  · Orchestrator 决定是否启动并行，不由 subagent 自行决定
```

### 3.3 完整 Agent 清单

```
层一：路由层
  · Intent Router          识别用户意图 + 评估 effort_level

层二：编排层
  · Vision Analyst         无状态视觉分析，输出结构化图像描述
  · Edit Planner           生成 Edit Plan + 路由 task_type + 决定是否触发素材检索
  · Asset Retrieval Agent  按需检索外部素材（花字/贴纸/字体等），结果注入专项 Agent
  · Dialogue Agent         对话响应，展示结果，收集反馈
  · Memory Extractor       异步偏好提炼，写入 Project/Profile Memory

层三：专项 Agent 层（按需扩展）
  · Compositing Agent      多图拼接
  · Style Transfer Agent   风格迁移
  · Text Overlay Agent     文字排版
  · General Edit Agent     通用兜底（必须实现）

模型层（各 Agent 直接调用）
  · LLM                    Edit Planner、Dialogue Agent 的推理基座
  · VLM                    Vision Analyst 的推理基座
  · Embedding 模型          CLIP（视觉）、text-embedding（文本）
  · 图像编辑/生成模型        专项 Agent 的核心执行模型
  · 图像分割/检测模型        Compositing Agent 等直接调用

层四：工具层（确定性操作）
  · 文件与存储              图像文件读写、云存储、CDN
  · 数据库操作              向量DB、Redis、关系DB
  · 外部服务               素材库、字体、色彩服务
  · 图像处理工具            格式转换、裁剪缩放、元数据读写
```

---

## 4. 各 Agent 的上下文组成

### Intent Router

```
目标：快速分类意图 + 评估任务复杂度

SYSTEM:
  [固定层]       意图分类角色 + 意图类型定义 + 复杂度评估标准
HUMAN MESSAGES:
  [短期记忆层]   最近 2 轮对话
  [当前输入]     用户当前消息

输出 schema：
  {
    "intent_type": "new_edit | refine | compare | accept | undo | query",
    "effort_level": "light | medium | heavy",
    "reason": "一句话说明判断依据"
  }

effort_level 语义：
  · light  → 单 Agent + 少量工具调用（微调颜色、调整文字）
  · medium → 3~4 个 Agent + 并行检索（重新规划构图、更换风格）
  · heavy  → 多方向 Plan 并行执行（出几个风格方案对比）

Edit Planner 据此分配资源：
  · light  → 复用 Session Context 的 vision_result，跳过 Vision Analyst
  · medium → 重跑 Vision Analyst + 单方向 Plan
  · heavy  → 重跑 Vision Analyst + 并行生成 N 个方向
```

### Vision Analyst

```
目标：客观描述图像内容，不受用户偏好影响（无状态）

SYSTEM:
  [固定层]       分析角色 + 输出 schema 定义
  [知识检索层]   相似图像类型的已知特征 top-2
                query：当前图像的 CLIP embedding
HUMAN MESSAGES:
  [当前任务层]   原始图像（直接传图，无历史）
```

### Asset Retrieval Agent

```
目标：按 Edit Plan 中的素材需求，从外部素材库检索可直接使用的设计资产

触发条件：Edit Planner 判断计划中含有素材依赖（如"需要花字样式""需要贴纸装饰"）
          才触发，与 Vision Analyst 并行执行；普通编辑任务不触发

SYSTEM:
  [固定层]       素材检索角色 + 素材库结构说明 + 输出格式规范
HUMAN MESSAGES:
  [当前任务层]   Edit Plan 中的素材需求描述（关键词 / 风格标签）

输出：retrieved_assets —— 可执行资产的引用列表
  [
    { "type": "花字", "name": "霓虹风-1", "url": "...", "tags": ["夜晚", "炫彩"] },
    { "type": "贴纸", "name": "星星-金色", "url": "...", "tags": ["装饰", "节日"] }
  ]
  → 写入 ImageState.retrieved_assets，由专项 Agent 在执行时直接引用

与三种 RAG 的边界：
  · 本 Agent 检索的是"可执行资产文件"，不是文本上下文
  · 领域知识库 / 用户档案的 RAG 注入 Edit Planner（影响规划决策）
  · 本 Agent 的结果注入专项 Agent（影响执行内容）
```

### Edit Planner

```
目标：生成精准的 Edit Plan，让执行器无需再理解用户意图

SYSTEM:
  [固定层]       规划角色 + Edit Plan 输出格式规范
  [用户档案层]   Profile Memory top-3（query：当前图像风格描述）
                示例："用户偏好：照片保留更多背景，不要紧裁"
                示例："用户拒绝过：过多文字装饰叠加的方案"
  [项目记忆层]   Project Memory：本项目历史版本摘要 top-2（仅有历史时注入）
                示例："v1：构图左置，用户接受；v2：加了贴纸装饰，用户拒绝"
  [知识检索层]   领域知识库：相似风格的优质 Edit Plan top-2
                query：当前图像 CLIP embedding
HUMAN MESSAGES:
  [当前任务层]   Vision Analyst 分析结果 + 用户当前指令
  [会话历史层]   Session Context 中最近 3 轮编辑记录（相关性过滤）
                格式：{instruction, plan_summary, feedback, accepted: bool}

Token 预算：
  固定层     ≈  1K
  用户档案   ≈  1K
  项目记忆   ≈  2K
  知识检索   ≈  4K
  当前任务   ≈  8K
  编辑历史   ≈  2K
  ─────────────────
  合计       ≈ 18K
```

以下示例展示五层上下文如何在代码中组装：

```python
PLANNER_SYSTEM_PROMPT = """你是一名专业的图像编辑规划师。
根据图像分析结果和用户指令，生成结构化的 Edit Plan。
输出格式：{"task_type": "...", "steps": [...], "material_required": [...]}"""


def build_planner_prompt(
    state: ImageState,
    project_history: list[str],
    user_instruction: str,
) -> list[dict]:
    """组装 Edit Planner 的五层上下文，返回可直接传入 LLM 的 messages 列表"""

    # ── 层一：固定层 ────────────────────────────────────────────────────────
    system_parts = [PLANNER_SYSTEM_PROMPT]

    # ── 层二：用户档案层（Profile Memory RAG 结果，已在 Pipeline 启动前注入 State）
    if state.get("profile_memories"):
        block = "【用户偏好记忆（请在规划时遵守）】\n" + "\n".join(
            f"· {m}" for m in state["profile_memories"]
        )
        system_parts.append(block)

    # ── 层三：项目记忆层（仅有历史时注入，避免空内容干扰）────────────────────
    if project_history:
        block = "【本项目历史版本（可参考用户的接受/拒绝轨迹）】\n" + "\n".join(
            f"· {h}" for h in project_history
        )
        system_parts.append(block)

    # ── 层四：知识检索层（Design KB RAG 结果）────────────────────────────────
    if state.get("knowledge_examples"):
        block = "【相似风格优质 Edit Plan 参考】\n" + "\n".join(
            state["knowledge_examples"]
        )
        system_parts.append(block)

    messages: list[dict] = [
        {"role": "system", "content": "\n\n".join(system_parts)}
    ]

    # ── 层五：会话历史层（最近 3 轮编辑记录，格式化为对话形式）─────────────────
    for record in state.get("edit_history", [])[-3:]:
        messages.append({
            "role": "user",
            "content": f"指令：{record['instruction']}"
        })
        verdict = "接受 ✓" if record["accepted"] else "拒绝 ✗"
        messages.append({
            "role": "assistant",
            "content": f"方案摘要：{record['plan_summary']}  [{verdict}]"
        })

    # ── 当前任务（Human turn）────────────────────────────────────────────────
    messages.append({
        "role": "user",
        "content": (
            f"图像分析结果：\n{state['vision_result']}\n\n"
            f"用户指令：{user_instruction}"
        )
    })

    return messages
```

### Compositing Agent（专项 Agent 代表）

```
目标：以深领域知识完成多图拼接任务
触发：Edit Plan.task_type == "compositing"

SYSTEM:
  [固定层]       多图拼接专家角色 + 拼接规则（构图法则、出血、对齐）
  [知识检索层]   相似拼接风格的成功方案 top-2
                query：Edit Plan 中的布局描述 embedding
  [当前任务层]   Edit Plan + 各输入图像的 Vision Analyst 分析结果
HUMAN MESSAGES:
  （无，任务是确定性执行）

直接调用的模型：图像分割模型、图像合成/编辑模型
使用的工具：transform_image()、save_output()
```

其他专项 Agent 遵循相同模式，差异只在固定层领域规则、调用的模型、知识检索的专项子集。

### Dialogue Agent

```
目标：与用户自然沟通，展示结果，解释操作

SYSTEM:
  [固定层]       对话助手角色 + 语气风格定义
  [用户档案层]   Profile Memory top-2（沟通风格偏好）
  [当前任务层]   项目状态摘要（当前版本、已做了什么）
HUMAN MESSAGES:
  [短期记忆层]   最近 6 条对话（token 感知窗口）
```

对话历史的窗口管理使用 token 感知策略，超出预算时进一步用相关性过滤保留关键上下文：

```python
import numpy as np
import tiktoken
from openai import OpenAI

client = OpenAI()
enc = tiktoken.encoding_for_model("gpt-4o")


def _token_count(msg: dict) -> int:
    return len(enc.encode(msg["content"])) + 4    # +4 for role tokens


def build_dialogue_history(
    history: list[dict],
    current_task: str,
    token_budget: int = 6_000,
    min_recent: int = 4,     # 无论相关度，强制保留最近 2 轮（防止对话断层）
) -> list[dict]:
    """
    为 Dialogue Agent 构建对话历史：
    1. 先按 token 预算截断（从最新往前）
    2. 若候选消息仍过多，按与当前任务的语义相关度过滤，保留最相关的 top-k

    保留最近 min_recent 条不参与相关性过滤，保证对话连贯。
    """
    if not history:
        return []

    # ── 步骤一：token 感知窗口 ────────────────────────────────────────────
    within_budget: list[dict] = []
    used = 0
    for msg in reversed(history):
        t = _token_count(msg)
        if used + t > token_budget:
            break
        within_budget.insert(0, msg)
        used += t

    if len(within_budget) <= min_recent:
        return within_budget

    # ── 步骤二：相关性过滤（超出 min_recent 的部分按语义打分）───────────────
    forced_recent = within_budget[-min_recent:]
    candidates    = within_budget[:-min_recent]

    def _embed(text: str) -> list[float]:
        return client.embeddings.create(
            model="text-embedding-3-small", input=text
        ).data[0].embedding

    task_vec = np.array(_embed(current_task))
    scored = sorted(
        candidates,
        key=lambda m: float(np.dot(task_vec, np.array(_embed(m["content"])))),
        reverse=True,
    )

    # 只保留相关度 top-4 的历史，再按原始顺序重排
    keep_set = set(id(m) for m in scored[:4])
    ordered  = [m for m in candidates if id(m) in keep_set]

    return ordered + forced_recent
```

### Memory Extractor

```
目标：从编辑行为中提炼结构化偏好（异步，不在关键路径）

触发后接收：
  · 本次编辑完整记录（instruction + edit_plan + output_img_id）
  · 用户明确反馈（accept / reject + 可选原话）

输出：偏好条目 JSON 列表，写入 Profile Memory 或 Project Memory

提炼维度：
  dimension:  crop | color | text | layout | style | workflow
  polarity:   positive | negative
  content:    一句话（将直接注入 Planner 的 System Prompt）
  confidence: 0-1（反馈越明确，置信度越高）
  evidence:   用户原话或行为描述
```

以下示例展示 Memory Extractor 的完整生命周期——提炼偏好并写入向量数据库：

```python
from dataclasses import dataclass
from openai import OpenAI
import json

client = OpenAI()


@dataclass
class VisualPreference:
    user_id:    str
    project_id: str | None    # None = 跨项目通用偏好
    dimension:  str           # crop | color | text | layout | style | workflow
    polarity:   str           # positive | negative
    confidence: float         # 0-1
    content:    str           # 直接注入 Planner 的一句话（也作为 embedding 文本）
    evidence:   str           # 用户原话或行为（可溯源）
    hit_count:  int = 0       # 被检索命中的次数（频繁命中 = 更可信）
    embedding:  list[float] | None = None


class ProfileMemoryService:
    """用户偏好记忆服务（生产环境替换为 pgvector）"""

    def __init__(self):
        self._store: list[VisualPreference] = []    # 生产用 pgvector

    # ── 写入 ────────────────────────────────────────────────────────────────

    def extract_and_write(
        self, user_id: str, project_id: str,
        instruction: str, edit_plan: dict,
        feedback: str, accepted: bool,
    ) -> list[VisualPreference]:
        """Memory Extractor 核心方法：从编辑行为中提炼偏好并写入"""
        raw = json.dumps({
            "instruction": instruction,
            "edit_plan":   edit_plan,
            "feedback":    feedback,
            "accepted":    accepted,
        }, ensure_ascii=False)

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{
                "role": "system",
                "content": """从以下编辑记录中提炼用户的视觉偏好，输出 JSON：
{"preferences": [{"dimension": "crop|color|text|layout|style|workflow",
                  "polarity": "positive|negative",
                  "content": "一句话，将直接注入规划师的 System Prompt",
                  "confidence": 0.0-1.0,
                  "evidence": "用户原话或行为描述"}]}"""
            }, {
                "role": "user",
                "content": raw,
            }]
        )
        items = json.loads(resp.choices[0].message.content)["preferences"]

        written: list[VisualPreference] = []
        for item in items:
            pref = VisualPreference(
                user_id=user_id, project_id=project_id if not accepted else None,
                **item,
            )
            pref.embedding = self._embed(pref.content)
            self._store.append(pref)    # 生产：INSERT INTO visual_preferences ...
            written.append(pref)
        return written

    # ── 检索 ────────────────────────────────────────────────────────────────

    def recall(self, user_id: str, query: str, top_k: int = 3) -> list[str]:
        """按当前任务意图检索最相关的偏好条目，返回可直接注入 Prompt 的字符串列表"""
        import numpy as np
        user_prefs = [p for p in self._store if p.user_id == user_id]
        if not user_prefs:
            return []

        query_vec = np.array(self._embed(query))
        scored = sorted(
            user_prefs,
            key=lambda p: float(np.dot(query_vec, np.array(p.embedding))),
            reverse=True,
        )
        result = scored[:top_k]
        for p in result:
            p.hit_count += 1    # 记录命中次数
        return [p.content for p in result]

    def _embed(self, text: str) -> list[float]:
        resp = client.embeddings.create(model="text-embedding-3-small", input=text)
        return resp.data[0].embedding
```

---

## 5. 全局数据流

### 新项目启动

```
用户上传图像素材
    │
    ├─ 计算图像 CLIP embedding
    ├─ 查询 Project Memory（相似 embedding → 是否续作已有项目？）
    │
    ├─[新项目] 生成 project_id，初始化 Session Context
    └─[续作]   加载历史版本摘要到 Session Context
    │
    ▼
并行执行（互不依赖）：
    ├─ Vision Analyst 分析图像 → vision_result
    ├─ 检索 Profile Memory（用户偏好 top-3）
    └─ 检索 Design Knowledge Base（相似方案 top-2）
    │
    ▼（三路结果汇入 Session Context）
    │
Edit Planner（五层上下文）→ Edit Plan → Session Context
    │
    ├─[Plan 含素材需求] Asset Retrieval Agent → retrieved_assets → ImageState
    └─[Plan 不含素材需求] 跳过
    │
专项 / General Edit Agent（读取 retrieved_assets）→ 输出图像 v1 → Session Context（版本列表）
    │
Dialogue Agent → 向用户展示结果
```

### 用户迭代（对话阶段）

```
用户输入新指令
    │
    ▼
Intent Router → { intent_type, effort_level }
    │
    ├─[new_edit]   重新运行 Vision Analyst + Edit Planner（完整流程）
    │
    ├─[refine]     跳过 Vision Analyst，复用 vision_result
    │              Edit Planner（新指令 + 现有 vision_result）→ 新版本
    │
    ├─[compare]    Edit Planner 并行生成 N 个方向
    │              Image Executor × N → 多版本输出
    │
    ├─[accept]     标记 accepted = true
    │              触发 Memory Extractor（异步）
    │              Dialogue Agent 确认
    │
    ├─[undo]       从 Session Context 版本列表回滚
    │
    └─[query]      Dialogue Agent 直接回复（不触发 Planner / Executor）
```

---

## 6. 记忆写入协议

```
┌─────────────────────┬───────────────────┬────────────────────────┐
│ 触发信号             │ 写入目标           │ 写入内容               │
├─────────────────────┼───────────────────┼────────────────────────┤
│ 用户 accept         │ Project Memory     │ Edit Plan + 版本标注   │
│                     │ Profile Memory     │ 正向偏好条目（提炼）    │
│                     │ Design Knowledge   │ 满足质量阈值时入库      │
├─────────────────────┼───────────────────┼────────────────────────┤
│ 用户 reject         │ Profile Memory     │ 负向偏好条目（提炼）    │
│                     │ Project Memory     │ 拒绝记录（可溯源）      │
│                     │ Design Knowledge   │ 不写入                 │
├─────────────────────┼───────────────────┼────────────────────────┤
│ Session 正常结束     │ Project Memory     │ 完整编辑轨迹摘要        │
│                     │ Profile Memory     │ 会话级偏好批量提炼      │
├─────────────────────┼───────────────────┼────────────────────────┤
│ 用户隐式反馈         │ Profile Memory     │ 低置信度标注，积累后写  │
│ （继续迭代 ≈ 不满意） │                   │ 不立即写入，防止噪声    │
└─────────────────────┴───────────────────┴────────────────────────┘
```

**Profile Memory 条目 Schema：**

```python
@dataclass
class VisualPreference:
    user_id:    str
    project_id: str | None    # None = 跨项目通用偏好
    dimension:  str           # crop | color | text | layout | style | workflow
    polarity:   str           # positive | negative
    confidence: float         # 0-1，Memory Extractor 评估
    content:    str           # 直接注入 Planner 的一句话描述（也作为 embedding 文本）
    evidence:   str           # 用户原话或行为描述（可溯源）
    hit_count:  int           # 被检索使用的次数
    created_at: str
    embedding:  list[float]   # content 的向量，用于相似度检索
```

---

## 7. 可观测性

每个 Agent 执行后写入结构化 Trace 日志：

```python
@dataclass
class AgentTrace:
    trace_id:         str          # 全局唯一，贯穿整次请求
    session_id:       str
    agent_name:       str          # "intent_router" | "vision_analyst" | ...
    started_at:       str
    duration_ms:      int
    context_summary:  str          # system prompt 前 200 字
    memory_hits:      list[str]    # 命中的记忆条目 ID
    routing_decision: str          # 路由给哪个下游 Agent
    effort_level:     str | None   # 仅 Intent Router 填写
    tool_calls:       list[dict]   # [{tool, input_hash, success, duration_ms}]
    output_summary:   str          # 输出前 200 字
    success:          bool
    error:            str | None
```

**存储策略：**

```
Phase 1-2：
  · JSON Lines 按 session_id 分文件，支持按 trace_id 重播决策链路

Phase 3+：
  · Langfuse / LangSmith（LangGraph 生态原生支持）
  · 或 OpenTelemetry + Jaeger（技术栈无关）
```

---

## 8. 容错与检查点

### 设计原则

1. **不重来，只继续**：工具失败从失败点恢复，Session Context 天然充当检查点
2. **专项 Agent 失败 → 降级到 General Edit Agent**：用户无感知，系统自动兜底
3. **工具幂等性**：相同输入多次调用结果一致，可安全重试

### Session Context 作为检查点

```python
@dataclass
class SessionCheckpoint:
    session_id:           str
    last_completed_stage: str     # "intent_routing" | "vision_analysis"
                                  # | "edit_planning" | "image_execution"
    stage_outputs:        dict    # 各已完成阶段的输出，可直接复用
    failed_stage:         str | None
    failure_reason:       str | None
    version_snapshots:    list    # [{version_id, image_path, plan}]
```

以下是 Session Context 在 Redis 中的具体操作，展示检查点写入与容错恢复的实现路径：

```python
import redis
import json

class SessionContextService:
    """会话上下文服务：Redis 持久化 + 检查点管理"""

    TTL = 86_400    # 24 小时

    def __init__(self):
        self.r = redis.Redis(host="localhost", port=6379, decode_responses=True)

    # ── 阶段输出（检查点）────────────────────────────────────────────────────

    def save_stage(self, session_id: str, stage: str, output: dict) -> None:
        """保存阶段输出。容错恢复时直接读取，跳过已完成阶段的重跑"""
        self.r.hset(f"session:{session_id}:stages", stage, json.dumps(output))
        self.r.expire(f"session:{session_id}:stages", self.TTL)

    def load_stage(self, session_id: str, stage: str) -> dict | None:
        """读取已完成阶段的输出。返回 None 则需要重新执行该阶段"""
        data = self.r.hget(f"session:{session_id}:stages", stage)
        return json.loads(data) if data else None

    def recover(self, session_id: str) -> tuple[str, dict]:
        """恢复逻辑：返回（最后完成的阶段名，所有已完成阶段的输出）"""
        raw = self.r.hgetall(f"session:{session_id}:stages")
        stage_order = ["intent_routing", "vision_analysis", "edit_planning", "image_execution"]
        completed = {k: json.loads(v) for k, v in raw.items()}
        last = next((s for s in reversed(stage_order) if s in completed), "")
        return last, completed

    # ── 版本快照 ─────────────────────────────────────────────────────────────

    def push_version(self, session_id: str, version: dict) -> None:
        """追加新生成版本（image_path、plan、accepted 标志）"""
        key = f"session:{session_id}:versions"
        self.r.rpush(key, json.dumps(version))
        self.r.expire(key, self.TTL)

    def get_versions(self, session_id: str) -> list[dict]:
        """获取全部版本快照，供 undo 和展示使用"""
        return [json.loads(v) for v in self.r.lrange(f"session:{session_id}:versions", 0, -1)]

    def undo(self, session_id: str) -> dict | None:
        """弹出最新版本，返回上一个版本；无历史时返回 None"""
        key = f"session:{session_id}:versions"
        self.r.rpop(key)
        last = self.r.lindex(key, -1)
        return json.loads(last) if last else None

    # ── 编辑历史 ─────────────────────────────────────────────────────────────

    def push_edit_record(self, session_id: str, record: dict) -> None:
        """追加编辑记录 {instruction, plan_summary, feedback, accepted}"""
        key = f"session:{session_id}:edit_history"
        self.r.rpush(key, json.dumps(record))
        self.r.expire(key, self.TTL)

    def get_edit_history(self, session_id: str, last_n: int = 3) -> list[dict]:
        """取最近 N 轮编辑记录，注入 Edit Planner 的会话历史层"""
        items = self.r.lrange(f"session:{session_id}:edit_history", -last_n, -1)
        return [json.loads(i) for i in items]
```

### 错误降级链路

```
专项 Agent 执行失败
    │
    ├─ 自动重试（最多 2 次）
    │
    ├─ 重试失败 → 降级到 General Edit Agent（接管同一 Edit Plan）
    │
    └─ General Edit Agent 也失败
           ├─ 保留上一个成功版本（从 version_snapshots 取）
           └─ Dialogue Agent 返回友好错误 + 提供回滚选项
```

---

## 9. 技术栈

```
┌──────────────────┬──────────────────────────────────────────────────┐
│ 组件              │ 推荐选型                                          │
├──────────────────┼──────────────────────────────────────────────────┤
│ 执行内存          │ LangGraph State（进程内存）                        │
├──────────────────┼──────────────────────────────────────────────────┤
│ 会话上下文        │ Redis（TTL、断线续连）                              │
│                  │ key: session:{session_id}                         │
├──────────────────┼──────────────────────────────────────────────────┤
│ 项目记忆          │ PostgreSQL（结构化版本记录）                         │
│                  │ + pgvector（图像 embedding 相似检索）               │
├──────────────────┼──────────────────────────────────────────────────┤
│ 用户档案          │ pgvector 或 Chroma                                │
│                  │ 文本 embedding：text-embedding-3-small             │
├──────────────────┼──────────────────────────────────────────────────┤
│ 领域知识库        │ pgvector                                          │
│                  │ 图像 embedding：CLIP ViT-L/14                     │
├──────────────────┼──────────────────────────────────────────────────┤
│ Memory Extractor │ 异步任务队列（Celery / ARQ）                        │
├──────────────────┼──────────────────────────────────────────────────┤
│ 可观测性          │ Phase 1-2：JSON Lines                             │
│                  │ Phase 3+：Langfuse / LangSmith                    │
└──────────────────┴──────────────────────────────────────────────────┘
```

---

## 10. 分阶段实现路线

```
Phase 1：核心骨架
  · LangGraph State 管理各 Agent 中间产物
  · Session Context（Redis）持久化
  · 编辑历史结构化记录 {instruction, plan_summary, accepted}
  · Intent Router：6 种意图分类 + effort_level 评估
  · General Edit Agent 兜底
  · 工具层基础：图像编辑 + 文件存储，严格执行工具描述规范
  · 可观测性：JSON Lines Trace 日志
  · 容错：Session Context 作为检查点，工具层幂等性

Phase 2：项目记忆 + 并行
  · 项目识别：CLIP embedding 相似度判断是否续作
  · 版本管理：每次生成存 edit_plan + output_img + feedback
  · Edit Planner 注入项目历史摘要
  · 层次一并行上线：Vision Analyst + Profile 检索 + KB 检索三路并行
  · 专项 Agent 降级链路上线
  · 工具层扩展：CLIP、分割等视觉感知工具

Phase 3：用户档案 + 第一个专项 Agent
  · Memory Extractor 异步提炼（accept/reject 触发）
  · Profile Memory 向量存储，Edit Planner 注入偏好 top-3
  · 第一个专项 Agent 上线（建议 Compositing Agent）
  · 可观测性升级：接入 Langfuse / LangSmith

Phase 4：领域知识库 + 完整扩展
  · 收集多用户认可的优质 Edit Plan，匿名化后入库
  · 建立图像 CLIP embedding 索引
  · 历史编辑记录启用相关性过滤
  · 按需扩展专项 Agent（Style Transfer、Text Overlay 等）
  · compare 意图的多版本并行执行上线
  · 工具层完整：外部服务工具组（素材库、字体等）
```


