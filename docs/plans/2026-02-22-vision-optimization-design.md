# Vision 优化设计文档

**日期**: 2026-02-22
**方案**: 方案 A — 精准深化
**影响文件**: `src/mujoco_mcp/tools/vision.py`, `tests/test_vision.py`

---

## 目标

在四个维度各做最核心的一件改进，全面提升 Vision API 的分析质量、性能、功能覆盖和工程健壮性。

---

## Section 1 — 分析质量：prompt 意图路由

### 问题

`_build_system_prompt` 对所有 prompt 返回同一个通用模板，无论用户问的是碰撞力还是关节姿态，Gemini 收到的上下文引导完全相同。

### 设计

新增 `_detect_intent(prompt: str) -> str` 函数，通过关键词匹配将用户 prompt 分类为 4 种意图：

| 意图 | 触发关键词 | system prompt 特点 |
|---|---|---|
| `physics` | contact, touch, force, collision, friction, penetration | 突出接触力、穿透深度、摩擦系数 |
| `kinematics` | pose, position, joint, angle, where, extend, reach | 突出关节角度、末端位置、工作空间 |
| `comparison` | compare, difference, change, before, after, versus | 强调两状态间的变化量 |
| `general` | 默认 | 现有通用模板 |

`_build_system_prompt(m, d, intent="general")` 增加 `intent` 参数，根据意图在 **Response Format** 段注入专项引导：

- `physics`: 要求量化接触法向力估算、碰撞几何描述
- `kinematics`: 要求按关节逐一列出角度、用前向运动学描述末端位置
- `comparison`: 要求列出变化前后的具体数值差（Δ）

调用方在 `analyze_scene` 中：
```python
intent = _detect_intent(prompt)
system_prompt = _build_system_prompt(m, d, intent=intent)
```

---

## Section 2 — 性能：客户端缓存 + 图像格式自适应

### 问题

1. 每次 Gemini 调用重建 `genai.Client`，存在不必要的初始化开销
2. 所有图像固定使用 PNG，大尺寸图像传输慢、消耗 token 多

### 设计

**客户端缓存**：

```python
_gemini_client_cache: dict[str, "genai.Client"] = {}

def _get_client(api_key: str) -> "genai.Client":
    if api_key not in _gemini_client_cache:
        _gemini_client_cache[api_key] = genai.Client(api_key=api_key)
    return _gemini_client_cache[api_key]
```

`_call_gemini` 和 `_call_gemini_multi_image` 均改用 `_get_client(api_key)`。

**图像格式自适应**：

- 新增环境变量 `MUJOCO_MCP_VISION_JPEG_QUALITY`（默认 `85`）
- 规则：`width * height > 262144`（512²）时自动使用 JPEG，否则保持 PNG
- mime_type 随格式变化（`image/jpeg` vs `image/png`）

`_render_slot_png` 重命名为 `_render_slot_image`，返回 `(bytes, mime_type)` 元组。

---

## Section 3 — 功能扩展：`render_figure_strip` 工具

### 设计

从已录制的轨迹（`sim_record` + `sim_step`）中按指定时间戳渲染关键帧，适合生成论文图片。

**接口**：

```python
@mcp.tool()
@safe_tool
async def render_figure_strip(
    ctx: Context,
    timestamps: list[float],    # 要渲染的时间点列表（秒）
    sim_name: str | None = None,
    camera: str | None = None,
) -> list:  # [TextContent, ImageContent, ImageContent, ...]
```

**返回**：
- `TextContent`: `{"timestamps_rendered": [...], "nearest_frames": [...]}`
- 每个时间戳一个 `ImageContent`（PNG）

**实现逻辑**：
1. 检查 `slot.trajectory` 非空
2. 为每个 `timestamp` 在轨迹中找最近帧（按 `t` 字段二分查找）
3. `mj_set_state(qpos, qvel)` + `mj_forward` 恢复该帧状态，渲染快照
4. 渲染完成后恢复原始状态（保存/恢复 `qpos`/`qvel`/`time`）

**错误情况**：`slot.trajectory` 为空时返回 error TextContent 提示先用 `sim_record`。

---

## Section 4 — 工程质量：重试机制 + 错误码标准化

### 重试机制

新增 `_call_with_retry(fn, max_retries=3, base_delay=2.0)` 装饰所有 Gemini 调用：

```python
def _call_with_retry(fn, max_retries=3, base_delay=2.0):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
            if is_rate_limit and attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))  # 2s → 4s → 8s
                continue
            raise
```

### 错误码标准化

统一所有 Vision 工具的错误返回格式：

```json
{
  "error": "QUOTA_EXCEEDED",
  "message": "You exceeded your current quota...",
  "retry_after": 2.0
}
```

错误类型枚举（`_VisionError`）：

| 常量 | 触发条件 |
|---|---|
| `NO_API_KEY` | `GEMINI_API_KEY` 未设置 |
| `NO_GENAI` | `google-genai` 未安装 |
| `QUOTA_EXCEEDED` | 429 / RESOURCE_EXHAUSTED |
| `MODEL_NOT_FOUND` | 404 模型不存在 |
| `CAPTURE_FAILED` | 渲染失败 |
| `BODY_NOT_FOUND` | body 名称不存在 |
| `NO_TRAJECTORY` | `slot.trajectory` 为空 |

`_make_error(code, message, **kwargs)` 统一构造错误 JSON。

---

## 实现顺序

1. Section 4 工程基础（`_make_error` + `_call_with_retry` + `_get_client`）
2. Section 2 性能（`_render_slot_image` 重命名 + 格式自适应）
3. Section 1 分析质量（`_detect_intent` + `_build_system_prompt` intent 参数）
4. Section 3 功能（`render_figure_strip` 新工具）

每步独立可测，互不依赖。

---

## 测试策略

- Section 4：单元测试 `_make_error` 格式 + `_call_with_retry` 重试次数
- Section 2：mock 验证大图用 JPEG mime_type，小图用 PNG
- Section 1：`_detect_intent` 关键词分类测试（8-10 cases）
- Section 3：mock 轨迹 + 验证状态恢复后 qpos 不变
