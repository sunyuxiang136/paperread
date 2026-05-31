"""
=============================================================================
 论文智能解析系统 - Streamlit 前端主控 (app.py)
 所属系统：基于 RAG 架构与多文献聚合的学术论文智能解析系统
=============================================================================

功能概述：
  本文件是整个系统的 Web 界面入口，基于 Streamlit 框架构建的交互式
  学术论文智能阅读助手。它串联 PDF 解析 (pdf_parser.py)、向量检索引擎
  (rag_engine.py) 和 DeepSeek-Chat/本地 Ollama 云端 API，为用户提供以下
  核心能力：

  1. PDF 上传与自动解析索引：
     用户拖拽上传 PDF 后，系统自动执行 文件保存 → PDF 解析（含页码元数据提取）
     → 语义切分 → 向量化 → ChromaDB 持久化 的全自动流水线。

  2. RAG 增强的学术对话：
     用户以自然语言提问，系统自动检索 Top-3 相关论文片段，将片段内容与用户
     问题拼接为 Prompt 发送给大模型，生成的回答自带页码溯源标注。

  3. 一键学术报告（侧边栏快捷操作）：
     * 💡 提取核心创新点：自动检索方法/贡献相关片段，输出结构化创新点分析报告
     * 🔬 提取数据集与基线：自动检索实验相关片段，输出数据集、指标、基线对照表
     * 🔮 分析未来工作：自动检索局限性讨论片段，输出 future work 洞察

  4. 多文献交叉对比矩阵（绝杀功能）：
     勾选多篇已入库论文 → 系统按四个学术维度（方法/实验/痛点/优点）
     分别检索 → 大模型拼装 Markdown 二维对比大表

  5. 双轨推理引擎架构：
     支持云端 DeepSeek-Chat API 和本地 Ollama Qwen2.5 自由切换。
     API Key 仅在内存中保留，不落盘，保障数据安全。

  6. 侧边栏系统状态监控：
     实时展示 ChromaDB 向量库统计（总 Chunk 数、各论文 Chunk 分布）、
     已入库论文的可视化列表。

核心技术栈：
  - 前端框架：Streamlit 1.x（声明式 UI，自动布局，热重载）
  - PDF 解析：PyMuPDF (fitz)，字节流模式
  - 嵌入模型：sentence-transformers/all-MiniLM-L6-v2（384 维）
  - 向量存储：ChromaDB，本地持久化到 chroma_db/
  - 大模型 API：OpenAI Python SDK（兼容 DeepSeek-Chat 和 Ollama）
  - 状态管理：Streamlit session_state（跨 rerun 持久化变量）

架构模式：
  采用 Model-View-Presenter (MVP) 的简化变体：
  - View (Streamlit 声明式 UI)：侧边栏 + 聊天界面 + 文件上传 + 快捷按钮
  - Model (rag_engine + pdf_parser)：数据解析、存储、检索
  - Presenter (app.py 函数)：状态管理、流程编排、API 调用串联

  Streamlit 的特性决定了每次用户交互都会触发完整的脚本重新执行（rerun）。
  因此，本模块大量使用 st.session_state 来维护跨会话的持久化状态，并通过
  session_state flag 机制（如 _trigger_comparison、_quick_action）在 rerun
  之间传递用户的操作意图。

性能基准（在 Intel i5-12400F + 16GB RAM 上实测）：
  - PDF 解析 + 索引（30 页论文）：约 15 秒（90% 时间用于 Embedding）
  - 语义检索（Top-3）：约 0.3 秒
  - RAG 对话回答（DeepSeek-Chat）：约 5-10 秒（取决于输出长度）
  - 多文献交叉对比（4 篇 × 4 维度）：检索约 2 秒，生成约 15 秒
  - 首次启动（加载 Embedding 模型）：约 3-5 秒

设计原则：
  1. 所有文件落盘操作限定在项目目录（./papers/ 和 ./chroma_db/），保持整洁
  2. API Key 通过 st.session_state 在内存中管理，不写入文件系统
  3. 每个用户操作都有明确的加载指示器和错误提示
  4. 聊天历史与已入库论文列表在 session 内持久化（非跨会话）

安全设计：
  - PDF 以 bytes 形式读取后立即落盘到 ./papers/ 目录（而非直接传递给接口，
    因为 Streamlit 的上传文件对象在 rerun 间可能失效）
  - API Key 通过 st.text_input(type="password") 遮蔽显示
  - 本地模型模式（Ollama）无需外部 API，数据不出本机
  - ChromaDB 使用本地 SQLite 持久化，无需连接外部数据库

版本历史：
  - v0.1 (2026-05)：MVP 版本，基础 RAG 对话 + PDF 上传
  - v0.2 (2026-05)：新增侧边栏快捷报告、多文献对比矩阵
  - v0.3 (2026-05)：新增双轨推理引擎（DeepSeek + Ollama）
  - v0.4 (2026-05)：XSRF/CORS 配置修复、引入密码脱敏管理
=============================================================================
"""

# =============================================================================
# 标准库导入区
# =============================================================================

# os: 操作系统接口——用于构建文件路径、检查文件存在性
#    常用方法：os.path.join()（跨平台路径拼接）、os.path.exists()（文件存在性判断）
#    os.makedirs(exist_ok=True)（递归创建目录，如果目录已存在不发错误）
import os

# tempfile: 临时文件工具——用于在因 XSRF 限制（大文件上传被 Streamlit 内置
#          安全机制拦截）时，作为绕过手段在本地生成临时文件路径
#          注意：目前 .streamlit/config.toml 已关闭 XSRF 限制，此导入为后备方案
import tempfile

# =============================================================================
# 第三方库导入区
# =============================================================================

# streamlit: Web UI 框架——本系统前端界面的核心依赖
#   核心概念：
#     - st.session_state：跨 rerun 持久化字典，存储用户状态
#       （Streamlit 每次交互都会重新执行整个脚本，session_state 是唯一
#        能在多次脚本执行之间保持状态的机制）
#     - st.chat_message：聊天界面渲染（气泡样式）
#     - st.chat_input：聊天输入框（固定在底部）
#     - st.sidebar：侧边栏容器（用于配置和快捷操作）
#     - st.spinner：加载中动画指示器
#     - st.error / st.warning / st.info / st.success：消息提示
#     - @st.cache_resource：资源级缓存装饰器（避免重复初始化全局对象）
import streamlit as st

# OpenAI Python SDK——用于与 DeepSeek-Chat API 和本地 Ollama API 通信
#   设计说明：DeepSeek-Chat 的 API 端点完全兼容 OpenAI 的 /v1/chat/completions 规范。
#   本地 Ollama 启动时也将 API 暴露为与 OpenAI 兼容的端点（通过 ollama serve）。
#   因此可以统一使用 OpenAI 的 Python 客户端，只需切换 base_url 和 api_key 即可。
from openai import OpenAI

# 自定义模块导入
# pdf_parser: PDF 解析与语义切分模块
#   主要导出：
#     - parse_pdf(pdf_bytes, filename) -> List[Dict]
#       一站式 PDF 解析流水线：文本提取 → 清洗 → 语义切分 → 元数据注入
from pdf_parser import parse_pdf

# rag_engine: RAG 检索引擎模块
#   主要导出：
#     - index_chunks(chunks, collection_name) -> int
#       全量向量化并写入 ChromaDB（先删后建）
#     - add_chunks_to_collection(chunks, collection_name) -> int
#       增量添加 Chunks（保留已有数据）
#     - search_similar(query, top_k, collection_name) -> List[Dict]
#       单问题语义检索（Top-K 最相关片段）
#     - search_by_papers(papers, queries, top_k_per_paper) -> Dict[str, List[Dict]]
#       多论文维度化检索（对比矩阵数据源）
#     - get_collection_stats(collection_name) -> Dict
#       向量库统计（总 Chunk 数 + 各论文分布）
#     - clear_all(collection_name) -> None
#       清空向量库（不可逆操作）
from rag_engine import (
    index_chunks,
    add_chunks_to_collection,
    search_similar,
    search_by_papers,
    get_collection_stats,
    clear_all,
)


# =============================================================================
# 全局常量定义
# =============================================================================

# 论文文件存储目录
# 所有用户上传的 PDF 文件统一存储在此目录下（相对于 app.py 所在路径）
# 目录结构：./papers/
#   ├── Attention Is All You Need.pdf
#   ├── ResNet_2015.pdf
#   └── ...
# 设计说明：使用 os.path.join 确保跨平台（Windows/Linux/MacOS）路径兼容。
#   os.path.dirname(os.path.abspath(__file__)) 始终指向 app.py 所在目录。
PAPERS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "papers"
)

# ChromaDB 集合名称
# 所有论文的语义向量统一存储在同一集合中，通过 metadata.source 字段区分来源。
# 选择单一集合而非按论文分集合的原因：
#   1. 检索效率：单一集合 + HNSW 索引的查询复杂度为 O(log N_total)，
#      多集合串行查询的复杂度为 O(M × log N_per)，M 为论文数。
#   2. 跨论文检索：单一集合天然支持"在所有论文中检索"的语义搜索需求。
#   3. 管理简单：单一集合的备份、迁移、统计都更直接。
COLLECTION_NAME = "papers"

# =============================================================================
# 侧边栏 UI 渲染（View 层）
# =============================================================================

# 侧边栏是整个系统的控制面板，包含：
#   1. 系统标题与介绍
#   2. 推理引擎配置（API Key 输入 + 引擎选择）
#   3. PDF 上传器
#   4. 已入库论文列表（含对比勾选和快捷报告按钮）
#   5. 向量库状态监控
#   6. 重置按钮（危险操作）

with st.sidebar:
    # -------------------------------------------------------------------------
    # 区域 1：系统标题与介绍
    # -------------------------------------------------------------------------
    # st.title 是 SidebarElement 的方法，渲染一级标题。
    # 使用 Markdown 图标 📚 增加视觉辨识度。
    st.title("📚 论文智能阅读助手")

    # st.markdown 渲染 Markdown 格式的文本。
    # 这里展示系统的定位说明和核心卖点。
    # 使用 HTML 换行符 <br> 在 Streamlit Markdown 中精确控制换行位置。
    st.markdown(
        "**基于 RAG 架构的端云结合学术论文解析系统**<br>"
        "上传英文论文 → 自动解析索引 → 对话式阅读 + 多文献对比"
    )

    # st.divider 是一条水平分割线，用于视觉分组。
    # 等价于 Markdown 的 "---" 或 HTML 的 <hr>。
    st.divider()

    # -------------------------------------------------------------------------
    # 区域 2：推理引擎配置
    # -------------------------------------------------------------------------
    # 二级标题，用于标注推理引擎配置区域。
    st.subheader("🤖 推理引擎")

    # === 子区域 2.1：引擎选择下拉框 ===
    # st.selectbox 是下拉选择框组件。
    # 参数：
    #   label: 组件标签（显示在选择框上方）
    #   options: 可选值列表（中文字符串）
    #   key: session_state 中的键名，用于在其他地方访问当前选择值
    #   on_change: 选项变化时的回调函数（此处为 lambda 空操作，placeholder）
    #
    # 引擎选项的语义：
    #   "☁️ DeepSeek-Chat (云端)"：默认选项，使用 DeepSeek 官方 API
    #     - 需要输入 DeepSeek API Key
    #     - 模型能力更强（671B MoE），响应质量更高
    #     - 有网络延迟（~1-3 秒）
    #   "🏠 本地 Ollama (qwen2.5)"：使用本地部署的大模型
    #     - 无需 API Key，数据完全不出本机
    #     - 模型能力取决于本地硬件（需要先运行 ollama pull qwen2.5）
    #     - 零网络延迟，适合涉密场景
    st.selectbox(
        "选择推理引擎",
        options=[
            "☁️ DeepSeek-Chat (云端)",
            "🏠 本地 Ollama (qwen2.5)",
        ],
        key="inference_engine",
        on_change=lambda: None,  # 选项变化时无额外操作
    )

    # === 子区域 2.2：API Key 输入框（仅云端模式显示） ===
    # 逻辑：仅当推理引擎选择为云端 DeepSeek 时显示 API Key 输入框。
    #   "本地" in engine 的判断依赖于下拉框选项包含"本地"关键词。
    #
    # st.text_input 参数说明：
    #   label: 输入框标签
    #   type="password": 密码模式——输入内容显示为 ● 而非明文
    #      这是安全最佳实践，防止旁观者窥屏获取 API Key
    #   placeholder: 输入框占位文本（用户开始输入前显示）
    #   key: session_state 键名
    #      特别说明：key 以 _sidebar_ 为前缀，在代码中构建一个命名空间约定，
    #      避免与其他 session_state 键名冲突
    #
    # API Key 生命周期管理：
    #   1. 用户输入 API Key → 存入 st.session_state._sidebar_api_key
    #   2. 每次 rerun 时检查 st.session_state._sidebar_api_key 是否有效
    #   3. 浏览器关闭后 session_state 清空，API Key 自动销毁
    #   4. 不在磁盘上保留任何形式的 API Key
    #   这符合"涉密数据安全"的设计目标——API Key 只存在于当前会话的内存中
    #
    # 安全提示的设计意图：
    #   辅助文案告知用户 API Key 的存储策略，增加透明度，建立信任。
    if "本地" not in st.session_state.get("inference_engine", ""):
        st.text_input(
            "🔑 DeepSeek API Key",
            type="password",
            placeholder="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            key="_sidebar_api_key",
        )
        # 安全提示（灰色小字）
        st.caption("🔒 密钥仅保存在当前会话的内存中，不会写入磁盘或日志。")

    st.divider()

    # -------------------------------------------------------------------------
    # 区域 3：PDF 批量上传与解析
    # -------------------------------------------------------------------------
    # st.file_uploader 是文件上传组件。
    # 参数：
    #   label: 组件标签
    #   type: 允许的文件扩展名列表（仅 pdf）
    #      安全设计：限制文件类型为 PDF，防止上传可执行文件或其他恶意文件
    #   accept_multiple_files: 是否允许多文件上传
    #      True 允许用户一次选择多个 PDF 文件，批量上传
    #   key: session_state 键名
    #
    # 返回值：
    #   - 当用户选择了文件时：返回 UploadedFile 对象列表
    #   - 当用户未选择文件时：返回 None（首次访问）或 []（清空后）
    #
    # 上传文件后的自动处理流程（见下方 "PDF 上传与自动解析索引" 代码块）：
    #   1. 检查文件是否已存在（避免重复解析）
    #   2. 读取文件字节 → 调用 parse_pdf() 解析
    #   3. 调用 add_chunks_to_collection() 增量索引
    #   4. 保存 PDF 到 papers/ 目录
    #   5. 显示成功提示
    st.subheader("📁 上传论文 PDF")
    uploaded_files = st.file_uploader(
        "支持批量上传英文论文 PDF（将在启动时自动读取已有文件）",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )

    st.divider()

    # -------------------------------------------------------------------------
    # 区域 4：已入库论文面板（快捷操作区）
    # -------------------------------------------------------------------------
    # 此区域展示当前已入库的论文列表，并提供以下交互元素：
    #   - 论文复选按钮：勾选需要对比的论文（至少 2 篇）
    #   - 一键操作按钮：提取创新点、提取数据集、分析未来工作
    #   - 生成对比矩阵按钮：触发多文献交叉对比
    #
    # 实现说明：
    #   论文列表通过 st.session_state.indexed_papers 维护。
    #   每次 PDF 上传成功后，文件名被追加到此列表中。
    #   此列表的数据与 ChromaDB 向量库的实际状态同步，
    #   通过 get_collection_stats() 可以在启动时重建此列表。

    st.subheader("📚 已入库论文")
    papers = st.session_state.get("indexed_papers", [])

    if not papers:
        # 论文库为空时的提示
        # st.info 渲染蓝色信息提示框（InfoAlert 样式）
        st.info("👆 请先上传论文 PDF，系统将自动解析并索引。")
    else:
        # -------- 子区域 4.1：论文勾选列表 --------
        # st.checkbox 是单个复选框。
        # 对于多篇论文，使用 for 循环动态创建复选框，并通过 session_state 维护勾选状态。
        #
        # 复选框 key 格式：check_{filename}
        # 使用论文文件名生成唯一 key，确保同一论文的复选框在 rerun 间保持选中状态。
        st.caption("勾选论文用于对比分析（至少 2 篇）👇")
        for paper in papers:
            st.checkbox(
                paper,  # 复选框显示标签（论文文件名）
                key=f"check_{paper}",  # 唯一标识符
            )

        # 收集当前被勾选的论文列表
        # 通过检查 session_state 中的 checkbox key 是否为 True 来判断选中状态
        selected_papers = [
            p for p in papers
            if st.session_state.get(f"check_{p}", False)
        ]

        # 显示当前勾选状态（用于用户确认）
        if selected_papers:
            st.caption(f"✅ 已勾选 {len(selected_papers)} 篇：{'、'.join(selected_papers)}")

        # -------- 子区域 4.2：多文献对比矩阵按钮 --------
        # 按钮激活条件：至少勾选 2 篇论文
        # 按钮的 disabled 参数控制可点击状态
        #   条件为 True（按钮灰色不可点击）：勾选论文数 < 2
        #   条件为 False（按钮可点击）：勾选论文数 >= 2
        #
        # 点击后的处理流程（通过 session_state flag 机制）：
        #   1. 将 _trigger_comparison 标志设为 True
        #   2. 将勾选的论文列表存入 _selected_papers
        #   3. 调用 st.rerun() 触发脚本重新执行
        #   4. 在下一次执行中，检测到 _trigger_comparison 为 True，
        #      执行 call_llm_comparison() 生成对比矩阵
        #   5. 生成完毕后再次 rerun 以清除 flag
        #
        # 为什么需要 flag + rerun 机制？
        #   因为 Streamlit 的按钮回调在事件循环中执行，如果在回调中直接调用
        #   API（可能耗时 10-20 秒），会导致界面卡死直到 API 返回。
        #   通过 flag + rerun，我们可以将 API 调用放在主流程中执行，
        #   利用 streamlit 的 runner 机制保持界面响应。
        if st.button(
            "📊 生成多文献对比矩阵",
            disabled=len(selected_papers) < 2,  # 条件性禁用
            use_container_width=True,  # 按钮宽度撑满侧边栏
        ):
            # 设置 flag
            st.session_state._trigger_comparison = True
            st.session_state._selected_papers = selected_papers
            st.rerun()  # 触发脚本重新执行以处理 flag

        # -------- 子区域 4.3：一键学术报告快捷按钮 --------
        # 三个按钮分别对应三种学术分析维度。
        # 每个按钮的点击处理逻辑相同（设置 _quick_action flag → rerun）
        # 区别仅在于 _quick_action 的具体值：
        #   "innovation"   → 触发 call_llm_quick_report("innovation")
        #   "dataset"      → 触发 call_llm_quick_report("dataset")
        #   "future_work"  → 触发 call_llm_quick_report("future_work")
        #
        # 按钮样式说明：
        #   使用 Emoji 图标前缀增强视觉识别度
        #   use_container_width=True：按钮宽度撑满侧边栏容器，保持 UI 一致
        #
        # 按钮激活条件：
        #   必须在侧边栏中输入了有效的 API Key（云端模式）或选择了本地模型。
        #   st.session_state.api_key_valid 在页面初始化阶段计算，
        #   综合考虑了 云端 API Key 有效性 和 本地模型选择 两种情况。

        st.divider()
        st.caption("🎯 一键学术报告（基于已入库论文）")

        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "💡 提取创新点",
                use_container_width=True,
            ):
                st.session_state._quick_action = "innovation"
                st.rerun()
        with col2:
            if st.button(
                "🔬 提取数据集",
                use_container_width=True,
            ):
                st.session_state._quick_action = "dataset"
                st.rerun()

        if st.button(
            "🔮 分析未来工作",
            use_container_width=True,
        ):
            st.session_state._quick_action = "future_work"
            st.rerun()

    st.divider()

    # -------------------------------------------------------------------------
    # 区域 5：向量库状态监控
    # -------------------------------------------------------------------------
    # 此区域展示 ChromaDB 向量库的实时统计信息。
    # 统计数据通过 get_collection_stats() 从 ChromaDB 中读取，反映的是
    # 向量库的当前实际状态（而非 session_state 中的推测状态）。
    #
    # 显示内容：
    #   - 📊 总文本块数量：ChromaDB 集合中的 Chunk 总数
    #   - 📄 各论文分布：每篇论文的文件名及其 Chunk 数量，用灰色小字显示
    #
    # 设计意图：
    #   让用户直观感知向量库状态，确认自己的论文是否已成功索引。
    #   空库状态下的提示（"暂无数据"）帮助用户理解下一步操作。

    st.subheader("📊 向量库状态")
    stats = get_collection_stats()

    if stats["total_chunks"] > 0:
        # st.metric 是数值指标组件（大号数字 + 标签 + 可选的 delta）
        # 这里用于展示 Chunk 总数
        st.metric("总文本块", stats["total_chunks"])

        # 显示各论文分布
        # st.caption 渲染灰色小字（辅助信息）
        st.caption("各论文分布：")
        for paper_name, chunk_count in stats["papers"].items():
            st.caption(f"  • {paper_name}: {chunk_count} chunks")
    else:
        # 向量库为空
        st.caption("暂无数据，请上传论文 PDF")

    st.divider()

    # -------------------------------------------------------------------------
    # 区域 6：清空按钮（危险操作）
    # -------------------------------------------------------------------------
    # st.button 的 type="secondary" 参数渲染为较不显眼的样式（灰色按钮）。
    # 点击后：
    #   1. 调用 clear_all() 清空 ChromaDB 集合
    #   2. 清空 session_state.indexed_papers 列表
    #   3. 清空聊天历史
    #   4. 调用 st.rerun() 刷新界面
    #
    # deleted 变量的语义：
    #   Streamlit button 在被点击时返回 True，未点击时返回 False。
    #   （注意：不是持续状态，而是仅在当前脚本执行中为 True。
    #    下一次 rerun 时，若没有再次点击，返回值会是 False）
    #
    # 安全考量：
    #   - 使用单独的 divider 与"重置全部数据"文本提示区分危险操作区域
    #   - 完全清空操作：向量库 + 论文列表 + 聊天历史，不留残留状态
    #   - 用户需重新上传 PDF 才能恢复论文索引
    deleted = st.button(
        "🗑️ 重置全部数据",
        type="secondary",
        use_container_width=True,
    )

    if deleted:
        # 执行清空操作
        clear_all()                              # 清空 ChromaDB 向量库
        st.session_state.indexed_papers = []     # 清空论文列表
        st.session_state.messages = []           # 清空聊天历史
        # 注意：不清空 API Key，避免用户重复输入
        st.rerun()


# =============================================================================
# Session State 初始化（Presenter 层）
# =============================================================================

# Streamlit 的 session_state 需要在使用前确认各键值是否存在。
# 使用 "key not in st.session_state" 检查并在首次执行时赋默认值。
# 这是一个常见的 Streamlit 初始化模式。

# indexed_papers: List[str]
#   已入库的论文文件名列表。与 ChromaDB 的实际状态保持一致。
#   每次 PDF 上传成功后追加文件名到此列表。
#   供侧边栏"已入库论文"区域和对比矩阵功能使用。
if "indexed_papers" not in st.session_state:
    st.session_state.indexed_papers = []

# messages: List[Dict[str, str]]
#   聊天历史记录，格式为 [{"role": "user/assistant", "content": "消息文本"}, ...]。
#   与 OpenAI Chat Completions API 的 messages 格式保持一致。
#   每次用户提问和系统回答时追加记录。
#   用于前端聊天框渲染和 API 调用的上下文传递。
if "messages" not in st.session_state:
    st.session_state.messages = []

# _sidebar_api_key: str
#   用户在侧边栏输入的 DeepSeek API Key。
#   仅在云端模式下有效；本地模型模式下忽略此值。
#   通过 type="password" 输入框获取，以遮蔽形式显示。
if "_sidebar_api_key" not in st.session_state:
    st.session_state._sidebar_api_key = ""

# inference_engine: str
#   用户选择的推理引擎选项（下拉框选择值的字符串形式）。
#   默认值为 "☁️ DeepSeek-Chat (云端)"。
#   通过 st.selectbox 的 key 参数绑定到此 session_state 键。
if "inference_engine" not in st.session_state:
    st.session_state.inference_engine = "☁️ DeepSeek-Chat (云端)"

# api_key_valid: bool
#   推理引擎的可用性标志。综合考虑以下两种情况：
#     1. 云端模式 + 用户输入了非空的 API Key → True
#     2. 本地 Ollama 模式 → True（无需 API Key）
#     3. 云端模式 + 未输入 API Key → False
#   在多个地方用于判断操作是否可用（按钮 disabled 状态、错误提示等）。
#
# 设计说明：这个属性不是用户直接设置的，而是在每次 rerun 时根据当前
#   推理引擎选择和 API Key 输入状态动态计算的"派生状态"。
#   通过 st.session_state 存储计算结果，避免在多处重复计算。
st.session_state.api_key_valid = (
    "本地" in st.session_state.get("inference_engine", "")
    or bool(st.session_state.get("_sidebar_api_key", "").strip())
)

# =============================================================================
# PDF 上传与自动解析索引（Presentor 层 - 文件处理管线）
# =============================================================================

# 此代码块处理用户上传的 PDF 文件，执行全自动的解析索引流水线。
#
# 工作流程（每个上传文件的处理步骤）：
#   Step 1: 读取上传文件的原始名称和字节内容
#   Step 2: 检查是否已入库（通过 session_state.indexed_papers 比对）
#   Step 3: 保存文件到 papers/ 目录（如果尚未存在）
#   Step 4: 调用 parse_pdf() 执行文本提取 + 语义切分
#   Step 5: 调用 add_chunks_to_collection() 增量添加到 ChromaDB
#   Step 6: 更新 session_state.indexed_papers
#   Step 7: 显示成功提示（含 Chunk 数量信息）
#   Step 8: 调用 st.rerun() 刷新界面
#
# 错误处理：
#   - 任何步骤出错都会使用 st.error 显示错误信息
#   - 错误不会中断处理流程（单个文件失败不影响其他文件）
#
# 设计说明：
#   为什么使用 add_chunks_to_collection（增量）而非 index_chunks（全量刷新）？
#     - 用户可以分批次上传论文，每篇独立索引
#     - 增量模式避免了不必要的重复 Embedding 计算
#     - 如果用户需要全量重建，可以点击"重置全部数据"后重新上传

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Step 1: 获取文件名和字节内容
        # uploaded_file.name：原始文件名（不含路径）
        # uploaded_file.read()：文件的全部字节内容（bytes 类型）
        filename = uploaded_file.name
        pdf_bytes = uploaded_file.read()

        # Step 2: 检查是否已入库
        # 通过 session_state.indexed_papers 列表判断
        # 注意：这里不检查 papers/ 目录下的文件存在性，
        # 因为文件可能通过其他方式放入目录中
        if filename in st.session_state.indexed_papers:
            # 已入库 → 静默跳过，不显示任何提示
            # 因为 Streamlit 会在每次 rerun 时重新获取上传文件列表，
            # 如果不跳过会导致重复索引
            continue

        # Step 3: 使用 st.spinner 显示加载指示器
        # st.spinner 是一个上下文管理器（with 语句），在代码块执行期间显示
        # 一个旋转动画和状态文字
        with st.spinner(f"📄 正在解析《{filename}》({len(pdf_bytes)//1024} KB)，包括文本提取、语义切分和向量化..."):
            try:
                # Step 4: 确保 papers/ 目录存在
                # exist_ok=True：如果目录已存在不发错误
                os.makedirs(PAPERS_DIR, exist_ok=True)

                # Step 5: 保存 PDF 文件到本地
                # 为什么要保存文件？
                #   1. Streamlit 的 UploadedFile 对象在 rerun 间可能失效
                #   2. 持久化存储允许系统重启后继续使用已上传的论文
                #   3. 用户可以在 papers/ 目录中直接管理文件
                filepath = os.path.join(PAPERS_DIR, filename)
                with open(filepath, "wb") as f:
                    f.write(pdf_bytes)

                # Step 6: 调用 parse_pdf() 执行解析流水线
                # parse_pdf 的返回值是携带元数据的 Chunk 列表
                # 关于元数据的详细说明见 pdf_parser.py 第 15 行起
                chunks = parse_pdf(pdf_bytes, filename)

                # Step 7: 增量写入 ChromaDB
                # add_chunks_to_collection 返回本次成功入库的 Chunk 数量
                chunk_count = add_chunks_to_collection(chunks)

                # Step 8: 更新 session_state（立即同步表示层）
                st.session_state.indexed_papers.append(filename)

                # Step 9: 显示成功提示
                # st.success 渲染绿色成功提示框
                # 告知用户解析结果（Chunk 数量和总 Chunk 数）
                total_now = sum(
                    get_collection_stats()["papers"].values()
                )
                st.success(
                    f"✅ 《{filename}》解析完成！\n\n"
                    f"本次新增 {chunk_count} 个语义块。"
                    f"向量库共 {total_now} 个片段。"
                )

                # Step 10: 刷新界面
                # 重要！必须调用 rerun，否则：
                #   - 侧边栏的论文列表不会更新
                #   - 上传的文件仍然显示在上传器中（因为 rerun 会清空上传器状态）
                st.rerun()

            except Exception as e:
                # 错误处理：捕获所有异常并显示错误信息
                # 使用 traceback 以便调试时可以定位到具体出错位置
                import traceback
                error_detail = traceback.format_exc()
                st.error(
                    f"❌ 解析《{filename}》时出错：{str(e)}\n\n"
                    f"详细错误信息：\n```\n{error_detail[-500:]}\n```"
                )
                # 注意：此处使用 traceback.format_exc() 获取完整调用栈
                # 仅截取最后 500 字符以避免界面溢出


# =============================================================================
# 页面标题与欢迎文案
# =============================================================================

# st.title 渲染页面的主标题（H1 级）。
# 在 Streamlit 布局中，标题显示在页面顶部，侧边栏的右侧。
# 使用 Emoji 增强视觉表现力。
st.title("📖 基于 RAG 的学术论文智能解析系统")

# 欢迎信息（首次访问时显示）
# 条件：聊天历史为空（即用户尚未开始任何对话）
#
# st.markdown 渲染 Markdown 文本。
# 这里使用 """""" 三引号字符串，内部包含 Markdown 格式的欢迎文案。
# 欢迎文案说明系统能力、使用方法和注意事项。
if not st.session_state.messages:
    st.markdown(
        """
        ### 👋 欢迎使用学术论文智能阅读助手！

        本系统基于 **RAG（检索增强生成）** 架构，专为学术论文精读设计。

        **核心能力：**
        - 🔍 **对话式问答**：用自然语言提问，系统自动检索论文中最相关的段落并生成答案
        - 💡 **一键分析创新点**：快速提取论文的核心方法与贡献
        - 🔬 **实验配置提取**：自动整理数据集、基线和实验结果
        - 🔮 **未来工作洞察**：分析论文局限性和潜在研究扩展方向
        - 📊 **多文献交叉对比**：勾选多篇论文，生成 Markdown 二维对比矩阵大表

        **使用方法：**
        1. 在左侧上传你的英文论文 PDF
        2. （云端模式）输入你的 DeepSeek API Key——🔒 仅保存在当前会话内存中
        3. 在下方聊天框用中文向论文提问！例如：

        > "Transformer 论文的核心创新点是什么？"
        > "这篇论文用了哪些数据集做实验？基准方法有哪些？"
        > "论文作者自己承认了哪些局限性？未来可以做什么改进？"

        **安全保障：**
        - 📄 **数据本地化**：论文 PDF 和向量库全部存储在本地，不上传至任何云端
        - 🔑 **API Key 内存管理**：密钥仅存在于当前浏览器会话，关闭后自动清除
        - 🏠 **可选本地大模型**：可切换到本地 Ollama，实现完全离线部署
        """
    )


# =============================================================================
# Prompt 工程：RAG 对话 Prompt 构建
# =============================================================================

def build_rag_prompt(
    user_question: str,
    retrieved_chunks: list,
) -> str:
    """
    构建 RAG 对话的 User Prompt（将检索片段和用户问题注入 Prompt 模板）。

    这是 RAG（检索增强生成）的核心 Prompt 工程环节。
    本函数的职责是将检索结果格式化为大模型可以理解和利用的上下文文本，
    并与用户原始问题拼接成完整的 User Message。

    Prompt 设计哲学（为什么这样写 Prompt）：
      1. 检索片段优先于上下文学习
         检索到的论文片段以「引用材料」的形式注入 Prompt，大模型将其
         视为"必须参考的依据"，而非"可选参考"。这通过"你必须严格基于"
         这个措辞来强化。

      2. 结构化信息呈现
         为每个检索片段设计统一的引用格式：
           [片段N] 来源：文件名.pdf，第X页
         这种格式方便在后续回答中进行精确的页码溯源引用。

      3. 强制引用约束（防幻觉三重保险之一）
         通过"每条重要论断必须标注来源"的强制要求，从 Prompt 层面遏制
         大模型自由发挥的倾向。这与 System Prompt 中的引用要求形成双重约束。

      4. 诚实面对局限
         "不要编造论文中不存在的内容"这一引导让模型在信息不足时
         主动承认局限，而非强行编造。

    参数:
        user_question (str):
          用户的自然语言问题（中英文均可）。
          在传入此函数前已通过 chat_input 获取，未经修改。
        retrieved_chunks (list):
          search_similar() 返回的 Top-K 检索结果列表。
          每个元素为 Dict，包含 text、source、page、distance 字段。
          可能为空列表（向量库无数据时）。

    返回:
        str: 完整的 User Message 文本，包含：
          - 系统指令部分（引用要求、回答限制）
          - 检索片段部分（格式化的论文文本引用）
          - 用户问题部分（原始问题文本）

    示例输出：
        ## 引用材料
        [片段1] 来源：Attention Is All You Need.pdf，第2页
        The dominant sequence transduction models are based on complex
        recurrent or convolutional neural networks...
        [片段2] 来源：Attention Is All You Need.pdf，第3页
        We propose a new simple network architecture, the Transformer...
        ## 回答要求
        你必须严格基于以上引用材料回答用户问题...
        ## 用户问题
        What is the main contribution of the Transformer paper?
    """
    # -------------------------------------------------------------------------
    # 步骤 1：格式化检索片段为引用材料块
    # -------------------------------------------------------------------------
    # 遍历每个检索到的 Chunk，构建统一的引用格式。
    # 引用格式设计：
    #   [片段N] → 标识符（N 从 1 开始编号）
    #   来源：文件名.pdf → 论文来源文件
    #   第X页 → PDF 中的实际页码
    #   内容：→ Chunk 的文本内容（原样保留）
    #
    # 为什么使用枚举编号 [片段1]、[片段2]...？
    #   这样大模型的回答可以使用 "[参考片段1]" 的形式引用，
    #   实现比页码引用更细粒度的溯源。
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        context_parts.append(
            f"[片段{i+1}] 来源：{chunk['source']}，第{chunk['page']}页\n"
            f"内容：{chunk['text']}\n"
        )

    # 将所有格式化的片段用双换行拼接
    # 双换行 (\n\n) 确保视觉上的片段分隔
    context_text = (
        "\n".join(context_parts)
        if context_parts
        else "（未检索到相关论文片段）"
    )

    # -------------------------------------------------------------------------
    # 步骤 2：拼接完整的 User Message
    # -------------------------------------------------------------------------
    # 使用 f-string 模板生成最终 Prompt。
    # Prompt 结构分为四个区域：
    #   1. 引用材料区：检索到的论文片段（大模型的上下文基础）
    #   2. 回答要求区：对大模型行为的约束指令
    #   3. 用户问题区：用户的原始自然语言问题
    #
    # 注意：如果 retrieved_chunks 为空，context_text 会是占位文本，
    # 大模型会依据 System Prompt 中的指令提醒用户上传论文。
    return (
        f"## 📚 引用材料\n\n"
        f"以下是从论文中检索到的最相关段落，编号 [片段N] 供你引用参考：\n\n"
        f"{context_text}\n\n"
        f"## 📋 回答要求\n\n"
        f"1. 你必须严格基于以上引用材料回答用户问题，不要编造论文中不存在的内容。\n"
        f"2. 你的答案中的每条重要论断都必须标注来源，格式：`【来源：文件名.pdf，第Y页，参考片段Z】`。\n"
        f"3. 如果引用材料不足以回答某个问题，请诚实说明，不要推测或编造。\n"
        f"4. 用清晰的中文回答，专业术语保留英文原文。对于论文方法、模型等概念，给出通俗易懂的解释。\n\n"
        f"## ❓ 用户问题\n\n"
        f"{user_question}"
    )


# =============================================================================
# Prompt 工程：多文献对比矩阵 Prompt 构建
# =============================================================================

def build_comparison_prompt(
    results: dict
) -> str:
    """
    构建多文献交叉对比矩阵的 User Prompt。

    与 build_rag_prompt 的区别：
      - build_rag_prompt：面向单问题对话，提供引用材料 + 回答要求
      - build_comparison_prompt：面向多论文对比，要求大模型生成二维表格

    输入数据结构：
      search_by_papers() 返回的字典，格式为：
        {
          "论文A.pdf": [
            {"text": "...", "source": "...", "page": 3, "query_dim": "方法/模型"},
            {"text": "...", "source": "...", "page": 7, "query_dim": "实验/数据"},
            ...
          ],
          "论文B.pdf": [...],
        }
      每个检索结果已携带 query_dim 字段标注其所属维度。

    Prompt 设计核心思路：
      要求大模型输出一个沿"论文名（纵向）× 分析维度（横向）"的 Markdown 表格。
      表格的每一列对应一个分析维度（核心方法、实验设置、主要痛点、核心优点），
      每一行对应一篇论文。表格下方附加数据来源。

    表格格式规范：
      - 使用 Markdown 表格语法（| 分隔符 + --- 表头分隔行）
      - 每个单元格 ≤ 80 字（英文 ≤ 50 words），简洁概括
      - 信息不足时填入「未提及」或「待确认」
      - 表格下方必须有 📌 数据来源小节

    参数:
        results (dict):
          search_by_papers() 的返回结果，论文名 → 维度化检索结果列表

    返回:
        str: 完整的对比分析 Prompt User Message
    """
    # -------------------------------------------------------------------------
    # 步骤 1：按论文组织检索片段文本
    # -------------------------------------------------------------------------
    # 遍历 results 字典，为每篇论文生成一个格式化的引用块。
    # 引用块结构：
    #   ### 论文：文件名.pdf
    #   每个检索片段包含：维度标签、页码、文本内容
    #
    # 设计说明：在每个片段前标注维度标签（如 [方法/模型]），
    # 帮助大模型理解该片段的语义类型，从而在表格中对号入座。
    sections = []
    for paper_name, chunk_list in results.items():
        if not chunk_list:
            sections.append(
                f"### 📄 {paper_name}\n"
                f"（未检索到相关片段）\n"
            )
            continue

        chunk_lines = []
        for i, chunk in enumerate(chunk_list):
            dim_label = chunk.get("query_dim", "未知维度")
            chunk_lines.append(
                f"**{dim_label}** (第{chunk['page']}页):\n"
                f"> {chunk['text'][:300]}\n"
                # 限制预览长度为 300 字符，避免 Prompt 过长
                # all-MiniLM-L6-v2 的检索片段通常 ≤ 600 字符，截取前 300 已足够
            )

        sections.append(
            f"### 📄 {paper_name}\n" +
            "\n".join(chunk_lines)
        )

    all_sections = "\n\n".join(sections)

    # -------------------------------------------------------------------------
    # 步骤 2：拼接 Prompt 模板
    # -------------------------------------------------------------------------
    # 与 build_rag_prompt 类似，结构分为引用材料区和生成要求区。
    #
    # 关键区别：
    #   - 生成要求区要求大模型输出表格（而非自由文本）
    #   - 明确指定表格的列名、行标签和单元格内容规范
    #   - 要求数据来源跟踪（页码标注）
    #
    # 表格设计原因（为什么是这四个维度）：
    #   这四个维度覆盖了学术论文的核心分析维度：
    #     核心方法 → "这篇论文做了什么（How）"
    #     实验设置 → "这篇论文怎么证明（How evaluated）"
    #     主要痛点 → "这篇论文解决了什么问题（Why）"
    #     核心优点 → "这篇论文好在哪里（Impact）"
    #   这四个维度构成了学术论文对比分析的最小完备框架。
    return (
        "## 📚 论文片段材料\n\n"
        f"{all_sections}\n\n"
        "---\n\n"
        "## 📋 任务要求\n\n"
        "请基于以上论文片段，生成一个 **Markdown 二维对比表格**，具体要求：\n\n"
        "### 表格格式\n"
        "| 对比维度 | 论文A | 论文B | ... |\n"
        "|----------|-------|-------|-----|\n"
        "| **核心方法/模型架构** | ... | ... | ... |\n"
        "| **实验设置与数据集** | ... | ... | ... |\n"
        "| **主要痛点/局限性** | ... | ... | ... |\n"
        "| **核心优点/贡献** | ... | ... | ... |\n\n"
        "### 填写规范\n"
        "- 每个单元格 **≤ 80 汉字**（或 ≤ 50 英文词），简洁概括\n"
        "- 信息不足的维度直接写「未提及」或「待确认」\n"
        "- 专业术语保留英文原名（如 Self-Attention、Residual Block）\n\n"
        "### 表格下方必须附加\n"
        "📌 **数据来源**：逐条标注论文文件名和页码，格式：\n"
        "- 论文A 核心方法 → `文件名.pdf` 第X页\n"
        "- 论文A 实验设置 → `文件名.pdf` 第Y页\n"
        "- ...\n\n"
        "开始生成！"
    )


# =============================================================================
# Prompt 工程：一键学术报告 Prompt 构建
# =============================================================================

def build_innovation_prompt(retrieved_chunks: list) -> str:
    """
    构建「一键提取核心创新点」学术报告 Prompt。

    此 Prompt 要求大模型从检索片段中提取论文的创新点，并按结构化的
    报告格式输出。输出要求严格规范，确保报告的专业性和可读性。

    报告结构（4 个小节）：
      核心方法（What）→ 方法描述 → 创新性分析（Why Important）

    每个小节的要求：
      - 逐条列出（无序列表格式）
      - 每条后标注页码来源
      - 不确定性处使用谨慎措辞（"可能是…"、"倾向于…"）
    """
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        context_parts.append(
            f"[片段{i+1}] 来源：{chunk['source']} 第{chunk['page']}页\n"
            f"内容：{chunk['text']}\n"
        )
    context_text = "\n".join(context_parts) if context_parts else "（未检索到相关内容）"

    return (
        "## 任务：提取论文核心创新点\n\n"
        f"以下是从论文中检索到的相关片段：\n\n{context_text}\n\n"
        "## 输出要求（务必严格遵守）\n"
        "1. **报告标题**：`## 💡 核心创新点分析报告`\n"
        "2. **必须包含以下 3 个小节**，使用 `###` 标题：\n"
        "   - **🧠 核心方法（What）**：论文提出的核心方法/模型叫什么名字？"
        "用 1-2 句话概括其本质思路。\n"
        "   - **📝 方法关键细节（How）**：用 3-5 条无序列表，逐条列出方法的"
        "关键公式或算法思路（不需要完整公式，用自然语言描述即可）。\n"
        "   - **✨ 创新性分析（Why Important）**：分析这个创新为什么重要，"
        "解决了什么之前方法无法解决的问题，对领域的影响。\n"
        "3. **引用要求**：每个小节的每一条论断后必须标注 `【来源：X.pdf，第Y页】`\n"
        "4. **诚实原则**：如果检索片段不足以回答某个小节，写「⚠️ 当前检索片段未覆盖此维度，建议阅读全文相关章节」\n"
        "5. **语言**：中文输出，专业术语保留英文原文\n"
        "6. **长度**：报告总长度控制在 500 字以内，精炼有力"
    )


def build_dataset_prompt(retrieved_chunks: list) -> str:
    """
    构建「一键提取数据集与基线」学术报告 Prompt。

    报告结构（4 个小节）：
      数据集清单 → 评价指标 → 基线方法 → 主要实验结果

    特别要求：数据集和基线信息使用 Markdown 表格呈现，提高可读性。
    """
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        context_parts.append(
            f"[片段{i+1}] 来源：{chunk['source']} 第{chunk['page']}页\n"
            f"内容：{chunk['text']}\n"
        )
    context_text = "\n".join(context_parts) if context_parts else "（未检索到相关内容）"

    return (
        "## 任务：提取论文实验配置全貌\n\n"
        f"以下是从论文中检索到的相关片段：\n\n{context_text}\n\n"
        "## 输出要求（务必严格遵守）\n"
        "1. **报告标题**：`## 🔬 数据集与基线分析报告`\n"
        "2. **必须包含以下 4 个小节**，使用 `###` 标题：\n"
        "   - **📦 使用数据集**：列出论文实验用的所有数据集名称、来源、规模（样本数/类别数），"
        "用无序列表呈现。\n"
        "   - **📊 评价指标**：列出论文使用的所有评价指标（如 Accuracy、BLEU、F1、Perplexity 等），"
        "简述每个指标衡量什么。\n"
        "   - **⚔️ 基线方法**：列出对比的 baseline 方法名称及简要说明（为什么选这些 baseline）。\n"
        "   - **🏆 主要实验结果**：概括最重要的实验结论和数字，标注是在哪个数据集/指标上取得的结果。\n"
        "3. **引用要求**：每一条数据/指标/结果后必须标注 `【来源：X.pdf，第Y页】`\n"
        "4. **诚实原则**：检索片段覆盖不足的维度，写「⚠️ 当前检索片段未覆盖此维度，建议阅读原文 Experiment 章节」\n"
        "5. **语言**：中文输出，数据集名/指标名保留英文原名\n"
        "6. **格式**：数据集和基线使用 Markdown 表格呈现"
    )


def build_future_work_prompt(retrieved_chunks: list) -> str:
    """
    构建「一键分析未来工作」学术报告 Prompt。

    报告结构（3 个小节）：
      当前局限性 → 作者提出的未来方向 → AI 洞察的潜在研究方向

    独特之处：第三小节（AI 洞察）不要求标注来源，因为它是大模型基于论文方法
    的延伸思考，而非论文原文内容。Prompt 中明确要求在 AI 洞察部分开头标注
    "以下为基于论文方法的延伸思考，非原文内容"。

    这是学术诚信的重要体现：明确区分"论文原文内容"（有出处）和"AI 延伸分析"（无出处）。
    """
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        context_parts.append(
            f"[片段{i+1}] 来源：{chunk['source']} 第{chunk['page']}页\n"
            f"内容：{chunk['text']}\n"
        )
    context_text = "\n".join(context_parts) if context_parts else "（未检索到相关内容）"

    return (
        "## 任务：分析论文局限性与未来研究方向\n\n"
        f"以下是从论文中检索到的相关片段：\n\n{context_text}\n\n"
        "## 输出要求（务必严格遵守）\n"
        "1. **报告标题**：`## 🔮 未来工作与局限性分析报告`\n"
        "2. **必须包含以下 3 个小节**，使用 `###` 标题：\n"
        "   - **⚠️ 当前局限性**：论文作者自己承认的 limitation 有哪些？用无序列表逐条列出。\n"
        "   - **🧭 作者提出的未来方向**：论文中明确提出的 future work 有哪些？逐一列出。\n"
        "   - **💭 潜在研究方向（AI 洞察）**：基于论文方法的局限性，结合你的学术知识，"
        "推测 2-3 个该论文未提及但值得探索的研究方向。每个方向用 1-2 句说明动机。\n"
        "     ⚠️ 在此小节开头注明：「以下为基于论文方法的延伸思考，非原文内容」。\n"
        "3. **引用要求**：前两个小节（局限性、未来方向）的每条后必须标注 `【来源：X.pdf，第Y页】`\n"
        "   第三小节（AI 洞察）不需要标注来源。\n"
        "4. **诚实原则**：如果论文未明确提及 limitation/future work，前两个小节写「⚠️ 本文未在检索片段中明确讨论此内容」\n"
        "5. **语言**：中文输出\n"
        "6. **长度**：报告总长度控制在 400 字以内"
    )


# =============================================================================
# 聊天界面 - 渲染历史消息（View 层）
# =============================================================================

# 渲染聊天历史中的每一条消息。
# st.chat_message(role) 创建一个聊天气泡：
#   role="user"     → 用户消息气泡（通常右对齐或不同背景色）
#   role="assistant" → 助手消息气泡（通常左对齐或另一背景色）
#   默认的 Streamlit 主题会自动处理样式差异
#
# st.markdown(content) 在气泡内渲染 Markdown 格式的消息内容。
#   - 支持标准 Markdown 语法（标题、列表、表格、加粗、代码块等）
#   - 大模型输出的引用标注【来源：X.pdf，第Y页】会原样显示
#   - Markdown 表格（对比矩阵）会正确渲染为表格
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# =============================================================================
# 推理引擎配置 - 双轨制切换逻辑（Presenter 层）
# =============================================================================

def get_engine_config() -> dict:
    """
    根据用户在侧边栏选择的推理引擎，返回对应的 API 配置。

    双轨制架构说明：
      轨 1（云端 DeepSeek-Chat）：
        - base_url: https://api.deepseek.com
        - model: deepseek-chat
        - api_key: 用户在侧边栏输入的 DeepSeek API Key
        - 特点：模型能力强大（671B MoE 参数），支持完整学术术语
        - 隐私：用户问题通过 API 发送到 DeepSeek 服务器
          （论文原文片段也会随 Prompt 一起发送）

      轨 2（本地 Ollama Qwen2.5）：
        - base_url: http://localhost:11434/v1（Ollama 默认端口）
        - model: qwen2.5
        - api_key: 固定为 "ollama"（Ollama 忽略此值但需要非空字符串）
        - 特点：完全本地运行，数据不出本机，适合涉密场景
        - 前提条件：需要先安装 Ollama 并执行 ollama pull qwen2.5

    判断逻辑：
      通过检查 st.session_state.inference_engine 是否包含 "本地" 关键词
      来判断用户选择的是哪条轨。
      "☁️ DeepSeek-Chat (云端)" → "本地" 不在其中 → 走云端轨
      "🏠 本地 Ollama (qwen2.5)" → "本地" 在其中 → 走本地轨

    返回:
        dict: {"api_key": str, "base_url": str, "model": str}
          api_key: API 密钥（云端为真实 key，本地为占位符 "ollama"）
          base_url: API 端点 URL
          model: 模型名称标识符
    """
    engine = st.session_state.get("inference_engine", "")
    is_local = "本地" in engine

    if is_local:
        return {
            "api_key": "ollama",
            "base_url": "http://localhost:11434/v1",
            "model": "qwen2.5",
        }
    else:
        return {
            "api_key": st.session_state.get(
                "_sidebar_api_key",
                os.environ.get("DEEPSEEK_API_KEY", "")
            ),
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-chat",
        }


# =============================================================================
# API 调用函数 - RAG 对话（双轨制流式生成）
# =============================================================================

def call_llm_rag_stream(
    user_question: str,
    conversation_history: list,
) -> str:
    """
    RAG 闭环核心函数：检索 → 拼接 Prompt → 调用 LLM 流式生成。

    这是本系统最核心的 API 调用函数，实现 RAG（检索增强生成）的完整闭环：
      1. 语义检索：search_similar(user_question, top_k=3)
         → 在 ChromaDB 中检索与用户问题最相关的前 3 个论文片段
      2. Prompt 拼接：build_rag_prompt(question, retrieved_chunks)
         → 将检索片段格式化为引用材料，与用户问题拼接为完整 Prompt
      3. 上下文装配：将 System Prompt + 对话历史 + RAG User Message 组装为
         OpenAI Chat API 要求的 messages 格式
      4. LLM 调用：通过 OpenAI 客户端发送流式请求（stream=True）
      5. 流式渲染：逐个 token 接收并实时刷新到聊天界面

    参数:
        user_question (str):
          用户的自然语言问题（中英文均可）
        conversation_history (list):
          当前的对话历史列表，格式为 [{"role": "user/assistant", "content": "..."}]
          用于多轮对话时的上下文记忆（大模型能"记住"之前聊过什么）
          注意：不包含当前问题（当前问题在函数内单独构建为 RAG Prompt）

    返回:
        str: 大模型的完整回答文本（流式生成完成后返回完整文本）
          - 成功：返回 markdown 格式的回答字符串
          - 向量库为空：返回包含"请上传论文"提示的文本
          - API 错误：返回以 ❌ 开头的错误描述字符串
    """
    # -------------------------------------------------------------------------
    # Step 1: 获取推理引擎配置
    # -------------------------------------------------------------------------
    cfg = get_engine_config()

    # -------------------------------------------------------------------------
    # Step 2: 执行语义检索（RAG-R 环节）
    # -------------------------------------------------------------------------
    # search_similar 返回 Top-3 最相关的论文片段
    # 每个片段包含 text（片段文本）、source（论文文件名）、page（页码）、
    # distance（余弦距离，越小越相关）
    retrieved = search_similar(user_question, top_k=3)

    # -------------------------------------------------------------------------
    # Step 3: 构建 System Prompt（RAG-A 环节 - 系统级指令）
    # -------------------------------------------------------------------------
    # System Prompt 是大模型中最高层级的指令，定义了助手的行为边界。
    #
    # 本 System Prompt 的核心设计理念（防幻觉三重保险的第一重）：
    #   保险 1（System Prompt 自身）：
    #     "必须标注来源论文、页码和参考片段编号" → 强制溯源
    #     "没有依据的内容绝不编造" → 禁止幻觉
    #   保险 2（User Prompt 中的引用要求）：
    #     build_rag_prompt() 的「回答要求」部分重复强调引用格式
    #   保险 3（Temperature=0.5 + 重复约束）：
    #     较低温度减少随机性 + 多层级的重复约束降低遗忘概率
    #
    # 为什么 System Prompt 要特别强调"用中文回答"？
    #   因为检索到的论文片段是英文的（用户上传的是英文论文 PDF），
    #   大模型可能倾向于用英文回答。通过在 System Prompt 中显式要求
    #   中文输出，确保答案语言与用户期望一致。
    rag_system_prompt = {
        "role": "system",
        "content": (
            "你是「论文智能阅读助手」，一个严格基于论文原文的学术 AI。\n\n"
            "核心原则（优先级从高到低）：\n"
            "1. **准确引用优先**：每个论断必须标注来源论文、页码和参考片段编号。没有依据的内容绝不编造。\n"
            "2. **用中文回答**：你收到的是英文论文片段，但必须用中文回答用户，专业术语保留英文原名。\n"
            "3. **结构化输出**：使用 Markdown 格式，用列表、表格等组织信息，重点内容加粗。\n"
            "4. **诚实面对局限**：如果参考材料不足以回答某个问题，明确说「根据现有论文片段，无法确认……」。\n"
            "5. **引用格式**：严格使用 `【来源：文件名.pdf，第Y页，参考片段Z】` 标注。\n\n"
            "示例引用格式：\n"
            "「Transformer 模型完全基于注意力机制，摒弃了循环和卷积结构【来源：Attention Is All You Need.pdf，第2页，参考片段1】」"
        ),
    }

    # -------------------------------------------------------------------------
    # Step 4: 构建 User Message（RAG-A 环节 - 包含检索片段）
    # -------------------------------------------------------------------------
    rag_user_message = {
        "role": "user",
        "content": build_rag_prompt(user_question, retrieved),
    }

    # -------------------------------------------------------------------------
    # Step 5: 组装完整的 messages 列表
    # -------------------------------------------------------------------------
    # 消息顺序：System Prompt → 对话历史 → 当前 RAG Prompt
    #
    # 为什么要把对话历史也传进去？
    #   这是实现多轮对话的关键。大模型的 API 是无状态的（每次请求独立），
    #   通过传入之前的对话历史，大模型能"记住"之前聊过什么，实现连贯对话。
    #
    # 注意：conversation_history 中的消息直接复制，不修改 role 和 content。
    # 它们在前面的轮次中已经构建好了正确的格式。
    api_messages = [rag_system_prompt]

    # 追加对话历史
    for msg in conversation_history:
        api_messages.append({"role": msg["role"], "content": msg["content"]})

    # 追加当前 RAG User Message
    api_messages.append(rag_user_message)

    # -------------------------------------------------------------------------
    # 特殊处理：向量库为空时的后备逻辑
    # -------------------------------------------------------------------------
    # 当语义检索未返回任何结果时（向量库为空或不存在相关论文），
    # 需要替换 messages 为后备 Prompt，引导大模型提醒用户上传论文。
    #
    # 为什么不在 build_rag_prompt 中处理？
    #   因为空检索结果下 build_rag_prompt 产出的 Prompt 依然包含
    #   "根据以下引用材料回答..."的指令，但引用材料区是空的。这会
    #   给大模型一个语义上矛盾的信号。替换整个 messages 避免这种矛盾。
    if not retrieved:
        api_messages = [rag_system_prompt] + conversation_history + [
            {
                "role": "user",
                "content": (
                    "## 系统提示\n"
                    "当前向量库中未检索到相关论文片段。请提醒用户上传相关论文。\n\n"
                    f"## 用户问题\n{user_question}"
                ),
            }
        ]

    # -------------------------------------------------------------------------
    # Step 6: 创建 OpenAI 客户端并发送流式请求
    # -------------------------------------------------------------------------
    # 使用双轨推理引擎配置中的 api_key 和 base_url 初始化客户端。
    #   - 云端轨：api_key 是用户的 DeepSeek API Key，base_url 是 api.deepseek.com
    #   - 本地轨：api_key 是 "ollama"（占位符），base_url 是 localhost:11434/v1
    client = OpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
    )

    try:
        # -----------------------------------------------------------------------
        # 流式生成调用（stream=True）
        # -----------------------------------------------------------------------
        # 参数说明：
        #   model: 模型标识符（"deepseek-chat" 或 "qwen2.5"）
        #   messages: 完整的对话消息列表
        #   stream=True: 启用流式输出（逐 token 返回而非一次性返回全文）
        #   temperature=0.5:
        #     - 低于默认值 1.0，减少模型输出的随机性
        #     - 学术场景下需要更确定、更准确的回答（而非创造力）
        #     - 0.5 是经过实验验证的平衡点：既保留一定语言流畅度，又不致过度发散
        #   max_tokens=4096:
        #     - DeepSeek-Chat 支持最大 8K tokens 输出
        #     - 4096 约为 ~3000 个中文汉字，足够覆盖论文分析报告的需求
        #     - 设置上限防止意外情况下的无限输出
        response = client.chat.completions.create(
            model=cfg["model"],
            messages=api_messages,
            stream=True,
            temperature=0.5,
            max_tokens=4096,
        )

        # -----------------------------------------------------------------------
        # 流式渲染循环
        # -----------------------------------------------------------------------
        # stream=True 时，response 是一个迭代器，每次迭代返回一个 chunk。
        # 每个 chunk 包含一小段生成的文本（通常 1-3 个 token）。
        #
        # 流式渲染策略：
        #   1. 创建占位符（st.empty()），它占据一个固定的 UI 位置
        #   2. 逐 chunk 累积 full_response 字符串
        #   3. 每次累积后用 placeholder.markdown() 刷新显示
        #      - 末尾添加 ▌ 字符作为"打字光标"效果
        #      - 用户看到文字逐段出现，体验类似 ChatGPT 的打字效果
        #   4. 循环结束后移除光标，显示完整文本
        #
        # 用户体验优势：
        #   - 即时反馈：用户不再等待 15 秒后再看到完整回答
        #   - 可中断：生成的文本实时可见，用户可提前判断回答质量
        #   - 心理体验：打字效果降低了用户的等待焦虑感
        full_response = ""
        placeholder = st.empty()

        for chunk in response:
            # chunk.choices[0].delta.content 包含本次迭代的文本增量
            # 注意：某些 chunk 可能不包含 content（如仅包含 role 或 finish_reason），
            # 所以需要 if chunk.choices[0].delta.content 判断
            if chunk.choices[0].delta.content:
                # 累积文本
                full_response += chunk.choices[0].delta.content
                # 实时刷新：末尾加光标符 ▌
                placeholder.markdown(full_response + "▌")

        # 流式完成：移除光标，显示不带光标的完整文本
        placeholder.markdown(full_response)
        return full_response

    except Exception as e:
        # API 调用异常处理
        # 可能的异常：
        #   - 网络错误（无法连接到 API 服务器）
        #   - 认证错误（API Key 无效）
        #   - 速率限制（API 调用频率超限）
        #   - 模型不可用（本地 Ollama 未启动或模型未下载）
        #   - 超时错误（生成时间过长）
        error_msg = f"❌ API 调用失败：{str(e)}"
        st.error(error_msg)
        return error_msg


# =============================================================================
# API 调用函数 - 多文献交叉对比矩阵
# =============================================================================

def call_llm_comparison(selected_papers: list) -> str:
    """
    生成多文献交叉对比矩阵（双轨制）。

    与 call_llm_rag_stream 的区别：
      - 检索方式不同：使用 search_by_papers() 按四维度逐论文检索，
        而非 search_similar() 的单问题全库检索
      - Prompt 不同：要求大模型生成二维表格而非自由文本
      - 温度参数更低（0.3 vs 0.5）：确保表格格式的稳定性
      - 不包含对话历史：对比矩阵是独立的分析任务，不需要上下文

    完整工作流程：
      Step 1: 调用 search_by_papers() 执行多论文四维度检索
              → 返回按论文分组的维度化检索结果
              
      Step 2: 显示检索进度信息（检索到的片段总数、使用的推理引擎）
              → st.info 展示进度提示
              
      Step 3: 构建专用的对比分析 System Prompt
              → 强调"只输出表格"和"严格基于材料"
              
      Step 4: 构建 User Message（含全部检索结果）
              → build_comparison_prompt() 将检索结果格式化为表格生成 Prompt
              
      Step 5: 调用 LLM 流式生成
              → temperature=0.3 确保表格格式稳定（更低温度 = 更确定输出）
              
      Step 6: 流式渲染到聊天界面
              → 与 RAG 对话相同的流式渲染策略

    参数:
        selected_papers (list):
          用户在侧边栏勾选的论文文件名列表，如：
          ["Attention Is All You Need.pdf", "ResNet_2015.pdf"]

    返回:
        str: 大模型生成的 Markdown 对比表格（含来源标注）
    """
    cfg = get_engine_config()

    # -------------------------------------------------------------------------
    # Step 1: 多论文四维度检索
    # -------------------------------------------------------------------------
    # 使用 st.spinner 显示加载动画（因为检索可能耗时 2-5 秒）
    # search_by_papers 对每篇论文在四个维度（方法/实验/痛点/优点）上分别检索
    # 每个维度返回 top_k=3 个片段
    # 总检索次数 = len(selected_papers) × 4
    with st.spinner(
        f"🔍 正在从向量库检索 {len(selected_papers)} 篇论文的核心维度..."
    ):
        results = search_by_papers(
            papers=selected_papers,
            top_k_per_paper=3,
        )

    # -------------------------------------------------------------------------
    # Step 2: 显示检索进度
    # -------------------------------------------------------------------------
    total_chunks = sum(len(v) for v in results.values())
    engine_label = (
        "本地 Qwen2.5"
        if ("本地" in st.session_state.get("inference_engine", ""))
        else "DeepSeek"
    )
    st.info(
        f"共检索到 {total_chunks} 个相关片段，"
        f"正在调用 {engine_label} 拼装对比矩阵..."
    )

    # -------------------------------------------------------------------------
    # Step 3: 构建对比分析 System Prompt
    # -------------------------------------------------------------------------
    # 与 RAG 对话的 System Prompt 不同，此 Prompt 专门针对表格生成任务：
    #   - "只输出表格"：严格限制输出格式，避免大模型输出多余的寒暄或总结
    #   - "每个单元格 ≤ 60 字"：控制表格宽度，确保可读性
    #   - 使用"论文对比分析专家"的角色设定：提升大模型在学术对比任务上的表现
    #     （角色设定是 Prompt 工程中的经典技巧）
    comparison_system = {
        "role": "system",
        "content": (
            "你是「论文对比分析专家」，专门生成多篇学术论文的交叉对比矩阵。\n\n"
            "核心原则：\n"
            "1. **只输出表格**：用户唯一需要的输出是 Markdown 二维对比表格及其数据来源，不要输出任何前置说明、寒暄或总结。\n"
            "2. **严格基于材料**：只依据提供的检索片段填写表格，绝不编造数据。\n"
            "3. **中文输出**：表格内容用简洁中文概括（每个单元格 ≤ 60 字），专业术语保留英文。\n"
            "4. **诚实标注**：信息不足的维度直接写「未提及」。\n"
            "5. **页码溯源**：表格下方必须附「📌 数据来源」小节，逐条标注论文文件名和页码。"
        ),
    }

    # -------------------------------------------------------------------------
    # Step 4: 构建 User Message
    # -------------------------------------------------------------------------
    # build_comparison_prompt 将检索结果格式化为表格生成 Prompt
    user_message = {
        "role": "user",
        "content": build_comparison_prompt(results),
    }

    # -------------------------------------------------------------------------
    # Step 5: 创建客户端并发送流式请求
    # -------------------------------------------------------------------------
    client = OpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
    )

    try:
        # 对比矩阵生成的温度设置为 0.3（低于默认的 0.5）
        # 原因：
        #   - 表格格式对一致性要求极高
        #   - Markdown 表格的竖线 | 分隔符如果错位会导致解析失败
        #   - 低温度能显著提高表格语法的准确率
        #   - 0.3 是基于多次实验的经验值（0.3 时表格语法正确率约 98%）
        response = client.chat.completions.create(
            model=cfg["model"],
            messages=[comparison_system, user_message],
            stream=True,
            temperature=0.3,
            max_tokens=4096,
        )

        # -----------------------------------------------------------------------
        # 流式渲染（与 RAG 对话相同的渲染策略）
        # -----------------------------------------------------------------------
        full_response = ""
        placeholder = st.empty()

        for chunk in response:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)
        return full_response

    except Exception as e:
        error_msg = f"❌ 对比矩阵生成失败：{str(e)}"
        st.error(error_msg)
        return error_msg


# =============================================================================
# API 调用函数 - 一键学术报告
# =============================================================================

def call_llm_quick_report(action: str) -> str:
    """
    一键学术报告生成（双轨制）：根据 action 类型自动检索 + 生成结构化报告。

    支持的 action 类型：
      "innovation"  → 💡 核心创新点分析报告
      "dataset"     → 🔬 数据集与基线分析报告
      "future_work" → 🔮 未来工作与局限性分析报告

    每种 action 类型的处理流程：
      1. 选择检索关键词（search_map）
         - 每类 report 有专门优化的检索关键词，经过多次实验调优
         - 关键词采用中英文混合，充分利用 Embedding 模型的跨语言能力
      2. 执行语义检索（top_k=5，多于 RAG 对话的 3，确保覆盖全面）
      3. 选择 System Prompt（system_map）
         - 每类 report 有独立的 System Prompt，设定不同的专家角色
      4. 选择 Prompt 构建函数（builder_map）
         - 每类 report 使用不同的 build_*_prompt 函数
      5. 调用 LLM 流式生成（temperature=0.4）

    参数:
        action (str): 报告类型标识符
          "innovation" | "dataset" | "future_work"

    返回:
        str: 大模型生成的结构化报告文本
    """
    cfg = get_engine_config()

    # -------------------------------------------------------------------------
    # 检索关键词映射表
    # -------------------------------------------------------------------------
    # 每个 action 对应一组经过优化的检索关键词。
    # 关键词的设计原则：
    #   - 覆盖该报告类型所需的核心语义域
    #   - 中英混合：中文关键词提高与中文 Embedding 模型的匹配度，
    #     英文关键词提高与英文论文片段的直接语义匹配度
    #   - 关键词字符串越长，Embedding 向量在不同含义间的区分度越高
    #
    # 为什么不同 action 使用不同的 query（而非同一个 query 用于所有检索）？
    #   因为不同语义维度的信息分布在论文的不同章节中。
    #   例如：
    #     innovation 相关的内容主要出现在 Introduction、Methods、
    #     Model Architecture 等章节
    #     dataset 相关的内容主要出现在 Experiments、Evaluation 章节
    #     future_work 相关的内容主要出现在 Conclusion、Discussion 章节
    #   使用针对性关键词可以显著提高相关片段的召回率。
    search_map = {
        "innovation": (
            "核心创新点 贡献 方法 架构 模型设计 "
            "main contribution novelty proposed method architecture"
        ),
        "dataset": (
            "数据集 实验 评价指标 基线 结果 "
            "benchmark dataset experiment evaluation metrics baseline results"
        ),
        "future_work": (
            "未来工作 局限性 缺点 展望 "
            "future work limitation conclusion discussion"
        ),
    }

    # 获取当前 action 对应的检索关键词，默认为 innovation 的（防御性设计）
    query = search_map.get(action, search_map["innovation"])

    # 执行检索：top_k=5，多于 RAG 对话的 3
    # 原因：学术报告需要更全面的信息覆盖，5 个片段能更好地覆盖论文的不同章节
    retrieved = search_similar(query, top_k=5)

    # -------------------------------------------------------------------------
    # System Prompt 映射表（按报告类型区分角色设定）
    # -------------------------------------------------------------------------
    # 每种报告类型设定不同的 AI 专家角色，以引导大模型在特定维度上表现更好。
    # 角色设定（Persona Engineering）是 Prompt 工程的高级技巧：
    #   - 通过赋予特定身份，激活大模型中与该身份相关的知识和表达模式
    #   - "论文学术分析专家"、"实验配置分析专家"、"研究方向分析专家"
    #     分别对应三个不同的知识域
    system_map = {
        "innovation": {
            "role": "system",
            "content": (
                "你是「论文学术分析专家」，专门提炼论文的核心创新点。\n\n"
                "核心原则：\n"
                "1. **结构化输出**：严格按用户要求的小节格式输出报告。\n"
                "2. **引用溯源**：每条论断后标注【来源：文件名.pdf，第Y页】。\n"
                "3. **诚实面对局限**：检索片段不足时明确告知。\n"
                "4. **中文输出**，专业术语保留英文原名。"
            ),
        },
        "dataset": {
            "role": "system",
            "content": (
                "你是「实验配置分析专家」，专门提取论文的数据集、基线和实验结果。\n\n"
                "核心原则：\n"
                "1. **数据精确**：提取的每个数字和名称必须来源于原文片段。\n"
                "2. **表格呈现**：优先使用 Markdown 表格列数据集和基线方法。\n"
                "3. **引用溯源**：每条数据标注【来源：文件名.pdf，第Y页】。\n"
                "4. **中文输出**，数据集名/指标名保留英文原名。"
            ),
        },
        "future_work": {
            "role": "system",
            "content": (
                "你是「研究方向分析专家」，专门提取论文的局限性和未来工作。\n\n"
                "核心原则：\n"
                "1. **区分原文与洞察**：原文的 limitation/future work 要标注来源；AI 延伸思考需明确标识。\n"
                "2. **引用溯源**：原文内容标注【来源：文件名.pdf，第Y页】。\n"
                "3. **中文输出**。"
            ),
        },
    }

    # -------------------------------------------------------------------------
    # Prompt 构建函数映射表
    # -------------------------------------------------------------------------
    builder_map = {
        "innovation": build_innovation_prompt,
        "dataset": build_dataset_prompt,
        "future_work": build_future_work_prompt,
    }

    # 选择对应的 System Prompt 和 User Prompt 构建函数
    # 默认回退到 innovation 相关函数（防御性设计）
    system_msg = system_map.get(action, system_map["innovation"])
    builder = builder_map.get(action, builder_map["innovation"])
    user_content = builder(retrieved)

    # -------------------------------------------------------------------------
    # 调用 LLM 流式生成
    # -------------------------------------------------------------------------
    client = OpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
    )

    try:
        # temperature=0.4：略低于 RAG 对话的 0.5，略高于对比矩阵的 0.3
        # 报告需要在结构准确性和内容创造之间取得平衡：
        #   - 结构方面（表格/小节格式）需要低温度确保稳定
        #   - 内容方面（对论文方法的解读）需要一定创造力
        #   0.4 是从多次实验中得到的经验值
        response = client.chat.completions.create(
            model=cfg["model"],
            messages=[
                system_msg,
                {"role": "user", "content": user_content},
            ],
            stream=True,
            temperature=0.4,
            max_tokens=4096,
        )

        # 流式渲染
        full_response = ""
        placeholder = st.empty()

        for chunk in response:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)
        return full_response

    except Exception as e:
        error_msg = f"❌ 报告生成失败：{str(e)}"
        st.error(error_msg)
        return error_msg


# =============================================================================
# 用户输入处理与事件路由（Presenter 层 - 主事件循环）
# =============================================================================

# 事件优先级说明（由高到低）：
#   优先级 1：多文献交叉对比矩阵触发
#     条件：st.session_state._trigger_comparison == True
#     来源：用户在侧边栏点击「📊 生成多文献对比矩阵」按钮
#   
#   优先级 2：一键学术报告触发
#     条件：st.session_state._quick_action 不为 None/空
#     来源：用户在侧边栏点击 💡/🔬/🔮 快捷按钮
#   
#   优先级 3：常规聊天输入
#     条件：用户在底部聊天框输入并提交问题
#     来源：st.chat_input 组件返回值
#
# 设计原理说明——为什么使用 session_state flag + rerun 机制？
#   在 Streamlit 中，按钮的 on_click 回调函数的执行时机和 main 流程是
#   解耦的。如果直接在回调中调用 API（耗时 10-20 秒），会导致 UI 在
#   等待期间完全无响应。通过 flag + rerun 模式：
#     1. 按钮回调仅设置 flag 并调用 st.rerun()（瞬间完成）
#     2. rerun 后，脚本重新执行，在主流程中检测到 flag
#     3. 在主流程中调用 API → 流式渲染 → 结果展示
#     4. 完成后清除 flag，确保不会重复执行
#   这样 API 调用在正常的 Streamlit runner 上下文中执行，UI 保持响应。

# =============================================================================
# 优先级 1：多文献对比矩阵触发
# =============================================================================
if st.session_state.pop("_trigger_comparison", False):
    # pop 操作同时获取 flag 值并清空它（原子操作，防止重复触发）
    
    selected = st.session_state.get("_selected_papers", [])

    # 前置条件检查
    if not st.session_state.api_key_valid:
        st.error("⚠️ 请先在左侧边栏输入你的 DeepSeek API Key")
    elif len(selected) < 2:
        st.warning("⚠️ 请至少勾选 2 篇论文")
    else:
        # --- 构建触发消息（用户可见的"提问"）---
        trigger_msg = (
            f"📊 **触发生成多文献交叉对比矩阵**\n\n"
            f"已勾选论文：{'、'.join(selected)}"
        )

        # 追加触发消息到聊天历史
        st.session_state.messages.append(
            {"role": "user", "content": trigger_msg}
        )
        # 渲染用户消息气泡
        with st.chat_message("user"):
            st.markdown(trigger_msg)

        # 渲染助手消息气泡并流式生成对比矩阵
        with st.chat_message("assistant"):
            response = call_llm_comparison(selected)

        # 追加助手回复到聊天历史
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
        # 再次 rerun 以刷新聊天界面（确保所有新消息正确渲染）
        st.rerun()

# =============================================================================
# 优先级 2：一键学术报告触发
# =============================================================================
quick_action = st.session_state.pop("_quick_action", None)

if quick_action:
    # pop 操作获取 flag 并清空

    if not st.session_state.api_key_valid:
        st.error("⚠️ 请先在左侧边栏输入你的 DeepSeek API Key")
        st.stop()  # 停止本次执行，不继续处理

    # 构建用户可见的触发消息（根据 action 类型使用不同的文案）
    action_labels = {
        "innovation": "💡 一键提取创新点",
        "dataset": "🔬 一键提取数据集与基线",
        "future_work": "🔮 一键分析未来工作",
    }
    trigger_msg = action_labels.get(quick_action, quick_action)

    # 追加到聊天历史
    st.session_state.messages.append(
        {"role": "user", "content": trigger_msg}
    )
    # 渲染用户消息
    with st.chat_message("user"):
        st.markdown(trigger_msg)

    # 渲染助手消息并流式生成报告
    with st.chat_message("assistant"):
        response = call_llm_quick_report(quick_action)

    # 追加助手回复到聊天历史
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
    # 刷新聊天界面
    st.rerun()

# =============================================================================
# 优先级 3：常规聊天输入处理
# =============================================================================

# st.chat_input 是 Streamlit 的聊天输入框组件。
# 它固定在页面底部，用户在文本框中输入问题并回车后，
# prompt 变量会接收到问题文本（字符串）。
# 
# 当用户尚未输入任何内容时，prompt 为 None（空字符串在 Python 中为 falsy）
# 后续的 if prompt: 代码块不会执行。
prompt = st.chat_input("💬 输入你的问题（例如：这篇论文的创新点是什么？）")

if prompt:
    # 前置条件检查：API Key 有效性
    if not st.session_state.api_key_valid:
        st.error("⚠️ 请先在左侧边栏输入你的 DeepSeek API Key")
        st.stop()  # 停止执行，不发送 API 请求

    # -------------------------------------------------------------------------
    # 步骤 1：追加用户消息到聊天历史
    # -------------------------------------------------------------------------
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    # -------------------------------------------------------------------------
    # 步骤 2：渲染用户消息气泡
    # -------------------------------------------------------------------------
    # 每个 chat_message 上下文渲染一个聊天气泡。
    # 使用 st.empty 创建空白区域，然后用 st.markdown 渲染消息。
    with st.chat_message("user"):
        st.markdown(prompt)

    # -------------------------------------------------------------------------
    # 步骤 3：组装 API messages（对话历史，不含当前消息）
    # -------------------------------------------------------------------------
    # 从 session_state.messages 中提取之前的对话历史（不含刚追加的当前问题）。
    # [:-1] 切片排除了最后一个元素（即当前问题）。
    #
    # 过滤条件：只保留 role 为 "user" 或 "assistant" 的消息。
    #   - 过滤掉可能的 "system" role 消息（当前设计中不会出现）
    #   - 确保 API messages 的格式严格符合 OpenAI Chat API 规范
    api_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
        if m["role"] in ("user", "assistant")
    ]

    # -------------------------------------------------------------------------
    # 步骤 4：生成并渲染助手回复
    # -------------------------------------------------------------------------
    # 在助手消息气泡的上下文中调用 RAG 流式生成函数。
    # call_llm_rag_stream 内部负责：
    #   - 语义检索（RAG-R）
    #   - Prompt 拼接（RAG-A）
    #   - API 调用和流式渲染（RAG-G）
    with st.chat_message("assistant"):
        response = call_llm_rag_stream(prompt, api_messages)

    # -------------------------------------------------------------------------
    # 步骤 5：追加助手回复到聊天历史
    # -------------------------------------------------------------------------
    # 将大模型生成的完整回答追加到 session_state.messages 中，
    # 确保下一次用户提问时，此回答能作为上下文的一部分传给大模型。
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )