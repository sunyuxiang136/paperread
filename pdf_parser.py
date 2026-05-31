"""
=============================================================================
 PDF 解析与语义切块模块 (pdf_parser.py)
 所属系统：基于 RAG 架构与多文献聚合的学术论文智能解析系统
=============================================================================

功能概述：
  本模块负责将用户上传的 PDF 学术论文转换为携带页码元数据的语义文本块
  (Chunk)，为下游的 Embedding 向量化和 RAG 检索提供结构化输入数据。

核心技术栈：
  - PDF 解析引擎：PyMuPDF (fitz)，以字节流模式解析，无需落盘
  - 文本清洗：正则表达式过滤页码行、合并冗余换行
  - 语义切分：基于中英文句末标点的句子边界检测 + 贪心缓冲填充算法
  - 元数据注入：每个 Chunk 硬编码携带 {source: 文件名, page: 实际页码}

设计原则：
  1. 全内存操作：PDF 以 bytes 形式接收，解析全程不产生临时文件
  2. 页码不可篡改：元数据在切分的最早阶段注入，贯穿整个处理链路
  3. 语义完整性优先：切分优先在句子边界执行，避免在单词或公式中间截断
  4. 双栏论文适应：PyMuPDF 的 get_text("text") 自动按逻辑阅读顺序提取

算法复杂度：
  - 时间复杂度：O(N)，N 为 PDF 总字符数（单遍扫描）
  - 空间复杂度：O(N)，需在内存中持有全部句子和 Chunk 列表

版本历史：
  - v1.0 (2026-05)：初始版本，支持英文单栏/双栏 PDF
  - v1.1 (2026-05)：增加中文标点支持、超长句多级拆分、短块归并后处理
=============================================================================
"""

import fitz  # PyMuPDF: 高性能 PDF 解析库，支持字节流模式直接解析
import re
from typing import List, Dict, Tuple


# =============================================================================
# 全局常量定义
# =============================================================================

# 页码行过滤正则：匹配仅由 1-4 位数字组成的行（PDF 中常见的孤立页码行）
PAGE_NUMBER_PATTERN = re.compile(r'^\d{1,4}$')

# 多换行合并正则：将 3 个及以上连续换行压缩为双换行（段落分隔）
EXCESSIVE_NEWLINE_PATTERN = re.compile(r'\n{3,}')

# 句子边界检测正则（中英文混合）：
#   中文：。！？作为句末标点
#   英文：.!? 后跟空白 + 大写字母或中文（识别下一句起始）
#   关键设计：(?<=...) 为后顾断言（不消耗字符），(?=...) 为前瞻断言
#   避免在缩写（如 "e.g. "）处误切，因为缩写后的空格后通常是小写字母
SENTENCE_BOUNDARY_PATTERN = re.compile(
    r'(?<=[.!?。！？])\s+(?=[A-Z\u4e00-\u9fff])|(?<=[.!?。！？\n])(?=\s*$)'
)

# 子句边界检测正则（中英文逗号、分号），用于超长句的二级拆分
CLAUSE_BOUNDARY_PATTERN = re.compile(r'(?<=[,;，；])\s+')

# 默认 Chunk 大小参数（字符数）
DEFAULT_MIN_CHARS = 400  # Chunk 最小字符数，不足则向前合并
DEFAULT_MAX_CHARS = 600  # Chunk 最大字符数，超出则触发切分

# 硬切断点字符：在超长文本中按这些字符就近切断
HARD_SPLIT_CHARS = (' ', ',', '.', ';', '!', '?', '，', '。', '；', '、')

# 超长句硬切时的搜索窗口大小（字符）：避免在单词中间截断
HARD_SPLIT_WINDOW = 50

# =============================================================================
# 内部辅助函数
# =============================================================================


def _clean_text(text: str) -> str:
    """
    对 PDF 提取的原始文本执行清洗，去除排版噪音。

    清洗步骤：
      1. 逐行处理：去除首尾空白，过滤孤立页码行
      2. 合并空白行：将 3 个及以上连续换行压缩为双换行（保留段落边界）
      3. 不修改字符本身，保持原文完整性

    参数:
        text (str): PyMuPDF 提取的原始文本

    返回:
        str: 清洗后的文本，行内空白已规范化但字符内容未变

    设计说明：
        仅进行结构性清洗（去除页码行、合并空行），不做语义级修改。 
        论文中的公式、引用标记、特殊字符均原样保留，交由 Embedding 
        模型自行处理其语义表征。
    """
    # 第一步：逐行处理
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        # 过滤纯页码行：如 "1"、"42"、"100" 等 PDF 页脚/页眉数字
        # 不删除包含数字的正文行（如 "the top-1 accuracy is..."）
        if line and not PAGE_NUMBER_PATTERN.match(line):
            cleaned_lines.append(line)

    # 重新拼接为单个字符串
    text = '\n'.join(cleaned_lines)

    # 第二步：合并连续空行
    # 将 \n\n\n 及以上压缩为 \n\n，保留段落级别的视觉分隔
    # 单换行可能只是排版换行，双换行则是真正的段落边界
    text = EXCESSIVE_NEWLINE_PATTERN.sub('\n\n', text)

    return text


def _split_into_sentences(text: str) -> List[str]:
    """
    按句子边界拆分文本为独立的句子列表。

    切分策略（优先级从高到低）：
      1. 句末标点切分：识别英文(.!?)和中文(。！？)句末标点
         - 条件：句末标点后紧跟空白 + 下一个非空白字符是英文大写或中文
         - 设计：使用正则后顾/前瞻断言，不消耗标点字符
      2. 换行切分：文本内部的换行作为辅助切分边界
         - 论文中公式或图表后的换行往往是自然的语义边界
      3. 空句过滤：去除切分后的空字符串和仅含空白的句子

    参数:
        text (str): 单页的清洗后文本

    返回:
        List[str]: 句子列表，每个元素为一个完整的中英文句子

    注意事项：
        - 英文缩写（如 "Fig. 1", "e.g. ", "i.e. ", "et al. "）不会被误切，
          因为这些缩写后的空格后通常跟小写字母或数字，不匹配前瞻断言
        - 引文中的句号（如 "Smith et al. (2017)"）同理不会被误判为句末
    """
    # 第一步：使用正则匹配所有句子边界，执行切分
    # split 方法在匹配位置切割，返回切割后的片段列表
    parts = SENTENCE_BOUNDARY_PATTERN.split(text)

    # 第二步：过滤和处理切分结果
    result = []
    for part in parts:
        part = part.strip()
        if not part:
            continue  # 跳过空字符串

        # 如果片段内仍包含换行，进一步拆分
        # 换行在论文中通常表示自然段落边界或公式后的换行
        if '\n' in part:
            sub_parts = part.split('\n')
            for sp in sub_parts:
                sp = sp.strip()
                if sp:
                    result.append(sp)
        else:
            result.append(part)

    return result


def _split_long_by_clause(
    text: str,
    source: str,
    page: int,
    max_chars: int
) -> List[Dict]:
    """
    对超长句子执行二级拆分：先按逗号/分号切分，兜底按字符硬切。

    这是语义切分流程中的异常处理分支，仅在单句超过 max_chars 时触发。
    算法采用与主切分流程一致的贪心缓冲策略：

    拆分层级：
      第一层（子句级）：按中英文逗号(,，)和分号(;；)拆分
      第二层（硬切级）：若单个子句仍超长，按字符数硬切

    参数:
        text      (str): 需要拆分的超长句原文
        source    (str): 来源文件名（用于元数据注入）
        page      (int): 页码（用于元数据注入）
        max_chars (int): Chunk 最大字符数阈值

    返回:
        List[Dict]: 拆分后的 Chunk 列表，每个元素为标准 Chunk 结构
    """
    # 第一层拆分：按子句边界切分
    clauses = CLAUSE_BOUNDARY_PATTERN.split(text)
    chunks = []

    # 贪心缓冲变量
    buffer_texts = []  # 当前 Chunk 的子句文本列表
    buffer_len = 0      # 当前 Chunk 的累计字符数

    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue

        clause_len = len(clause)

        # 情况 1：单个子句仍超长 → 先提交已有缓冲，再硬切
        if clause_len > max_chars:
            if buffer_texts:
                chunks.append(
                    _build_chunk(" ".join(buffer_texts), source, page)
                )
                buffer_texts = []
                buffer_len = 0

            # 硬切：按字符数截断，尽力在标点处切断
            sub_chunks = _hard_split(clause, source, page, max_chars)
            chunks.extend(sub_chunks)
            continue

        # 情况 2：正常子句 → 尝试加入缓冲
        new_len = buffer_len + 1 + clause_len if buffer_texts else clause_len

        if new_len <= max_chars:
            # 缓冲未满，追加
            buffer_texts.append(clause)
            buffer_len = new_len
        else:
            # 缓冲已满，提交当前 Chunk 并开启新缓冲
            chunks.append(
                _build_chunk(" ".join(buffer_texts), source, page)
            )
            buffer_texts = [clause]
            buffer_len = clause_len

    # 处理尾缓冲
    if buffer_texts:
        chunks.append(
            _build_chunk(" ".join(buffer_texts), source, page)
        )

    return chunks


def _hard_split(
    text: str,
    source: str,
    page: int,
    max_chars: int
) -> List[Dict]:
    """
    兜底硬切策略：按最大字符数截断，在断点处尽力避免单词截断。

    算法流程：
      1. 从当前位置起，以 max_chars 为步长确定初始截断点
      2. 向左搜索：在 (截断点-50, 截断点] 范围内寻找最近的标点或空格
      3. 若左搜索未找到断点，向右搜索：在 [截断点, 截断点+50) 范围内寻找
      4. 如仍未找到合适断点，直接按 max_chars 截断
      5. 丢弃空白片段，进入下一次迭代

    参数:
        text      (str): 需要硬切的超长文本
        source    (str): 来源文件名
        page      (int): 页码
        max_chars (int): 每次切分的最大字符数

    返回:
        List[Dict]: 硬切后的 Chunk 列表
    """
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        # 计算初始截断点（不超过文本末尾）
        end = min(start + max_chars, text_len)

        if end < text_len:
            best = end

            # 策略 A：向左搜索断点（优先使用）
            # 在 (start, end] 范围内，从 end 向左搜索最多 50 个字符
            search_left_start = max(start, end - HARD_SPLIT_WINDOW)
            for i in range(end, search_left_start, -1):
                if text[i - 1] in HARD_SPLIT_CHARS:
                    # 找到断点，截断位置在字符之后（含该标点）
                    best = i
                    break

            # 策略 B：向左未找到 → 向右搜索断点
            if best == end:
                search_right_end = min(text_len, end + HARD_SPLIT_WINDOW)
                for i in range(end, search_right_end):
                    if text[i] in HARD_SPLIT_CHARS:
                        # 找到断点，截断位置在字符之后
                        best = i + 1
                        break

            end = best

        # 提取当前片段，丢弃纯空白
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(_build_chunk(chunk_text, source, page))

        # 移动到下一段的起始位置
        start = end

    return chunks


def _build_chunk(text: str, source: str, page: int) -> Dict:
    """
    构建单个语义块(Chunk)的标准字典结构。

    此函数是元数据注入的核心执行点。每创建一个 Chunk，都必须在此
    函数中硬编码注入 source 和 page 字段。这一设计保证了：
      - 元数据不可丢失：任何 Chunk 没有不带页码的创建路径
      - 元数据不可篡改：下游代码从 Chunk 读取时无需信任，直接取用
      - 类型一致性：page 统一为 int 类型，避免后续比较时的类型错误

    参数:
        text   (str): Chunk 的文本内容（已清洗和切分）
        source (str): 来源 PDF 文件名（如 "Transformer_2017.pdf"）
        page   (int): 该 Chunk 所属的 PDF 页码（从 1 开始）

    返回:
        Dict: 包含 'text' 和嵌套 'metadata' 字典的标准 Chunk 结构
              格式：{"text": str, "metadata": {"source": str, "page": int}}
    """
    return {
        "text": text.strip(),
        "metadata": {
            "source": source,
            "page": page,
        },
    }


def _merge_short_chunks(
    chunks: List[Dict],
    min_chars: int,
    max_chars: int
) -> List[Dict]:
    """
    后处理：将字符数不足阈值的短 Chunk 向前合并。

    为什么需要此步骤：
      在贪心填充过程中，某些句子组合可能刚好超出 max_chars 被提交，
      导致下一个 Chunk 的起始句子过短，形成碎片化的短 Chunk。这会影响：
      - Embedding 质量：过短的文本缺乏充分的语义上下文
      - 检索精度：短 Chunk 与查询之间的语义匹配可能不充分
      - 存储效率：过多的小 Chunk 会增大向量索引的规模

    合并策略：
      从前向后遍历，尝试将每个短 Chunk 与前一个 Chunk 合并。
      仅当合并后总字符数 ≤ max_chars 时才执行合并，否则保留独立。

    参数:
        chunks    (List[Dict]): 贪心填充阶段产出的 Chunk 列表
        min_chars (int): 短 Chunk 判定阈值（默认 400）
        max_chars (int): 合并后最大允许字符数（默认 600）

    返回:
        List[Dict]: 合并处理后的 Chunk 列表
    """
    if not chunks:
        return chunks

    merged = []
    buffer = chunks[0]  # 以第一个 Chunk 作为初始缓冲

    for i in range(1, len(chunks)):
        current = chunks[i]
        combined_text = buffer["text"] + " " + current["text"]

        # 检查合并后是否超限
        if len(combined_text) <= max_chars:
            # 可以合并：更新缓冲区的文本内容
            buffer["text"] = combined_text
        else:
            # 不可合并：提交缓冲，当前 Chunk 成为新缓冲
            merged.append(buffer)
            buffer = current

    # 处理最后一个 Chunk（此时在 buffer 中）
    if merged and len(buffer["text"]) < min_chars:
        # 最后一块过短，尝试与前一块合并
        last = merged[-1]
        if len(last["text"]) + 1 + len(buffer["text"]) <= max_chars:
            last["text"] = last["text"] + " " + buffer["text"]
        else:
            merged.append(buffer)
    else:
        merged.append(buffer)

    return merged


# =============================================================================
# 公开接口
# =============================================================================


def extract_text_by_page(pdf_bytes: bytes, filename: str) -> List[Dict]:
    """
    从 PDF 字节流中逐页提取文本并携带元数据。

    技术实现细节：
      - 使用 fitz.open(stream=pdf_bytes, filetype="pdf") 直接解析内存中的
        字节流，避免将上传文件写入磁盘（安全性更高，无临时文件残留）
      - page.get_text("text") 返回按逻辑阅读顺序排列的纯文本
        * 对于双栏排版的学术论文，PyMuPDF 会自动按"左栏→右栏"顺序输出
        * 对于包含表格和公式的页面，会提取其中的文字部分
      - 即使某页无有效文本，也会保留该页的记录（text 为空字符串），
        确保页号索引的连续性

    参数:
        pdf_bytes (bytes): PDF 文件的完整字节内容（如 Streamlit uploader 的 read() 结果）
        filename  (str):   原始文件名，用于元数据标注（如 "Transformer_2017.pdf"）

    返回:
        List[Dict]: 每页的文本及元数据，格式为：
          [
            {"text": "...", "source": "Transformer_2017.pdf", "page": 1},
            {"text": "...", "source": "Transformer_2017.pdf", "page": 2},
            ...
          ]
        列表长度等于 PDF 的总页数
    """
    pages = []

    # 打开 PDF 字节流（不落盘，纯内存操作）
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_num in range(len(doc)):
        page = doc[page_num]

        # 提取该页的纯文本（非 OCR，直接提取内嵌文本层）
        raw_text = page.get_text("text")

        # 清洗：去除页码行、合并冗余换行
        cleaned = _clean_text(raw_text)

        if cleaned.strip():
            # 有效文本页：保存清洗后的文本
            pages.append({
                "text": cleaned,
                "source": filename,
                "page": page_num + 1,  # PDF 页码从 0 开始，实际页码 +1
            })
        else:
            # 空白页（图表页、引用页等）：保留占位记录以维持页码连续
            pages.append({
                "text": "",
                "source": filename,
                "page": page_num + 1,
            })

    doc.close()  # 释放 PDF 文件句柄
    return pages


def semantic_chunk(
    pages: List[Dict],
    min_chars: int = DEFAULT_MIN_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> List[Dict]:
    """
    语义切分算法：按句子边界将逐页文本切分为 400-600 字的语义块。

    算法流程图解：

      输入: 逐页文本 (每页含 text + source + page)
        │
        ▼
      [阶段 1] 句子分拆
        逐页文本 → 正则匹配句子边界 → 注入页码 → 平铺为 (句子, 页码) 队列
        │
        ▼
      [阶段 2] 贪心填充
        遍历句子队列 → 逐句加入缓冲 → 达到 max_chars 阈值 → 提交 Chunk
        │                    │
        │                    └── 单句超长 → 二级拆分（子句→硬切）
        │
        ▼
      [阶段 3] 短块归并
        遍历所有 Chunk → 短块(< min_chars) 向前合并 → 确保信息密度
        │
        ▼
      输出: Chunk 列表 (含 text + metadata)

    切分优先级（从高到低）：
      1. 句子边界切分：在句号/问号/感叹号（中英文）处切分
      2. 子句边界切分：在逗号/分号（中英文）处切分（仅超长句触发）
      3. 硬切：按字符数截断，在标点处就近选断点（兜底策略）

    参数:
        pages     (List[Dict]): extract_text_by_page 的输出
        min_chars (int):       每个 Chunk 的最小字符数（默认 400）
        max_chars (int):       每个 Chunk 的最大字符数（默认 600）

    返回:
        List[Dict]: 语义切片后的 Chunk 列表，每个元素为：
          {'text': str, 'metadata': {'source': str, 'page': int}}

    设计权衡：
      - 为什么是 400-600 字？ 
        400 字 ≈ 3-5 个学术句子，足以承载一个完整的论证单元
        600 字上限确保单个 Chunk 不超过 Embedding 模型的上下文窗口
        且信息密度适中，不会因过长而稀释关键语义
        
      - 为什么用贪心而非全局最优？
        贪心策略的时间复杂度为 O(N)，适合实时处理大型论文（30+ 页）
        全局最优（如 DP）在此场景下收益有限但开销剧增
    """
    # ===========================================================================
    # 阶段 1：将所有页面的文本按句子拆分，展开为带页码的平面队列
    # ===========================================================================
    all_sentences = []  # 类型: List[Tuple[str, str, int]]
    # 每个元素为 (sentence_text, source_filename, page_number)

    for page_data in pages:
        text = page_data["text"]
        if not text.strip():
            continue  # 跳过纯空白页

        source = page_data["source"]
        page = page_data["page"]

        # 按句子边界拆分（支持中英文句末标点）
        sentences = _split_into_sentences(text)

        for sent in sentences:
            sent = sent.strip()
            if sent:
                # 每个句子绑定其来源页码，实现页码→句子的精确追溯
                all_sentences.append((sent, source, page))

    # ===========================================================================
    # 阶段 2：贪心缓冲填充——将句子组装为 400-600 字的 Chunk
    # ===========================================================================
    chunks = []

    # 贪心缓冲状态变量
    buffer_texts = []    # 当前 Chunk 的句子文本列表（用于最终 join）
    buffer_source = None # 当前 Chunk 的来源文件名
    buffer_page = None   # 当前 Chunk 起始句子的页码
    buffer_len = 0       # 当前 Chunk 的累计字符数

    for sent_text, source, page in all_sentences:
        sent_len = len(sent_text)

        # -----------------------------------------------------------------------
        # 特殊情况：单句长度超过 max_chars 阈值
        # 处理流程：(1) 先提交当前缓冲中的内容为一个完整 Chunk
        #          (2) 对超长句执行二级拆分（子句级→硬切）
        #          (3) 跳过正常的贪婪追加流程
        # -----------------------------------------------------------------------
        if sent_len > max_chars:
            # 步骤 (1)：提交当前缓冲（如果非空）
            if buffer_texts:
                chunks.append(
                    _build_chunk(" ".join(buffer_texts), buffer_source, buffer_page)
                )
                # 重置缓冲状态
                buffer_texts = []
                buffer_len = 0

            # 步骤 (2)：对超长句执行多级拆分
            sub_chunks = _split_long_by_clause(sent_text, source, page, max_chars)
            chunks.extend(sub_chunks)
            continue  # 跳过正常的追加流程

        # -----------------------------------------------------------------------
        # 正常情况：尝试将当前句子追加到缓冲
        # -----------------------------------------------------------------------
        if buffer_texts:
            # +1 为句子间的分隔空格
            new_len = buffer_len + 1 + sent_len
        else:
            # 缓冲为空，当前句子即为 Chunk 的第一个句子
            new_len = sent_len

        if new_len <= max_chars:
            # 缓冲未满：追加句子并更新状态
            buffer_texts.append(sent_text)
            buffer_len = new_len
            if buffer_source is None:
                # 记录 Chunk 起始页码（首个句子的页码）
                buffer_source = source
                buffer_page = page
        else:
            # 缓冲已满：提交当前 Chunk，开启新 Chunk
            chunks.append(
                _build_chunk(" ".join(buffer_texts), buffer_source, buffer_page)
            )
            # 新 Chunk 以当前句子为起始
            buffer_texts = [sent_text]
            buffer_source = source
            buffer_page = page
            buffer_len = sent_len

    # -----------------------------------------------------------------------
    # 提交最后一个 Chunk（尾缓冲）
    # -----------------------------------------------------------------------
    if buffer_texts:
        chunks.append(
            _build_chunk(" ".join(buffer_texts), buffer_source, buffer_page)
        )

    # ===========================================================================
    # 阶段 3：短块归并——将不足 min_chars 的 Chunk 向前合并
    # ===========================================================================
    chunks = _merge_short_chunks(chunks, min_chars, max_chars)

    return chunks


def parse_pdf(
    pdf_bytes: bytes,
    filename: str,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS
) -> List[Dict]:
    """
    一站式 PDF 解析入口：文本提取 → 语义切分 → 元数据注入。

    此函数将 extract_text_by_page 和 semantic_chunk 串联为完整流水线，
    是外部调用者（app.py）唯一需要调用的 PDF 解析接口。

    参数:
        pdf_bytes (bytes): PDF 文件的字节内容
        filename  (str):   原始文件名（用于元数据标注）
        min_chars (int):   每块最小字符数（默认 400）
        max_chars (int):   每块最大字符数（默认 600）

    返回:
        List[Dict]: Chunk 列表，每个元素为：
          {
            "text": "Transformer 模型完全基于注意力机制...",
            "metadata": {
              "source": "Attention Is All You Need.pdf",
              "page": 4
            }
          }

    使用示例:
        >>> with open("paper.pdf", "rb") as f:
        ...     chunks = parse_pdf(f.read(), "paper.pdf")
        >>> print(f"切分为 {len(chunks)} 个语义块")
        >>> print(f"第 1 个 Chunk: 第 {chunks[0]['metadata']['page']} 页")
    """
    # 第一阶段：PDF 文本提取 + 清洗
    pages = extract_text_by_page(pdf_bytes, filename)

    # 第二阶段：语义切分
    chunks = semantic_chunk(pages, min_chars=min_chars, max_chars=max_chars)

    return chunks


# =============================================================================
# 调试辅助函数
# =============================================================================


def print_chunks(chunks: List[Dict], filename: str):
    """
    在终端打印所有 Chunk 的详细信息，用于验证元数据正确性和切分质量。

    输出内容包括：
      - Chunk 总数
      - 每个 Chunk 的编号、页码、字符数、文本预览
      - 各页码的 Chunk 分布统计

    参数:
        chunks   (List[Dict]): parse_pdf 的输出
        filename (str):        论文文件名（用于显示标题）
    """
    print(f"\n{'='*70}")
    print(f"📄 论文: {filename}")
    print(f"📊 共 {len(chunks)} 个文本块 (Chunk)")
    print(f"{'='*70}")

    for i, chunk in enumerate(chunks, 1):
        # 文本预览：取前 80 字符并合并换行
        text_preview = chunk["text"][:80].replace('\n', ' ')
        meta = chunk["metadata"]
        print(f"\n--- Chunk #{i:03d} ---")
        print(f"  📍 元数据: source='{meta['source']}', page={meta['page']}")
        print(f"  📏 字符数: {len(chunk['text'])}")
        print(f"  📝 预览: {text_preview}...")

    print(f"\n{'='*70}")

    # 统计每页的 Chunk 分布（用于评估切分均匀度）
    page_dist = {}
    for chunk in chunks:
        p = chunk["metadata"]["page"]
        page_dist[p] = page_dist.get(p, 0) + 1

    print(f"\n📊 页码分布统计:")
    for page in sorted(page_dist.keys()):
        bar = '█' * page_dist[page]  # 简单可视化
        print(f"  第 {page:4d} 页 → {page_dist[page]:3d} 个 chunk {bar}")
    print(f"{'='*70}\n")


# =============================================================================
# CLI 独立测试入口
# =============================================================================
if __name__ == "__main__":
    """
    命令行测试：直接运行 python pdf_parser.py <pdf_path> 验证解析效果。
    
    用法:
        python pdf_parser.py "Attention Is All You Need.pdf"
    
    输出:
        终端打印所有 Chunk 的详细信息及页码分布统计
    """
    import sys

    if len(sys.argv) < 2:
        print("用法: python pdf_parser.py <pdf_file_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    # 提取文件名（不含路径）
    filename = pdf_path.replace("\\", "/").split("/")[-1]

    # 执行解析流水线
    chunks = parse_pdf(pdf_bytes, filename)

    # 打印详细结果
    print_chunks(chunks, filename)