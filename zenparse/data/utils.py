"""
工具函数集合

提供文本处理、质量评估和辅助功能。
"""

import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter


def calculate_chinese_ratio(text: str) -> float:
    """计算中文字符比例
    
    Args:
        text: 输入文本
    
    Returns:
        中文字符占总字符的比例
    """
    if not text:
        return 0.0
    
    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    return chinese_chars / len(text)


def is_chinese_text(text: str, threshold: float = 0.3) -> bool:
    """判断是否为中文文本
    
    Args:
        text: 输入文本
        threshold: 中文字符比例阈值
    
    Returns:
        是否为中文文本
    """
    return calculate_chinese_ratio(text) >= threshold


def extract_numbers(text: str) -> List[str]:
    """提取文本中的数字和金额
    
    Args:
        text: 输入文本
    
    Returns:
        数字列表
    """
    patterns = [
        r'\d+\.?\d*[万亿千百]?元',  # 金额
        r'\d+\.?\d*%',              # 百分比
        r'\d{4}年\d{1,2}月\d{1,2}日',  # 日期
        r'\d+\.?\d*',               # 普通数字
    ]
    
    numbers = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        numbers.extend(matches)
    
    return numbers


def extract_financial_terms(text: str) -> List[str]:
    """提取财务术语
    
    Args:
        text: 输入文本
    
    Returns:
        财务术语列表
    """
    # 预定义财务术语词典
    financial_terms = [
        '资产', '负债', '所有者权益', '股东权益',
        '流动资产', '非流动资产', '流动负债', '非流动负债',
        '货币资金', '应收账款', '应付账款', '存货',
        '固定资产', '无形资产', '商誉', '递延所得税',
        '营业收入', '营业成本', '营业利润', '净利润',
        '毛利率', '净利率', 'ROE', 'ROA', 'EPS',
        '经营活动现金流', '投资活动现金流', '筹资活动现金流',
        '资产负债率', '流动比率', '速动比率'
    ]
    
    found_terms = []
    for term in financial_terms:
        if term in text:
            found_terms.append(term)
    
    return found_terms


def clean_text(text: str) -> str:
    """清理文本
    
    Args:
        text: 输入文本
    
    Returns:
        清理后的文本
    """
    if not text:
        return ""
    
    # 移除多余空白
    text = re.sub(r'\s+', ' ', text)
    
    # 移除特殊字符
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)
    
    # 统一引号
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # 移除首尾空白
    text = text.strip()
    
    return text


def split_sentences(text: str) -> List[str]:
    """分割句子
    
    Args:
        text: 输入文本
    
    Returns:
        句子列表
    """
    # 中文句子分割
    sentences = re.split(r'[。！？\n]+', text)
    
    # 英文句子分割
    if calculate_chinese_ratio(text) < 0.5:
        sentences = re.split(r'[.!?\n]+', text)
    
    # 过滤空句子
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def calculate_text_similarity(text1: str, text2: str) -> float:
    """计算文本相似度（简单实现）
    
    Args:
        text1: 文本1
        text2: 文本2
    
    Returns:
        相似度分数 (0-1)
    """
    if not text1 or not text2:
        return 0.0
    
    # 转换为集合
    chars1 = set(text1)
    chars2 = set(text2)
    
    # Jaccard相似度
    intersection = chars1.intersection(chars2)
    union = chars1.union(chars2)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)


def generate_chunk_id(content: str, prefix: str = "chunk") -> str:
    """生成分块ID
    
    Args:
        content: 分块内容
        prefix: ID前缀
    
    Returns:
        分块ID
    """
    # 使用内容哈希生成ID
    content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
    return f"{prefix}_{content_hash}"


def estimate_tokens(text: str, model: str = "chinese") -> int:
    """估算文本token数
    
    Args:
        text: 输入文本
        model: 模型类型
    
    Returns:
        估算的token数
    """
    if model == "chinese":
        # 中文模型：平均每个字符1.5个token
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return int(chinese_chars * 1.5 + other_chars * 0.5)
    else:
        # 英文模型：平均每4个字符1个token
        return len(text) // 4


def validate_chunk_size(
    chunk: str,
    min_size: int = 50,
    max_size: int = 4000,
    optimal_size: int = 1000
) -> Dict[str, Any]:
    """验证分块大小
    
    Args:
        chunk: 分块内容
        min_size: 最小大小
        max_size: 最大大小
        optimal_size: 最优大小
    
    Returns:
        验证结果
    """
    size = len(chunk)
    
    result = {
        'size': size,
        'is_valid': min_size <= size <= max_size,
        'is_optimal': abs(size - optimal_size) < optimal_size * 0.2,
        'message': ''
    }
    
    if size < min_size:
        result['message'] = f"分块过小 ({size}字符 < {min_size})"
    elif size > max_size:
        result['message'] = f"分块过大 ({size}字符 > {max_size})"
    elif result['is_optimal']:
        result['message'] = "分块大小最优"
    else:
        result['message'] = "分块大小可接受"
    
    return result


def extract_metadata(text: str) -> Dict[str, Any]:
    """提取文本元数据
    
    Args:
        text: 输入文本
    
    Returns:
        元数据字典
    """
    metadata = {
        'length': len(text),
        'chinese_ratio': calculate_chinese_ratio(text),
        'sentence_count': len(split_sentences(text)),
        'has_numbers': bool(extract_numbers(text)),
        'has_financial_terms': bool(extract_financial_terms(text)),
        'line_count': text.count('\n') + 1
    }
    
    # 提取年份
    years = re.findall(r'(\d{4})年', text)
    if years:
        metadata['years'] = list(set(years))
    
    # 提取公司名称 - 扩展支持更多公司类型，支持括号内容和特殊格式
    company_pattern = r'([\u4e00-\u9fff]+(?:股份|集团|控股|科技|技术|金融|银行|证券|保险|地产|能源|医药|电子|通信|汽车|钢铁|化工|实业|投资|建设|贸易|物流|文化|教育|农业|航空|铁路|发展|企业)(?:（[^）]*）)?(?:有限|有限责任)?(?:股份)?(?:公司|企业))'
    companies = re.findall(company_pattern, text)
    if companies:
        metadata['companies'] = list(set(companies))
    
    return metadata


def merge_chunks(chunks: List[str], separator: str = "\n\n") -> str:
    """合并分块
    
    Args:
        chunks: 分块列表
        separator: 分隔符
    
    Returns:
        合并后的文本
    """
    return separator.join(chunk.strip() for chunk in chunks if chunk.strip())


def deduplicate_chunks(chunks: List[Dict[str, Any]], key: str = 'content') -> List[Dict[str, Any]]:
    """去重分块
    
    Args:
        chunks: 分块列表
        key: 用于去重的键
    
    Returns:
        去重后的分块列表
    """
    seen = set()
    unique_chunks = []
    
    for chunk in chunks:
        chunk_hash = hashlib.md5(str(chunk.get(key, '')).encode()).hexdigest()
        if chunk_hash not in seen:
            seen.add(chunk_hash)
            unique_chunks.append(chunk)
    
    return unique_chunks


def find_overlapping_chunks(
    chunks: List[Tuple[int, int]],
    threshold: float = 0.1
) -> List[Tuple[int, int]]:
    """查找重叠的分块
    
    Args:
        chunks: 分块位置列表 [(start, end), ...]
        threshold: 重叠阈值
    
    Returns:
        重叠的分块对
    """
    overlapping = []
    
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            start1, end1 = chunks[i]
            start2, end2 = chunks[j]
            
            # 计算重叠
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)
            
            if overlap_start < overlap_end:
                overlap_size = overlap_end - overlap_start
                min_size = min(end1 - start1, end2 - start2)
                
                if overlap_size / min_size > threshold:
                    overlapping.append((i, j))
    
    return overlapping


def calculate_chunk_quality(
    chunk: str,
    criteria: Optional[Dict[str, float]] = None
) -> float:
    """计算分块质量分数
    
    Args:
        chunk: 分块内容
        criteria: 评分标准权重
    
    Returns:
        质量分数 (0-1)
    """
    if not chunk:
        return 0.0
    
    # 默认评分标准
    if criteria is None:
        criteria = {
            'length': 0.2,
            'chinese': 0.2,
            'financial': 0.3,
            'structure': 0.3
        }
    
    scores = {}
    
    # 长度评分
    length = len(chunk)
    if length < 50:
        scores['length'] = 0.2
    elif length < 200:
        scores['length'] = 0.5
    elif length < 1000:
        scores['length'] = 1.0
    elif length < 2000:
        scores['length'] = 0.8
    else:
        scores['length'] = 0.6
    
    # 中文内容评分
    chinese_ratio = calculate_chinese_ratio(chunk)
    scores['chinese'] = chinese_ratio
    
    # 财务内容评分
    financial_terms = extract_financial_terms(chunk)
    scores['financial'] = min(1.0, len(financial_terms) / 5)
    
    # 结构评分（有标点、分段等）
    has_punctuation = bool(re.search(r'[，。！？；：]', chunk))
    has_structure = '\n' in chunk or '。' in chunk
    scores['structure'] = (0.5 if has_punctuation else 0) + (0.5 if has_structure else 0)
    
    # 加权计算总分
    total_score = sum(scores[k] * criteria.get(k, 0) for k in scores)
    
    return min(1.0, total_score)


def format_chunk_for_display(
    chunk: Dict[str, Any],
    max_length: int = 200,
    show_metadata: bool = True
) -> str:
    """格式化分块用于显示
    
    Args:
        chunk: 分块数据
        max_length: 最大显示长度
        show_metadata: 是否显示元数据
    
    Returns:
        格式化的字符串
    """
    lines = []
    
    # 分块ID
    chunk_id = chunk.get('chunk_id', 'unknown')
    lines.append(f"[分块 {chunk_id}]")
    
    # 内容预览
    content = chunk.get('content', '')
    if len(content) > max_length:
        content = content[:max_length] + "..."
    lines.append(f"内容: {content}")
    
    # 元数据
    if show_metadata:
        metadata = chunk.get('metadata', {})
        if metadata:
            lines.append("元数据:")
            for key, value in metadata.items():
                if isinstance(value, list):
                    value = f"[{len(value)} items]"
                lines.append(f"  - {key}: {value}")
    
    return '\n'.join(lines)