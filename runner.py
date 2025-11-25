"""
独立运行器：批量处理 PDF -> JSON 分块结果
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime

# 允许直接在仓库根目录执行
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import List, Dict, Any, Optional, Set

from zenparse.pipeline import ZenPipeline  # noqa: E402
from zenparse.core.logger import get_logger  # noqa: E402
from zenparse.core.exceptions import ZenParseError  # noqa: E402
from zenparse.data.models import ParentChunk, ChildChunk, TableGroup  # noqa: E402

logger = get_logger("Runner")


def _convert_pages_to_ranges(pages: List[int]) -> List[List[int]]:
    """
    将页码列表转换为范围数组格式
    
    输入: [1, 2, 3, 5, 6, 7, 100, 101]
    输出: [[1, 3], [5, 7], [100, 101]]
    
    Args:
        pages: 排序后的页码列表
        
    Returns:
        范围数组，每个范围为 [起始页, 结束页]
    """
    if not pages:
        return []
    
    ranges = []
    start = pages[0]
    end = pages[0]
    
    for i in range(1, len(pages)):
        if pages[i] == end + 1:
            # 连续页码，扩展当前范围
            end = pages[i]
        else:
            # 遇到间断，保存当前范围
            ranges.append([start, end])
            start = pages[i]
            end = pages[i]
    
    # 保存最后一个范围
    ranges.append([start, end])
    
    return ranges


def _build_overview(
    parents: List[ParentChunk],
    children: List[ChildChunk],
    table_groups: List[TableGroup],
    source_file: str,
) -> Dict[str, Any]:
    """生成简要可追溯概览，便于快速审阅"""
    parent_count = len(parents)
    child_count = len(children)
    table_group_count = len(table_groups)

    pages = set()
    for chunk in parents + children:
        if hasattr(chunk, "metadata") and getattr(chunk, "metadata", None):
            pages.update(chunk.metadata.page_numbers or [])

    quality_buckets = {"high": 0, "medium": 0, "low": 0}
    for chunk in parents + children:
        score = getattr(chunk, "quality_score", 0)
        if score >= 0.8:
            quality_buckets["high"] += 1
        elif score >= 0.5:
            quality_buckets["medium"] += 1
        else:
            quality_buckets["low"] += 1

    def preview(chunk: ParentChunk, max_len: int = 120) -> Dict[str, Any]:
        text = (chunk.content or "").strip()
        return {
            "chunk_id": chunk.chunk_id,
            "chars": len(text),
            "preview": text[:max_len] + ("..." if len(text) > max_len else ""),
            "page_numbers": getattr(chunk, "metadata", {}).page_numbers if getattr(chunk, "metadata", None) else [],
            "child_count": len(getattr(chunk, "child_ids", []) or []),
        }

    # 选取若干代表性父块（最长的前 5 个）
    top_parents = sorted(parents, key=lambda c: len(c.content or ""), reverse=True)[:5]

    return {
        "source_file": source_file,
        "counts": {
            "parents": parent_count,
            "children": child_count,
            "table_groups": table_group_count,
            "pages": _convert_pages_to_ranges(sorted(pages)),
            "page_count": len(pages),
        },
        "quality": quality_buckets,
        "avg_child_per_parent": round(child_count / parent_count, 2) if parent_count else 0,
        "parent_previews": [preview(p) for p in top_parents],
    }


def _contains_path_separator(value: str) -> bool:
    """判断字符串是否包含任意路径分隔符"""
    separators = {os.path.sep, "/", "\\"}
    return any(sep in value for sep in separators if sep)


def _resolve_pdf_path(filename: str, search_dirs: List[str]) -> Optional[str]:
    """根据搜索目录解析 PDF 文件路径"""
    if not filename:
        return None

    normalized = filename.strip()
    if not normalized:
        return None

    if _contains_path_separator(normalized):
        candidate = os.path.abspath(normalized)
        return candidate if os.path.exists(candidate) else None

    lowered = normalized.lower()
    for search_dir in search_dirs:
        base_candidate = os.path.join(search_dir, normalized)
        if os.path.exists(base_candidate):
            return base_candidate

        if not lowered.endswith(".pdf"):
            pdf_candidate = os.path.join(search_dir, f"{normalized}.pdf")
            if os.path.exists(pdf_candidate):
                return pdf_candidate

    return None


def _load_file_list(file_list_path: str, search_dirs: Optional[List[str]] = None) -> List[str]:
    """从文件中加载 PDF 文件名列表
    
    Args:
        file_list_path: 包含文件名的文本文件路径（每行一个文件名）
        search_dirs: 搜索文件的目录列表（如果文件名不包含完整路径）
        
    Returns:
        PDF 文件路径列表
    """
    pdf_files = []
    
    if not os.path.exists(file_list_path):
        logger.error(f"文件列表不存在: {file_list_path}")
        return pdf_files
    
    search_dirs = search_dirs or []

    with open(file_list_path, "r", encoding="utf-8") as f:
        for line in f:
            filename = line.strip()
            if not filename or filename.startswith("#"):
                continue  # 跳过空行和注释
            
            resolved = _resolve_pdf_path(filename, search_dirs)
            if resolved:
                pdf_files.append(os.path.abspath(resolved))
            else:
                if search_dirs:
                    logger.warning(
                        f"在以下目录中均未找到文件: {filename} (搜索目录: {', '.join(search_dirs)})"
                    )
                else:
                    logger.warning(f"无法定位文件（未指定搜索目录）: {filename}")
    
    return pdf_files


def _parse_file_names(file_names_str: str, search_dirs: Optional[List[str]] = None) -> List[str]:
    """解析逗号分隔的文件名列表
    
    Args:
        file_names_str: 逗号分隔的文件名字符串
        search_dirs: 搜索文件的目录列表（如果文件名不包含完整路径）
        
    Returns:
        PDF 文件路径列表
    """
    pdf_files = []
    search_dirs = search_dirs or []
    
    for filename in file_names_str.split(","):
        filename = filename.strip()
        if not filename:
            continue
        
        resolved = _resolve_pdf_path(filename, search_dirs)
        if resolved:
            pdf_files.append(os.path.abspath(resolved))
        else:
            if search_dirs:
                logger.warning(
                    f"在以下目录中均未找到文件: {filename} (搜索目录: {', '.join(search_dirs)})"
                )
            else:
                logger.warning(f"无法定位文件（未指定搜索目录）: {filename}")
    
    return pdf_files


def process_batch(
    pipeline: ZenPipeline,
    input_paths: Optional[List[str]],
    output_dir: str,
    use_timestamp: bool = False,
    file_list_path: str = None,
    file_names: str = None,
    search_dirs: Optional[List[str]] = None,
) -> None:
    """处理给定输入及其辅助列表
    
    Args:
        pipeline: ZenParse 处理管道
        input_paths: 直接指定的文件或目录列表
        output_dir: 输出目录
        use_timestamp: 是否在文件名中添加时间戳后缀（避免覆盖）
        file_list_path: 包含文件名的文本文件路径（每行一个文件名）
        file_names: 逗号分隔的文件名列表
        search_dirs: 解析文件列表/文件名时的搜索目录
    """
    input_paths = input_paths or []
    user_search_dirs = search_dirs or []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    normalized_search_dirs: List[str] = []
    seen_dirs: Set[str] = set()

    def _add_search_dir(path: str) -> None:
        if not path:
            return
        abs_dir = os.path.abspath(path)
        if not os.path.isdir(abs_dir):
            logger.warning(f"搜索目录不存在: {path}")
            return
        if abs_dir not in seen_dirs:
            normalized_search_dirs.append(abs_dir)
            seen_dirs.add(abs_dir)

    for directory in user_search_dirs:
        _add_search_dir(directory)

    for path in input_paths:
        if not path:
            continue
        abs_path = os.path.abspath(path)
        if os.path.isdir(abs_path):
            _add_search_dir(abs_path)
        else:
            parent_dir = os.path.dirname(abs_path)
            if parent_dir:
                _add_search_dir(parent_dir)

    pdf_files: List[str] = []

    for path in input_paths:
        if not path:
            continue
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            logger.warning(f"输入路径不存在，已跳过: {path}")
            continue

        if os.path.isdir(abs_path):
            matched = sorted(glob.glob(os.path.join(abs_path, "*.pdf")))
            if not matched:
                logger.warning(f"目录中未找到 PDF 文件: {path}")
            pdf_files.extend(matched)
        else:
            pdf_files.append(abs_path)

    if file_list_path:
        list_results = _load_file_list(file_list_path, normalized_search_dirs)
        if not list_results:
            logger.warning("文件列表为空或所有文件都未找到")
        pdf_files.extend(list_results)

    if file_names:
        name_results = _parse_file_names(file_names, normalized_search_dirs)
        if not name_results:
            logger.warning("文件名列表为空或所有文件都未找到")
        pdf_files.extend(name_results)

    unique_files: List[str] = []
    seen_files: Set[str] = set()
    for pdf in pdf_files:
        abs_pdf = os.path.abspath(pdf)
        if abs_pdf not in seen_files:
            unique_files.append(abs_pdf)
            seen_files.add(abs_pdf)

    if not unique_files:
        logger.error("没有可处理的文件")
        return

    total = len(unique_files)
    logger.info(f"开始处理 {total} 个文件")

    for i, pdf_path in enumerate(unique_files, 1):
        file_name = os.path.basename(pdf_path)
        base_name = os.path.splitext(file_name)[0]
        
        # 根据 use_timestamp 参数决定是否添加时间戳
        if use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_name = f"{base_name}_chunks_{timestamp}.json"
        else:
            json_name = f"{base_name}_chunks.json"
        
        save_path = os.path.join(output_dir, json_name)

        # 如果使用时间戳模式，不检查文件是否存在（总是生成新文件）
        if not use_timestamp and os.path.exists(save_path):
            logger.info(f"[{i}/{total}] 跳过已存在: {file_name}")
            continue

        try:
            logger.info(f"[{i}/{total}] 处理: {file_name}")
            parents, children, table_groups = pipeline.process(pdf_path)

            overview = _build_overview(parents, children, table_groups, file_name)

            result_data = {
                "source": file_name,
                "overview": overview,
                "parents": [p.to_dict() for p in parents],
                "children": [c.to_dict() for c in children],
            }

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)

        except ZenParseError as zpe:
            logger.error(f"[{i}/{total}] 业务异常: {file_name} - {zpe}")
        except Exception as exc:
            logger.error(f"[{i}/{total}] 未知错误: {file_name} - {exc}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(
        description="ZenParse 批量处理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理单个文件
  python runner.py -i input/pdfs/file.pdf
  
  # 处理多个目录/文件
  python runner.py -i input/pdfs/ -i extra/another.pdf
  
  # 使用文件列表并在多个目录中搜索
  python runner.py --file-list file_list.txt --search-dirs input/pdfs/ input/pdfs_raw/
        """
    )
    parser.add_argument(
        "--input",
        "-i",
        action="append",
        dest="inputs",
        default=[],
        help="输入 PDF 文件或目录，可重复提供多次"
    )
    parser.add_argument("--output", "-o", default="output", help="输出 JSON 目录")
    parser.add_argument("--config", "-c", default="config.yaml", help="配置文件路径")
    parser.add_argument("--timestamp", "-t", action="store_true", 
                       help="在输出文件名中添加时间戳后缀（避免覆盖现有文件）")
    parser.add_argument("--file-list", "-l", 
                       help="包含 PDF 文件名的文本文件路径（每行一个文件名，支持 # 开头的注释行）")
    parser.add_argument("--files", "-f",
                       help="逗号分隔的 PDF 文件名列表（例如: file1.pdf,file2.pdf,file3.pdf）")
    parser.add_argument(
        "--search-dirs",
        "-s",
        action="append",
        nargs="+",
        default=[],
        help="用于解析文件列表/文件名的额外搜索目录，可一次提供多个"
    )

    args = parser.parse_args()

    input_paths = args.inputs or []
    search_dirs: List[str] = []
    for group in args.search_dirs:
        search_dirs.extend(group)

    # 验证参数组合
    if args.file_list and args.files:
        logger.error("不能同时使用 --file-list 和 --files 参数")
        sys.exit(1)
    
    if not input_paths and not args.file_list and not args.files:
        logger.error("必须至少提供一个 --input、--file-list 或 --files 参数")
        sys.exit(1)

    try:
        pipeline = ZenPipeline(args.config)
        process_batch(
            pipeline, 
            input_paths, 
            args.output, 
            use_timestamp=args.timestamp,
            file_list_path=args.file_list,
            file_names=args.files,
            search_dirs=search_dirs,
        )
    except Exception as exc:
        logger.critical(f"程序启动失败: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
