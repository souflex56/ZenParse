"""
Quick preview/export helper for ZenParse chunk JSON.

功能:
- 终端预览：按“页码 + 页内位置”排序，输出轻量清单（带 layer 字段）
- 导出 Excel：默认拆成两个 sheet（parents / children），各自按页序排序

Usage:
  # 终端预览（前 50 条，父+子）
  python scripts/preview_chunks.py output/xxx.json --show-children --limit 50
  # 只看表格块
  python scripts/preview_chunks.py output/xxx.json --tables-only
  # 导出 Excel（父/子分表）
  python scripts/preview_chunks.py output/xxx.json --show-children --xlsx output/review.xlsx
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas 可能未安装
    pd = None


def entry_page(chunk: Dict[str, Any]) -> int:
    """Pick the earliest page number available on a chunk."""
    pages: List[int] = []
    if chunk.get("page_number") is not None:
        pages.append(int(chunk["page_number"]))

    meta = chunk.get("metadata") or {}
    for p in meta.get("page_numbers") or []:
        try:
            pages.append(int(p))
        except (TypeError, ValueError):
            continue

    return min(pages) if pages else math.inf


def y_top(chunk: Dict[str, Any]) -> float:
    """Use bbox y0 (top) to order chunks on the same page; fallback to inf."""
    bbox = chunk.get("bbox")
    if isinstance(bbox, Sequence) and len(bbox) >= 2:
        try:
            return float(bbox[1])
        except (TypeError, ValueError):
            return math.inf
    return math.inf


def position(chunk: Dict[str, Any]) -> int:
    """Relative order inside parent (children use this)."""
    try:
        return int(chunk.get("position") or 0)
    except (TypeError, ValueError):
        return 0


def snippet(text: str, length: int = 80) -> str:
    clean = (text or "").strip().replace("\n", " ")
    return clean[:length] + ("..." if len(clean) > length else "")


def as_row(chunk: Dict[str, Any], idx: int, snip_len: int, layer: str) -> Dict[str, Any]:
    """Flatten chunk fields for tabular export."""
    meta = chunk.get("metadata") or {}
    pages = meta.get("page_numbers") or []
    bbox = chunk.get("bbox")
    return {
        "idx": idx,
        "layer": layer,
        "chunk_id": chunk.get("chunk_id"),
        "chunk_type": chunk.get("chunk_type"),
        "is_table": bool(chunk.get("is_table")) or chunk.get("chunk_type") == "table_group",
        "entry_page": None if entry_page(chunk) is math.inf else int(entry_page(chunk)),
        "page_number": chunk.get("page_number"),
        "page_numbers": ",".join(str(p) for p in pages) if pages else "",
        "bbox": bbox,
        "y0": None if y_top(chunk) is math.inf else float(y_top(chunk)),
        "position": position(chunk),
        "quality_score": chunk.get("quality_score"),
        "parent_id": chunk.get("parent_id"),
        "child_count": len(chunk.get("child_ids") or []),
        "section_title": meta.get("section_title") or "",
        "section_path": " / ".join(meta.get("section_path") or []),
        "content_len": len(chunk.get("content") or ""),
        "content": chunk.get("content", "") or "",
    }


def load_chunks(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    parents = data.get("parents") or []
    children = data.get("children") or []
    return {"parents": parents, "children": children}


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview/Export ZenParse chunk JSON")
    parser.add_argument("json_path", help="Path to *_chunks.json produced by runner.py")
    parser.add_argument("--limit", type=int, default=50, help="Max rows to print/export (0 = no limit)")
    parser.add_argument("--tables-only", action="store_true", help="Only show table chunks")
    parser.add_argument("--include-sections", action="store_true", help="Include SECTION chunks (default off)")
    parser.add_argument("--show-children", action="store_true", help="Include children in the list (default off)")
    parser.add_argument("--xlsx", help="Export to Excel file path (e.g., output/review.xlsx)")
    parser.add_argument("--snippet-len", type=int, default=120, help="Snippet length for preview/export")
    args = parser.parse_args()

    path = Path(args.json_path)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    data = load_chunks(path)
    parents_raw = data["parents"]
    children_raw = data["children"]

    def _filter(ch: Dict[str, Any], is_parent: bool) -> bool:
        ctype = ch.get("chunk_type")
        if args.tables_only and not ch.get("is_table") and ctype != "table_group":
            return False
        if is_parent and (not args.include_sections) and ctype == "section":
            return False
        return True

    parents = [ch for ch in parents_raw if _filter(ch, True)]
    children = [ch for ch in children_raw if args.show_children and _filter(ch, False)]

    parents_sorted = sorted(parents, key=lambda c: (entry_page(c), y_top(c), position(c)))
    children_sorted = sorted(children, key=lambda c: (entry_page(c), y_top(c), position(c)))

    limit = args.limit if args.limit and args.limit > 0 else None

    # Excel 导出
    if args.xlsx:
        if pd is None:
            raise SystemExit("pandas 未安装，无法导出 Excel。请先安装 pandas 和 openpyxl。")

        def _rows(chunks: List[Dict[str, Any]], layer: str) -> List[Dict[str, Any]]:
            end = limit if limit else len(chunks)
            return [as_row(ch, i, args.snippet_len, layer) for i, ch in enumerate(chunks[:end], 1)]

        parent_rows = _rows(parents_sorted, "parent")
        child_rows = _rows(children_sorted, "child") if children_sorted else []

        try:
            with pd.ExcelWriter(args.xlsx) as writer:
                pd.DataFrame(parent_rows).to_excel(writer, sheet_name="parents", index=False)
                if child_rows:
                    pd.DataFrame(child_rows).to_excel(writer, sheet_name="children", index=False)
        except Exception as exc:  # pragma: no cover - 运行时失败提示
            raise SystemExit(f"导出 Excel 失败: {exc}\n请确认已安装 openpyxl。")

        print(
            f"Exported parents={len(parent_rows)}, children={len(child_rows)} to {args.xlsx} "
            f"(raw parents={len(parents_raw)}, raw children={len(children_raw)})"
        )
        return

    # 终端预览
    parent_show = len(parents_sorted) if not limit else min(limit, len(parents_sorted))
    child_show = len(children_sorted) if not limit else min(limit, len(children_sorted))
    print(
        f"File: {path.name} | parents={len(data['parents'])} "
        f"children={len(data['children'])} | showing parents={parent_show}/{len(parents_sorted)} "
        f"children={child_show}/{len(children_sorted)}"
    )
    print("idx\tlayer\tpage\ty0\tpos\ttype\tis_table\tquality\ttext")

    def _print(chunks: List[Dict[str, Any]], layer: str, count: int) -> None:
        for idx, ch in enumerate(chunks[:count], 1):
            page = entry_page(ch)
            page_str = "NA" if page is math.inf else str(int(page))
            y0 = y_top(ch)
            y_str = "NA" if y0 is math.inf else f"{y0:.1f}"
            quality = ch.get("quality_score")
            q_str = f"{quality:.2f}" if isinstance(quality, (int, float)) else "NA"
            print(
                f"{idx}\t{layer}\t{page_str}\t{y_str}\t{position(ch)}\t"
                f"{ch.get('chunk_type')}\t{bool(ch.get('is_table'))}\t{q_str}\t"
                f"{snippet(ch.get('content', ''), args.snippet_len)}"
            )

    _print(parents_sorted, "parent", parent_show)
    if children_sorted:
        _print(children_sorted, "child", child_show)


if __name__ == "__main__":
    main()
