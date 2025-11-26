"""
会计准则识别器

在给定财报的元素列表和财年下，判定收入/金融工具/租赁三条线执行的新旧准则。
仅在过渡期（默认 2017-2021）启用精细判定，其余年份直接采用默认新/旧策略。
"""

import re
from typing import Any, Dict, List, Optional

from .models import DocumentElement, ElementType, StandardsProfile, StandardFlag


class StandardsDetector:
    """三层判定：显式披露 > 表格关键词 > 年份兜底"""

    def __init__(self, config: Dict[str, Any]):
        sd_cfg = (config or {}).get("standards_detection", {})
        self.enabled = sd_cfg.get("enabled", True)

        # 仅在过渡期窗口内启用精细判定
        self.transition_start = sd_cfg.get("transition_start_year", 2017)
        self.transition_end = sd_cfg.get("transition_end_year", 2021)

        # 过渡期内部的年份阈值（层 3 兜底）
        self.rev_threshold = sd_cfg.get("revenue_new_threshold_year", 2019)
        self.fin_threshold = sd_cfg.get("financial_new_threshold_year", 2019)
        self.lease_threshold = sd_cfg.get("lease_new_threshold_year", 2021)

        # 时间窗外默认策略
        self.default_before = sd_cfg.get("default_before_transition", "old")
        self.default_after = sd_cfg.get("default_after_transition", "new")

        # 表格关键词特征（表格命中新准则词是强信号）
        self.features = {
            "revenue": {
                "new": ["合同负债", "合同资产"],
                "old": ["预收款项", "预收账款", "建造合同"],
            },
            "financial_instruments": {
                "new": ["应收款项融资", "其他权益工具投资", "信用减值损失", "预期信用损失"],
                "old": ["可供出售金融资产", "持有至到期投资"],
            },
            "lease": {
                "new": ["使用权资产", "租赁负债"],
                "old": [],
            },
        }

    def detect(self, elements: List[DocumentElement], fiscal_year: Optional[int]) -> StandardsProfile:
        """入口：根据年份选择精细判定或默认策略"""
        profile = StandardsProfile()
        if not self.enabled:
            return profile

        if not fiscal_year:
            # 缺少年份时保持 UNKNOWN，由上层决定是否采用其他兜底
            return profile

        year = fiscal_year

        # 过渡期之外：直接默认
        if year < self.transition_start:
            flag = StandardFlag.OLD if self.default_before == "old" else StandardFlag.NEW
            profile.revenue = profile.financial_instruments = profile.lease = flag
            profile.decided_by = "default_before_transition"
            return profile

        if year > self.transition_end:
            flag = StandardFlag.NEW if self.default_after == "new" else StandardFlag.OLD
            profile.revenue = profile.financial_instruments = profile.lease = flag
            profile.decided_by = "default_after_transition"
            return profile

        # 过渡期内：精细判定
        self._detect_for_transition_period(elements, year, profile)
        return profile

    def _detect_for_transition_period(
        self, elements: List[DocumentElement], year: int, profile: StandardsProfile
    ) -> None:
        # 预取文本：显式披露看前若干元素，关键词看表格
        header_text = " ".join(e.content for e in elements[:300] if e.content)
        table_text = " ".join(
            e.content for e in elements
            if e.element_type == ElementType.TABLE and e.content
        )

        # 层 1：显式披露（最高优先级）
        patterns = [
            (r"(已|将|自.*?起)?执行.*?新?收入准则", "revenue"),
            (r"(已|将|自.*?起)?执行.*?新?金融工具.*?准则", "financial_instruments"),
            (r"(已|将|自.*?起)?执行.*?新?租赁准则", "lease"),
            (r"自(\d{4})年1月1日起.*执行.*第14号", "revenue"),
            (r"自(\d{4})年1月1日起.*执行.*第22号", "financial_instruments"),
            (r"自(\d{4})年1月1日起.*执行.*第21号", "lease"),
        ]

        explicit_hits = set()
        for pat, field in patterns:
            m = re.search(pat, header_text)
            if not m:
                continue
            ctx = header_text[max(0, m.start() - 20): m.end() + 20]
            # 检查否定
            if any(neg in ctx for neg in ("未执行", "暂不执行", "不适用", "未适用", "暂不适用")):
                setattr(profile, field, StandardFlag.OLD)
            else:
                # 如果带年份，结合 fiscal_year 判断是否已生效
                if m.groups() and m.group(1) and m.group(1).isdigit():
                    start_year = int(m.group(1))
                    flag = StandardFlag.NEW if year >= start_year else StandardFlag.OLD
                else:
                    flag = StandardFlag.NEW
                setattr(profile, field, flag)
            explicit_hits.add(field)

        if len(explicit_hits) == 3:
            profile.decided_by = "explicit_note"
            return

        # 层 2：表格关键词特征（仅处理未决字段）
        for field, feat in self.features.items():
            if getattr(profile, field) != StandardFlag.UNKNOWN:
                continue
            new_hit = any(k in table_text for k in feat["new"])
            old_hit = any(k in table_text for k in feat["old"])
            if new_hit:
                setattr(profile, field, StandardFlag.NEW)
            elif old_hit:
                setattr(profile, field, StandardFlag.OLD)

        if profile.decided_by == "unknown" and (
            profile.revenue != StandardFlag.UNKNOWN
            or profile.financial_instruments != StandardFlag.UNKNOWN
            or profile.lease != StandardFlag.UNKNOWN
        ):
            profile.decided_by = "table_keywords"

        # 层 3：年份兜底（仅对 UNKNOWN）
        used_threshold = False
        if profile.revenue == StandardFlag.UNKNOWN:
            profile.revenue = StandardFlag.NEW if year >= self.rev_threshold else StandardFlag.OLD
            used_threshold = True
        if profile.financial_instruments == StandardFlag.UNKNOWN:
            profile.financial_instruments = StandardFlag.NEW if year >= self.fin_threshold else StandardFlag.OLD
            used_threshold = True
        if profile.lease == StandardFlag.UNKNOWN:
            profile.lease = StandardFlag.NEW if year >= self.lease_threshold else StandardFlag.OLD
            used_threshold = True

        if used_threshold:
            profile.decided_by = (
                "year_threshold"
                if profile.decided_by == "unknown"
                else profile.decided_by + "_with_year_fallback"
            )
