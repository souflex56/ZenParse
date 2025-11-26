"""
统一指标访问层

根据准则判定结果，在新旧科目之间做自动切换和回退。
_get_raw_value_from_chunks 需要接入你现有的取数逻辑。
"""

from typing import Any, Dict, List, Tuple

from .standards_mapping import STANDARD_SUBJECT_MAPPING
from .models import StandardsProfile, StandardFlag


class MetricsAccessor:
    """对外暴露业务指标名，内部做新旧准则适配"""

    def __init__(self, report_data: Dict[str, Any], standards_profile: Any):
        """
        :param report_data: pipeline 生成的完整 JSON 数据
        :param standards_profile: StandardsProfile 实例或其字典表示
        """
        self.data = report_data
        if isinstance(standards_profile, StandardsProfile):
            self.profile_dict = standards_profile.to_dict()
        else:
            self.profile_dict = standards_profile or {}

        self.mapping = STANDARD_SUBJECT_MAPPING

        # 指标所属的准则类别（决定使用 StandardsProfile 中的哪条旗标）
        self.metric_category = {
            "contract_liabilities_like": "revenue",
            "contract_assets_like": "revenue",
            "lease_liabilities_total": "lease",
            "notes_receivable_broad": "financial_instruments",
            "financial_investments_equity": "financial_instruments",
            "financial_investments_debt": "financial_instruments",
            "financial_investments_total": "financial_instruments",
            "impairment_losses": "financial_instruments",
            "rd_expenses": "revenue",
            "monetary_funds_broad": "financial_instruments",
            "taxes_and_surcharges": "revenue",
        }

    def _get_raw_value_from_chunks(self, subject_name: str) -> Any:
        """
        从解析结果中按科目提取数值。
        TODO: 需要与你现有的表格取数逻辑对接，当前占位返回 None。
        """
        # 占位实现：返回 None 表示未找到，避免错误
        return None

    def _try_extract(self, subject_list: List[str]) -> Tuple[float, List[str], bool]:
        """
        尝试从科目列表中取值并求和

        返回:
            total: 求和后的数值（仅累加非零）
            found_subjects: 实际命中的科目名列表
            has_data: 是否找到任何科目（即便值为 0 也视为找到，避免误判为“缺失”而回退）
        """
        total = 0.0
        found_subjects: List[str] = []
        has_data = False  # 表示是否找到过科目行，避免将“值为 0”误判为未找到
        for subject in subject_list:
            val = self._get_raw_value_from_chunks(subject)
            if val is None:
                continue
            # 看到科目行即可认为命中，避免 0 触发回退
            has_data = True
            try:
                if isinstance(val, str):
                    cleaned = val.replace(",", "").strip()
                    if not cleaned:
                        continue
                    val = float(cleaned)
                val_num = float(val)
                # 只有非零才累加，零值不影响总和但仍算命中
                if val_num != 0:
                    total += val_num
                    found_subjects.append(subject)
            except (TypeError, ValueError):
                continue
        return total, found_subjects, has_data

    def get_metric(self, metric_key: str) -> Dict[str, Any]:
        """
        获取统一口径指标：
        - 优先使用当前准则对应的科目列表
        - 若完全未取到，再回退到另一套科目
        """
        if metric_key not in self.mapping:
            return {"error": f"Metric '{metric_key}' not defined"}

        conf = self.mapping[metric_key]
        category = self.metric_category.get(metric_key, "revenue")
        profile_flag = self.profile_dict.get(category, StandardFlag.UNKNOWN.value)

        # 默认策略：非 old 均先试 new
        prefer_new = profile_flag != StandardFlag.OLD.value
        primary_list = conf["new"] if prefer_new else conf["old"]
        fallback_list = conf["old"] if prefer_new else conf["new"]

        total_val, subjects_used, success = self._try_extract(primary_list)
        standard_used = "new" if prefer_new else "old"

        if not success:
            total_val, subjects_used, success = self._try_extract(fallback_list)
            if success:
                standard_used = ("old" if prefer_new else "new") + "_fallback"

        note_parts: List[str] = []

        # 研发费用提示：旧准则或取值为0时提示口径问题
        if metric_key == "rd_expenses":
            if standard_used.startswith("old") or total_val == 0:
                note_parts.append("旧准则：研发费用通常包含在管理费用中，未单独列示，数值可能为0或偏低。")

        # 预收类负债增值税提示
        if metric_key == "contract_liabilities_like":
            if "合同负债" in subjects_used:
                note_parts.append("新准则口径：合同负债不含增值税。")
            elif any(s in subjects_used for s in ["预收款项", "预收账款"]):
                note_parts.append("旧准则口径：预收款项/预收账款含增值税，跨准则对比需注意税率差异。")

        # 广义应收票据提示
        if metric_key == "notes_receivable_broad" and "应收款项融资" in subjects_used:
            note_parts.append("已包含应收款项融资（用于背书/贴现的票据）。")

        # 广义货币资金提示
        if metric_key == "monetary_funds_broad" and "拆出资金" in subjects_used:
            note_parts.append("已加回拆出资金以还原集团资金池全貌。")

        return {
            "metric": metric_key,
            "value": total_val,
            "subjects_used": subjects_used,
            "standard_used": standard_used,
            "profile_flag": profile_flag,
            "note": " ".join(note_parts).strip(),
        }
