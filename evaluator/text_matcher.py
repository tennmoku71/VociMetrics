"""テキスト比較エンジン（STT結果と期待テキストの比較）"""

import logging
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class MatchMethod(Enum):
    """比較方法"""
    EXACT = "exact"  # 完全一致
    EDIT_DISTANCE = "edit_distance"  # 編集距離（Levenshtein距離）
    LLM = "llm"  # LLMベースの比較（未実装）


class TextMatcher:
    """テキスト比較エンジン"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.match_method = MatchMethod(config.get("match_method", "exact"))
        self.edit_distance_threshold = config.get("edit_distance_threshold", 0.8)  # 0.0-1.0（類似度）
    
    def match(self, expected_text: str, actual_text: str) -> Dict[str, Any]:
        """
        期待テキストと実際のテキストを比較します。
        
        Args:
            expected_text: 期待するテキスト
            actual_text: STTで認識されたテキスト
            
        Returns:
            比較結果の辞書:
            {
                "matched": bool,  # 一致したかどうか
                "method": str,  # 使用した比較方法
                "score": float,  # 類似度スコア（0.0-1.0）
                "details": dict  # 詳細情報
            }
        """
        if self.match_method == MatchMethod.EXACT:
            return self._exact_match(expected_text, actual_text)
        elif self.match_method == MatchMethod.EDIT_DISTANCE:
            return self._edit_distance_match(expected_text, actual_text)
        elif self.match_method == MatchMethod.LLM:
            return self._llm_match(expected_text, actual_text)
        else:
            logger.warning(f"Unknown match method: {self.match_method}")
            return {
                "matched": False,
                "method": str(self.match_method),
                "score": 0.0,
                "details": {}
            }
    
    def _exact_match(self, expected_text: str, actual_text: str) -> Dict[str, Any]:
        """完全一致による比較"""
        expected_normalized = self._normalize_text(expected_text)
        actual_normalized = self._normalize_text(actual_text)
        
        matched = expected_normalized == actual_normalized
        score = 1.0 if matched else 0.0
        
        return {
            "matched": matched,
            "method": "exact",
            "score": score,
            "details": {
                "expected": expected_normalized,
                "actual": actual_normalized
            }
        }
    
    def _edit_distance_match(self, expected_text: str, actual_text: str) -> Dict[str, Any]:
        """編集距離（Levenshtein距離）による比較"""
        try:
            import Levenshtein
        except ImportError:
            logger.warning("python-Levenshteinがインストールされていません。pip install python-Levenshteinでインストールしてください。")
            # フォールバック: 完全一致を使用
            return self._exact_match(expected_text, actual_text)
        
        expected_normalized = self._normalize_text(expected_text)
        actual_normalized = self._normalize_text(actual_text)
        
        # Levenshtein距離を計算
        distance = Levenshtein.distance(expected_normalized, actual_normalized)
        max_len = max(len(expected_normalized), len(actual_normalized))
        
        if max_len == 0:
            score = 1.0
        else:
            # 類似度スコア（0.0-1.0）を計算
            score = 1.0 - (distance / max_len)
        
        matched = score >= self.edit_distance_threshold
        
        return {
            "matched": matched,
            "method": "edit_distance",
            "score": score,
            "details": {
                "expected": expected_normalized,
                "actual": actual_normalized,
                "distance": distance,
                "max_length": max_len,
                "threshold": self.edit_distance_threshold
            }
        }
    
    def _llm_match(self, expected_text: str, actual_text: str) -> Dict[str, Any]:
        """LLMベースの比較（未実装）"""
        logger.warning("LLMベースの比較は未実装です。")
        return {
            "matched": False,
            "method": "llm",
            "score": 0.0,
            "details": {
                "error": "LLM matching not implemented"
            }
        }
    
    def _normalize_text(self, text: str) -> str:
        """
        テキストを正規化（比較用）
        - 空白を統一
        - 大文字小文字を統一（日本語の場合はそのまま）
        """
        # 空白を統一（連続する空白を1つに）
        import re
        normalized = re.sub(r'\s+', ' ', text.strip())
        return normalized


def create_text_matcher(config: Dict[str, Any]) -> TextMatcher:
    """設定に基づいてTextMatcherを作成"""
    return TextMatcher(config)

