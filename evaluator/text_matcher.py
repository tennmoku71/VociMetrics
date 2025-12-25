"""テキスト比較エンジン（STT結果と期待テキストの比較）"""

import json
import logging
import os
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class MatchMethod(Enum):
    """比較方法"""
    EXACT = "exact"  # 完全一致
    EDIT_DISTANCE = "edit_distance"  # 編集距離（Levenshtein距離）
    LLM = "llm"  # LLMベースの比較（OpenAI API使用）


class TextMatcher:
    """テキスト比較エンジン"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.match_method = MatchMethod(config.get("match_method", "exact"))
        self.edit_distance_threshold = config.get("edit_distance_threshold", 0.8)  # 0.0-1.0（類似度）
        
        # LLM設定
        llm_config = config.get("llm", {})
        self.llm_model = llm_config.get("model", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        self.llm_temperature = llm_config.get("temperature", 0.0)
        self.llm_similarity_threshold = llm_config.get("similarity_threshold", 0.7)
        self.llm_system_prompt = llm_config.get(
            "system_prompt",
            "あなたは対話システムの応答品質を評価する専門家です。期待される応答例と実際の応答を比較し、実際の応答が期待される応答として自然で適切かを0.0から1.0のスコアで評価してください。"
        )
        self.llm_user_prompt_template = llm_config.get(
            "user_prompt_template",
            "以下の期待される応答例と実際の応答を比較し、実際の応答が期待される応答として自然で適切かを評価してください。\n\n期待される応答例: {expected_text}\n実際の応答: {actual_text}\n\nJSON形式でスコアを返してください。"
        )
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_api_base = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
    
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
        """LLMベースの比較（意味的な類似度を評価）"""
        if not self.openai_api_key:
            logger.error("OPENAI_API_KEYが設定されていません。.envファイルにOPENAI_API_KEYを設定してください。")
            return {
                "matched": False,
                "method": "llm",
                "score": 0.0,
                "details": {
                    "error": "OPENAI_API_KEY not set"
                }
            }
        
        try:
            from openai import OpenAI
            import httpx
            
            # OpenAI APIのHTTPリクエストログを抑制
            httpx_logger = logging.getLogger("httpx")
            httpx_logger.setLevel(logging.WARNING)
            
            client = OpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_api_base,
                http_client=httpx.Client(timeout=30.0)
            )
            
            # プロンプトを作成（テンプレートから）
            user_prompt = self.llm_user_prompt_template.format(
                expected_text=expected_text,
                actual_text=actual_text
            )
            
            # JSONスキーマを定義（構造化出力用）
            json_schema = {
                "name": "response_quality_score",
                "schema": {
                    "type": "object",
                    "properties": {
                        "score": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "応答としての自然さ・適切さスコア（0.0-1.0）"
                        }
                    },
                    "required": ["score"],
                    "additionalProperties": False
                }
            }
            
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": self.llm_system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.llm_temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": json_schema
                }
            )
            
            # レスポンスからスコアを抽出
            response_content = response.choices[0].message.content.strip()
            try:
                result = json.loads(response_content)
                score = float(result.get("score", 0.0))
                # スコアを0.0-1.0の範囲にクランプ（念のため）
                score = max(0.0, min(1.0, score))
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"LLMが無効なJSONを返しました: {response_content}。エラー: {e}。0.0として扱います。")
                score = 0.0
            
            matched = score >= self.llm_similarity_threshold
            
            return {
                "matched": matched,
                "method": "llm",
                "score": score,
                "details": {
                    "expected": expected_text,
                    "actual": actual_text,
                    "threshold": self.llm_similarity_threshold,
                    "model": self.llm_model,
                    "raw_response": response_content
                }
            }
            
        except ImportError:
            logger.error("openaiライブラリがインストールされていません。pip install openaiでインストールしてください。")
            return {
                "matched": False,
                "method": "llm",
                "score": 0.0,
                "details": {
                    "error": "openai library not installed"
                }
            }
        except Exception as e:
            logger.error(f"LLM比較中にエラーが発生しました: {e}")
            return {
                "matched": False,
                "method": "llm",
                "score": 0.0,
                "details": {
                    "error": str(e)
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

