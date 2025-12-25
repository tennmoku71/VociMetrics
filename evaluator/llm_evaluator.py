"""LLMによる対話全体の評価エンジン"""

import json
import logging
import os
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class LLMConversationEvaluator:
    """LLMによる対話全体の評価エンジン
    
    評価項目:
    - 相槌・フィラーの適切さ（0.0-1.0）
    - トーンの一貫性（0.0-1.0）
    - おもてなしスコア（1-5）
    """
    
    def __init__(self, config: Dict[str, Any]):
        """LLM評価エンジンを初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        
        # LLM設定
        llm_config = config.get("conversation_llm", {})
        self.llm_model = llm_config.get("model", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        self.llm_temperature = llm_config.get("temperature", 0.0)
        self.llm_system_prompt = llm_config.get(
            "system_prompt",
            "あなたは対話システムのUX品質を評価する専門家です。一連の対話を分析し、以下の観点から評価してください。\n\n"
            "1. 相槌・フィラーの適切さ: 文脈を無視した不自然な相槌や、不要な発話停止（ノイズへの過剰反応）がないかを評価します。\n"
            "2. トーンの一貫性: 割り込み前後やAPI実行前後で、声のトーンやキャラ設定が崩れていないかを評価します。\n"
            "3. おもてなしスコア: 総合的な「対話の心地よさ」を5段階で評価します。"
        )
        self.llm_user_prompt_template = llm_config.get(
            "user_prompt_template",
            "以下の対話ログを分析し、評価してください。\n\n"
            "【対話ログ】\n{conversation_log}\n\n"
            "JSON形式で以下の評価を返してください：\n"
            "- backchannel_score: 相槌・フィラーの適切さ（0.0-1.0）\n"
            "- tone_consistency_score: トーンの一貫性（0.0-1.0）\n"
            "- omotenashi_score: おもてなしスコア（1-5の整数）"
        )
        # API Keyの優先順位: 環境変数 > config.json
        openai_config = config.get("openai", {})
        self.openai_api_key = os.getenv("OPENAI_API_KEY") or openai_config.get("api_key", "")
        self.openai_api_base = os.getenv("OPENAI_API_BASE_URL") or openai_config.get("api_base_url", "https://api.openai.com/v1")
    
    def evaluate_conversation(
        self,
        events: List[Dict[str, Any]],
        user_texts: List[Optional[str]],
        bot_texts: List[Optional[str]]
    ) -> Dict[str, Any]:
        """対話全体を評価
        
        Args:
            events: イベントリスト（タイムライン）
            user_texts: ユーザー発話のテキストリスト（STT結果）
            bot_texts: ボット発話のテキストリスト（STT結果）
            
        Returns:
            評価結果の辞書:
            {
                "backchannel_score": float,  # 0.0-1.0
                "tone_consistency_score": float,  # 0.0-1.0
                "omotenashi_score": int,  # 1-5
                "error": Optional[str]  # エラーメッセージ（エラー時）
            }
        """
        if not self.openai_api_key:
            logger.error("OPENAI_API_KEYが設定されていません。.envファイルにOPENAI_API_KEYを設定してください。")
            return {
                "backchannel_score": 0.0,
                "tone_consistency_score": 0.0,
                "omotenashi_score": 1,
                "error": "OPENAI_API_KEY not set"
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
            
            # 対話ログを構築
            conversation_log = self._build_conversation_log(events, user_texts, bot_texts)
            
            # プロンプトを作成
            user_prompt = self.llm_user_prompt_template.format(
                conversation_log=conversation_log
            )
            
            # JSONスキーマを定義（構造化出力用）
            json_schema = {
                "name": "conversation_quality_scores",
                "schema": {
                    "type": "object",
                    "properties": {
                        "backchannel_score": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "相槌・フィラーの適切さスコア（0.0-1.0）"
                        },
                        "tone_consistency_score": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "トーンの一貫性スコア（0.0-1.0）"
                        },
                        "omotenashi_score": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 5,
                            "description": "おもてなしスコア（1-5の整数）"
                        }
                    },
                    "required": ["backchannel_score", "tone_consistency_score", "omotenashi_score"],
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
                backchannel_score = float(result.get("backchannel_score", 0.0))
                tone_consistency_score = float(result.get("tone_consistency_score", 0.0))
                omotenashi_score = int(result.get("omotenashi_score", 1))
                
                # スコアを範囲内にクランプ（念のため）
                backchannel_score = max(0.0, min(1.0, backchannel_score))
                tone_consistency_score = max(0.0, min(1.0, tone_consistency_score))
                omotenashi_score = max(1, min(5, omotenashi_score))
                
                return {
                    "backchannel_score": backchannel_score,
                    "tone_consistency_score": tone_consistency_score,
                    "omotenashi_score": omotenashi_score,
                    "error": None
                }
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"LLMが無効なJSONを返しました: {response_content}。エラー: {e}。デフォルト値を使用します。")
                return {
                    "backchannel_score": 0.0,
                    "tone_consistency_score": 0.0,
                    "omotenashi_score": 1,
                    "error": f"Invalid JSON: {e}"
                }
            
        except ImportError:
            logger.error("openaiライブラリがインストールされていません。pip install openaiでインストールしてください。")
            return {
                "backchannel_score": 0.0,
                "tone_consistency_score": 0.0,
                "omotenashi_score": 1,
                "error": "openai library not installed"
            }
        except Exception as e:
            logger.error(f"LLM評価中にエラーが発生しました: {e}")
            return {
                "backchannel_score": 0.0,
                "tone_consistency_score": 0.0,
                "omotenashi_score": 1,
                "error": str(e)
            }
    
    def _build_conversation_log(
        self,
        events: List[Dict[str, Any]],
        user_texts: List[Optional[str]],
        bot_texts: List[Optional[str]]
    ) -> str:
        """対話ログを構築
        
        Args:
            events: イベントリスト
            user_texts: ユーザー発話のテキストリスト
            bot_texts: ボット発話のテキストリスト
            
        Returns:
            対話ログの文字列
        """
        log_lines = []
        
        # イベントを時系列順に並べ替え
        sorted_events = sorted(events, key=lambda e: e.get("time", 0))
        
        user_text_idx = 0
        bot_text_idx = 0
        
        for event in sorted_events:
            event_type = event.get("type")
            event_time = event.get("time", 0)
            time_sec = event_time / 1000.0
            
            if event_type == "USER_SPEECH_START":
                text = event.get("text")
                if not text and user_text_idx < len(user_texts):
                    text = user_texts[user_text_idx]
                    user_text_idx += 1
                if text:
                    log_lines.append(f"[{time_sec:.2f}s] USER: {text}")
                else:
                    log_lines.append(f"[{time_sec:.2f}s] USER: (speech started)")
            
            elif event_type == "USER_SPEECH_END":
                log_lines.append(f"[{time_sec:.2f}s] USER: (speech ended)")
            
            elif event_type == "USER_INTERRUPT_START":
                text = event.get("text")
                if text:
                    log_lines.append(f"[{time_sec:.2f}s] USER (INTERRUPT): {text}")
                else:
                    log_lines.append(f"[{time_sec:.2f}s] USER (INTERRUPT): (interrupt started)")
            
            elif event_type == "BOT_SPEECH_START":
                text = event.get("text")
                if not text and bot_text_idx < len(bot_texts):
                    text = bot_texts[bot_text_idx]
                    bot_text_idx += 1
                if text:
                    log_lines.append(f"[{time_sec:.2f}s] BOT: {text}")
                else:
                    log_lines.append(f"[{time_sec:.2f}s] BOT: (speech started)")
            
            elif event_type == "BOT_SPEECH_END":
                log_lines.append(f"[{time_sec:.2f}s] BOT: (speech ended)")
            
            elif event_type == "TOOLCALL":
                name = event.get("name", "unknown")
                arguments = event.get("arguments", {})
                log_lines.append(f"[{time_sec:.2f}s] TOOLCALL: {name}({json.dumps(arguments, ensure_ascii=False)})")
        
        return "\n".join(log_lines)


def create_llm_conversation_evaluator(config: Dict[str, Any]) -> LLMConversationEvaluator:
    """設定に基づいてLLMConversationEvaluatorを作成"""
    return LLMConversationEvaluator(config)

