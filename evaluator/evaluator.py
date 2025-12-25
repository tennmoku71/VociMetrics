"""評価エンジン - シナリオ実行後の評価処理を担当"""

import logging
from typing import Optional, Dict, Any
import numpy as np
from pathlib import Path

from evaluator.stt_engine import STTEngine, create_stt_engine
from tester.orchestrator import UnifiedLogger

logger = logging.getLogger(__name__)


class Evaluator:
    """評価エンジン - STT処理と評価指標の計算"""
    
    def __init__(self, config: Dict[str, Any]):
        """評価エンジンを初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        
        # STTエンジンを初期化
        stt_config = config.get("stt", {})
        self.stt_engine: Optional[STTEngine] = None
        try:
            self.stt_engine = create_stt_engine(
                engine_type=stt_config.get("engine", "speechrecognition"),
                language=stt_config.get("language", "ja-JP"),
                model_path=stt_config.get("vosk_model_path")
            )
            logger.info(f"[Evaluator] STT engine initialized: {stt_config.get('engine', 'speechrecognition')}")
        except Exception as e:
            logger.warning(f"[Evaluator] Failed to initialize STT engine: {e}. STT機能は無効です。")
    
    def evaluate_recording(
        self,
        user_audio: np.ndarray,
        bot_audio: np.ndarray,
        sample_rate: int,
        logger_instance: UnifiedLogger
    ) -> Dict[str, Any]:
        """録音音声を評価
        
        Args:
            user_audio: ユーザー音声データ（int16）
            bot_audio: ボット音声データ（int16）
            sample_rate: サンプリングレート（Hz）
            logger_instance: UnifiedLoggerインスタンス（タイムラインを更新）
            
        Returns:
            評価結果の辞書
        """
        results = {
            "user_text": None,
            "bot_text": None,
            "stt_enabled": self.stt_engine is not None
        }
        
        # STTで音声を認識
        if self.stt_engine:
            try:
                logger.info("[Evaluator] ユーザー音声を認識中...")
                user_text = self.stt_engine.transcribe(
                    user_audio.astype(np.float32) / 32767.0,
                    sample_rate
                )
                if user_text:
                    logger.info(f"[Evaluator] ユーザー音声: {user_text}")
                    results["user_text"] = user_text
                    # タイムラインのUSER_SPEECH_STARTイベントにテキストを追加
                    for event in logger_instance.events:
                        if event.get("type") == "USER_SPEECH_START":
                            event["text"] = user_text
                
                logger.info("[Evaluator] ボット音声を認識中...")
                bot_text = self.stt_engine.transcribe(
                    bot_audio.astype(np.float32) / 32767.0,
                    sample_rate
                )
                if bot_text:
                    logger.info(f"[Evaluator] ボット音声: {bot_text}")
                    results["bot_text"] = bot_text
                    # タイムラインのBOT_SPEECH_STARTイベントにテキストを追加
                    for event in logger_instance.events:
                        if event.get("type") == "BOT_SPEECH_START":
                            event["text"] = bot_text
            except Exception as e:
                logger.error(f"[Evaluator] 音声認識エラー: {e}", exc_info=True)
        
        return results
    
    def calculate_metrics(self, logger_instance: UnifiedLogger) -> Dict[str, Any]:
        """評価指標を計算
        
        Args:
            logger_instance: UnifiedLoggerインスタンス
            
        Returns:
            評価指標の辞書
        """
        metrics = {
            "response_latency_ms": None,
            "think_time_ms": None,
            "user_speech_duration_ms": None,
            "bot_speech_duration_ms": None
        }
        
        events = logger_instance.events
        eval_config = self.config.get("evaluation", {})
        
        # Response Latency: USER_SPEECH_ENDからBOT_SPEECH_STARTまでの時間
        user_speech_end_time = None
        bot_speech_start_time = None
        
        for event in events:
            if event.get("type") == "USER_SPEECH_END":
                user_speech_end_time = event.get("time")
            elif event.get("type") == "BOT_SPEECH_START":
                bot_speech_start_time = event.get("time")
                if user_speech_end_time is not None:
                    metrics["response_latency_ms"] = bot_speech_start_time - user_speech_end_time
                    break
        
        # Think Time: API_CALL_ENDからBOT_SPEECH_STARTまでの時間
        api_call_end_time = None
        for event in events:
            if event.get("type") == "API_CALL_END":
                api_call_end_time = event.get("time")
            elif event.get("type") == "BOT_SPEECH_START":
                if api_call_end_time is not None:
                    metrics["think_time_ms"] = event.get("time") - api_call_end_time
                    break
        
        # User Speech Duration: USER_SPEECH_STARTからUSER_SPEECH_ENDまでの時間
        user_speech_start_time = None
        for event in events:
            if event.get("type") == "USER_SPEECH_START":
                user_speech_start_time = event.get("time")
            elif event.get("type") == "USER_SPEECH_END":
                if user_speech_start_time is not None:
                    metrics["user_speech_duration_ms"] = event.get("time") - user_speech_start_time
                    break
        
        # Bot Speech Duration: BOT_SPEECH_STARTからBOT_SPEECH_ENDまでの時間
        bot_speech_start_time = None
        for event in events:
            if event.get("type") == "BOT_SPEECH_START":
                bot_speech_start_time = event.get("time")
            elif event.get("type") == "BOT_SPEECH_END":
                if bot_speech_start_time is not None:
                    metrics["bot_speech_duration_ms"] = event.get("time") - bot_speech_start_time
                    break
        
        # 閾値との比較
        response_latency_threshold = eval_config.get("response_latency_threshold_ms", 800)
        think_time_threshold = eval_config.get("think_time_threshold_ms", 500)
        
        metrics["response_latency_ok"] = (
            metrics["response_latency_ms"] is not None and
            metrics["response_latency_ms"] <= response_latency_threshold
        )
        metrics["think_time_ok"] = (
            metrics["think_time_ms"] is not None and
            metrics["think_time_ms"] <= think_time_threshold
        )
        
        return metrics

