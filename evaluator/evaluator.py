"""評価エンジン - シナリオ実行後の評価処理を担当"""

import logging
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from pathlib import Path

from evaluator.stt_engine import STTEngine, create_stt_engine
from tester.orchestrator import UnifiedLogger
from tester.vad_detector import VADDetector

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
            logger.debug(f"[Evaluator] STT engine initialized: {stt_config.get('engine', 'speechrecognition')}")
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
                logger.debug("[Evaluator] ユーザー音声を認識中...")
                user_text = self.stt_engine.transcribe(
                    user_audio.astype(np.float32) / 32767.0,
                    sample_rate
                )
                if user_text:
                    logger.debug(f"[Evaluator] ユーザー音声: {user_text}")
                    results["user_text"] = user_text
                    # タイムラインのUSER_SPEECH_STARTイベントにテキストを追加
                    for event in logger_instance.events:
                        if event.get("type") == "USER_SPEECH_START":
                            event["text"] = user_text
                
                logger.debug("[Evaluator] ボット音声を認識中...")
                # 各BOT_SPEECH_STARTイベントごとに個別のSTT結果を取得
                bot_speech_events = []
                for event in logger_instance.events:
                    if event.get("type") == "BOT_SPEECH_START":
                        bot_speech_events.append({"start": event, "end": None})
                    elif event.get("type") == "BOT_SPEECH_END" and bot_speech_events:
                        # 最後のBOT_SPEECH_STARTに対応するENDを見つける
                        for speech_event in reversed(bot_speech_events):
                            if speech_event["end"] is None:
                                speech_event["end"] = event
                                break
                
                bot_texts = []
                for i, speech_event in enumerate(bot_speech_events):
                    start_event = speech_event["start"]
                    end_event = speech_event["end"]
                    
                    # VAD検出時刻から音声セグメントを取得（VAD遅延を考慮済み）
                    vad_start_sample = start_event.get("vad_start_sample")
                    vad_end_sample = end_event.get("vad_end_sample") if end_event else None
                    
                    if vad_start_sample is not None:
                        # VAD検出時刻を使用（VAD遅延を考慮済み）
                        start_sample = vad_start_sample
                        if vad_end_sample is not None:
                            end_sample = vad_end_sample
                        else:
                            # ENDイベントがない場合は全体の最後まで
                            end_sample = len(bot_audio)
                        
                        # 範囲チェック
                        start_sample = max(0, min(start_sample, len(bot_audio)))
                        end_sample = max(start_sample, min(end_sample, len(bot_audio)))
                        
                        audio_segment = bot_audio[start_sample:end_sample]
                        
                        if len(audio_segment) > 0:
                            logger.debug(f"[Evaluator] ボット音声 {i+1}を認識中... (VAD期間: {start_sample/sample_rate:.2f}s - {end_sample/sample_rate:.2f}s)")
                            
                            # デバッグ用: 音声セグメントをWAVファイルとして保存
                            try:
                                import soundfile as sf
                                debug_output_dir = Path(logger_instance.output_dir) / "debug_segments"
                                debug_output_dir.mkdir(parents=True, exist_ok=True)
                                debug_file = debug_output_dir / f"{logger_instance.test_id}_bot_segment_{i+1}.wav"
                                # float32形式に変換（-1.0～1.0の範囲）
                                audio_segment_float = audio_segment.astype(np.float32) / 32767.0
                                sf.write(str(debug_file), audio_segment_float, sample_rate)
                                logger.debug(f"[Evaluator] デバッグ: 音声セグメント {i+1}を保存しました: {debug_file}")
                            except Exception as e:
                                logger.debug(f"[Evaluator] デバッグ音声ファイルの保存に失敗: {e}")
                            
                            bot_text = self.stt_engine.transcribe(
                                audio_segment.astype(np.float32) / 32767.0,
                                sample_rate
                            )
                            if bot_text:
                                logger.debug(f"[Evaluator] ボット音声 {i+1}: {bot_text}")
                                bot_texts.append(bot_text)
                                start_event["text"] = bot_text
                            else:
                                logger.debug(f"[Evaluator] ボット音声 {i+1}: STT結果なし")
                                bot_texts.append(None)
                        else:
                            logger.warning(f"[Evaluator] ボット音声 {i+1}: 音声セグメントが空")
                            bot_texts.append(None)
                    else:
                        # VAD検出時刻がない場合はタイムラインの時刻を使用（フォールバック）
                        logger.warning(f"[Evaluator] ボット音声 {i+1}: VAD検出時刻が記録されていません。タイムラインの時刻を使用します。")
                        start_time = start_event.get("time", 0) / 1000.0
                        start_sample = int(start_time * sample_rate)
                        if end_event:
                            end_time = end_event.get("time", 0) / 1000.0
                            end_sample = int(end_time * sample_rate)
                        else:
                            end_sample = len(bot_audio)
                        
                        start_sample = max(0, min(start_sample, len(bot_audio)))
                        end_sample = max(start_sample, min(end_sample, len(bot_audio)))
                        audio_segment = bot_audio[start_sample:end_sample]
                        
                        if len(audio_segment) > 0:
                            bot_text = self.stt_engine.transcribe(
                                audio_segment.astype(np.float32) / 32767.0,
                                sample_rate
                            )
                            if bot_text:
                                logger.debug(f"[Evaluator] ボット音声 {i+1}: {bot_text}")
                                bot_texts.append(bot_text)
                                start_event["text"] = bot_text
                            else:
                                bot_texts.append(None)
                        else:
                            bot_texts.append(None)
                
                # 後方互換性のため、最初のテキストを全体の結果としても設定
                if bot_texts and bot_texts[0]:
                    results["bot_text"] = bot_texts[0]
                results["bot_texts"] = bot_texts  # 各イベントごとのテキストリスト
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
        
        # Response Latency: USER_SPEECH_ENDからBOT_SPEECH_STARTまでの時間（すべてのペアの平均）
        user_speech_ends = [e for e in events if e.get("type") == "USER_SPEECH_END"]
        bot_speech_starts = [e for e in events if e.get("type") == "BOT_SPEECH_START"]
        
        response_latencies = []
        bot_idx = 0
        for user_end in user_speech_ends:
            user_end_time = user_end.get("time", 0)
            
            # Find corresponding BOT_SPEECH_START (first one after USER_SPEECH_END)
            for j in range(bot_idx, len(bot_speech_starts)):
                if bot_speech_starts[j].get("time", 0) > user_end_time:
                    bot_start_time = bot_speech_starts[j].get("time", 0)
                    response_latency = bot_start_time - user_end_time
                    response_latencies.append(response_latency)
                    bot_idx = j + 1
                    break
        
        if response_latencies:
            metrics["response_latency_ms"] = sum(response_latencies) / len(response_latencies)
            metrics["response_latency_ok"] = metrics["response_latency_ms"] <= eval_config.get("response_latency_threshold_ms", 800)
        else:
            metrics["response_latency_ms"] = None
            metrics["response_latency_ok"] = False
        
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

