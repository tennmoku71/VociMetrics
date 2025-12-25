"""評価エンジン - シナリオ実行後の評価処理を担当"""

import logging
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from pathlib import Path

from evaluator.stt_engine import STTEngine, create_stt_engine
from evaluator.sound_evaluator import SoundEvaluator
from evaluator.llm_evaluator import create_llm_conversation_evaluator
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
        
        # Sound評価エンジンを初期化
        try:
            self.sound_evaluator = SoundEvaluator(config)
            logger.debug("[Evaluator] Sound evaluator initialized")
        except Exception as e:
            logger.warning(f"[Evaluator] Failed to initialize sound evaluator: {e}. Sound評価機能は無効です。")
            self.sound_evaluator = None
        
        # LLM対話評価エンジンを初期化
        try:
            self.llm_conversation_evaluator = create_llm_conversation_evaluator(config)
            logger.debug("[Evaluator] LLM conversation evaluator initialized")
        except Exception as e:
            logger.warning(f"[Evaluator] Failed to initialize LLM conversation evaluator: {e}. LLM対話評価機能は無効です。")
            self.llm_conversation_evaluator = None
    
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
        
        # Sound評価を実行（ボット音声に対して）
        if self.sound_evaluator and len(bot_audio) > 0:
            try:
                # VAD ON/OFF区間を取得
                vad_on_samples = []
                vad_off_samples = []
                
                # BOT_SPEECH_START/ENDイベントからVAD ON区間を取得
                events = logger_instance.events
                bot_speech_starts = [e for e in events if e.get("type") == "BOT_SPEECH_START"]
                bot_speech_ends = [e for e in events if e.get("type") == "BOT_SPEECH_END"]
                
                # VAD ON区間を構築
                for i, start_event in enumerate(bot_speech_starts):
                    vad_start_sample = start_event.get("vad_start_sample")
                    if vad_start_sample is not None:
                        # 対応するENDイベントを探す
                        vad_end_sample = None
                        for j, end_event in enumerate(bot_speech_ends):
                            if j >= i and end_event.get("vad_end_sample") is not None:
                                vad_end_sample = end_event.get("vad_end_sample")
                                break
                        
                        if vad_end_sample is None:
                            vad_end_sample = len(bot_audio)
                        
                        vad_on_samples.append((vad_start_sample, vad_end_sample))
                
                # VAD OFF区間を構築（ON区間の間と前後）
                if len(vad_on_samples) > 0:
                    # 最初のON区間より前
                    if vad_on_samples[0][0] > 0:
                        vad_off_samples.append((0, vad_on_samples[0][0]))
                    
                    # ON区間の間
                    for i in range(len(vad_on_samples) - 1):
                        off_start = vad_on_samples[i][1]
                        off_end = vad_on_samples[i + 1][0]
                        if off_end > off_start:
                            vad_off_samples.append((off_start, off_end))
                    
                    # 最後のON区間より後
                    if vad_on_samples[-1][1] < len(bot_audio):
                        vad_off_samples.append((vad_on_samples[-1][1], len(bot_audio)))
                else:
                    # ON区間がない場合は全体がOFF
                    vad_off_samples.append((0, len(bot_audio)))
                
                # STT結果を準備（sound評価用）
                stt_results_for_sound = []
                for start_event in bot_speech_starts:
                    text = start_event.get("text")
                    if text:
                        vad_start_sample = start_event.get("vad_start_sample")
                        # 対応するENDイベントを探す
                        vad_end_sample = None
                        for end_event in bot_speech_ends:
                            if end_event.get("vad_end_sample") is not None:
                                vad_end_sample = end_event.get("vad_end_sample")
                                break
                        
                        if vad_start_sample is not None:
                            start_time = vad_start_sample / sample_rate
                            end_time = (vad_end_sample / sample_rate) if vad_end_sample else None
                            stt_results_for_sound.append({
                                "text": text,
                                "start_time": start_time,
                                "end_time": end_time
                            })
                
                # Sound評価を実行
                sound_results = self.sound_evaluator.evaluate_sound(
                    bot_audio=bot_audio,
                    sample_rate=sample_rate,
                    vad_on_samples=vad_on_samples,
                    vad_off_samples=vad_off_samples,
                    stt_results=stt_results_for_sound
                )
                results["sound_metrics"] = sound_results
                logger.debug(f"[Evaluator] Sound評価完了: SNR={sound_results.get('snr', {}).get('snr_db')}dB, Score={sound_results.get('score'):.1f}")
            except Exception as e:
                logger.error(f"[Evaluator] Sound評価エラー: {e}", exc_info=True)
                results["sound_metrics"] = None
        
        return results
    
    def calculate_metrics(
        self,
        logger_instance: UnifiedLogger,
        orchestrator_instance: Optional[Any] = None
    ) -> Dict[str, Any]:
        """評価指標を計算
        
        Args:
            logger_instance: UnifiedLoggerインスタンス
            orchestrator_instance: Orchestratorインスタンス（toolcall評価用）
            
        Returns:
            評価指標の辞書
        """
        metrics = {
            "response_latency_ms": None,
            "think_time_ms": None,
            "user_speech_duration_ms": None,
            "bot_speech_duration_ms": None,
            "interrupt_to_speech_end_ms": None,
            "toolcall_metrics": None
        }
        
        events = logger_instance.events
        eval_config = self.config.get("evaluation", {})
        
        # Toolcall評価用の設定を取得
        toolcall_latency_threshold_ms = eval_config.get("toolcall_latency_threshold_ms", 2000)
        
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
        
        # Interrupt to Speech End: USER_INTERRUPT_STARTからBOT_SPEECH_ENDまでの時間
        interrupt_start_times = [e.get("time", 0) for e in events if e.get("type") == "USER_INTERRUPT_START"]
        bot_speech_ends = [e.get("time", 0) for e in events if e.get("type") == "BOT_SPEECH_END"]
        
        interrupt_to_end_latencies = []
        bot_end_idx = 0
        for interrupt_time in interrupt_start_times:
            # 割り込みの後に来る最初のBOT_SPEECH_ENDを探す
            for j in range(bot_end_idx, len(bot_speech_ends)):
                if bot_speech_ends[j] > interrupt_time:
                    latency = bot_speech_ends[j] - interrupt_time
                    interrupt_to_end_latencies.append(latency)
                    bot_end_idx = j + 1
                    break
        
        if interrupt_to_end_latencies:
            metrics["interrupt_to_speech_end_ms"] = sum(interrupt_to_end_latencies) / len(interrupt_to_end_latencies)
        
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
        
        # Toolcall評価
        if orchestrator_instance and hasattr(orchestrator_instance, 'expected_toolcalls'):
            metrics["toolcall_metrics"] = self._evaluate_toolcalls(
                events=events,
                expected_toolcalls=orchestrator_instance.expected_toolcalls,
                eval_config=eval_config
            )
        
        return metrics
    
    def _evaluate_toolcalls(
        self,
        events: List[Dict[str, Any]],
        expected_toolcalls: List[Dict[str, Any]],
        eval_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Toolcallを評価
        
        評価軸:
        1. そもそも出るべきか、出ないべきか（期待されるtoolcallが送信されたか）
        2. 回数は適切か（期待回数と実際の回数が一致するか）
        3. argumentsは適切か（引数が一致するか）
        
        Args:
            events: イベントリスト
            expected_toolcalls: 期待されるtoolcallリスト
            
        Returns:
            toolcall評価結果の辞書
        """
        # 実際に受信したtoolcallを取得
        actual_toolcalls = [e for e in events if e.get("type") == "TOOLCALL"]
        
        # 評価結果
        toolcall_results = []
        total_score = 0.0
        max_score = 0.0
        
        # 使用済みのtoolcallを追跡（各期待toolcallに対して1回だけ使用）
        used_toolcall_indices = set()
        
        # 各期待toolcallを評価
        for i, expected in enumerate(expected_toolcalls):
            expected_name = expected.get("name")
            expected_args = expected.get("arguments", {})
            expected_count = 1  # 現時点では1回を期待
            
            # 対応する実際のtoolcallを探す（同じ名前で、まだ使用されていないもの）
            matching_toolcall = None
            matching_index = None
            for j, tc in enumerate(actual_toolcalls):
                if j in used_toolcall_indices:
                    continue
                if tc.get("name") == expected_name:
                    matching_toolcall = tc
                    matching_index = j
                    break
            
            # 同じ名前のtoolcallが何回送信されたかカウント（使用済みを除く）
            matching_count = sum(
                1 for j, tc in enumerate(actual_toolcalls)
                if j not in used_toolcall_indices and tc.get("name") == expected_name
            )
            
            result = {
                "index": i,
                "expected_name": expected_name,
                "expected_arguments": expected_args,
                "expected_count": expected_count,
                "found": matching_toolcall is not None,
                "actual_count": matching_count,
                "count_match": matching_count == expected_count,
                "latency_ms": None,
                "arguments_match": False,
                "score": 0.0
            }
            
            if matching_toolcall:
                # 使用済みとしてマーク
                used_toolcall_indices.add(matching_index)
                
                actual_args = matching_toolcall.get("arguments", {})
                
                # 引数の一致を確認
                result["arguments_match"] = self._compare_toolcall_arguments(
                    expected_args, actual_args
                )
                
                # 直前のBOT_SPEECH_ENDからの遅延を計算
                bot_speech_ends = [
                    e for e in events
                    if e.get("type") == "BOT_SPEECH_END"
                    and e.get("time", 0) < matching_toolcall.get("time", 0)
                ]
                if bot_speech_ends:
                    # 最も近いBOT_SPEECH_ENDを使用
                    last_bot_end = max(bot_speech_ends, key=lambda e: e.get("time", 0))
                    result["latency_ms"] = matching_toolcall.get("time", 0) - last_bot_end.get("time", 0)
                
                # スコア計算（各項目の重み付き）
                score = 0.0
                # 1. そもそも出るべきか、出ないべきか（送信されたか）: 30%
                if result["found"]:
                    score += 0.3
                # 2. 回数は適切か（期待回数と実際の回数が一致するか）: 25%
                if result["count_match"]:
                    score += 0.25
                # 3. argumentsは適切か（引数が一致するか）: 25%
                if result["arguments_match"]:
                    score += 0.25
                # 4. latencyは適切か（直前の発話からの遅延が適切か）: 20%
                if result["latency_ms"] is not None:
                    # 遅延が2秒以内なら満点、それ以上は線形に減点（5秒で0点）
                    if eval_config is None:
                        eval_config = {}
                    latency_threshold_ms = eval_config.get("toolcall_latency_threshold_ms", 2000)
                    latency_max_ms = latency_threshold_ms * 2.5  # 5秒
                    if result["latency_ms"] <= latency_threshold_ms:
                        score += 0.2  # 2秒以内: 満点
                    elif result["latency_ms"] <= latency_max_ms:
                        # 2秒超5秒以内: 線形に減点
                        penalty = 0.2 * ((result["latency_ms"] - latency_threshold_ms) / (latency_max_ms - latency_threshold_ms))
                        score += max(0.0, 0.2 - penalty)
                    # 5秒超: 0点
                
                result["score"] = score
            else:
                # toolcallが見つからない場合（送信されなかった）
                result["score"] = 0.0
            
            toolcall_results.append(result)
            total_score += result["score"]
            max_score += 1.0
        
        # 余分なtoolcall（期待されていないもの）をカウント
        unexpected_toolcalls = len(actual_toolcalls) - len(used_toolcall_indices)
        
        # 全体のスコア（余分なtoolcallがある場合は減点）
        if max_score > 0:
            base_score = (total_score / max_score * 100.0)
            # 余分なtoolcallがある場合、1つにつき10%減点（最小0%）
            penalty = min(unexpected_toolcalls * 10.0, base_score)
            overall_score = max(0.0, base_score - penalty)
        else:
            overall_score = 0.0
        
        return {
            "expected_count": len(expected_toolcalls),
            "actual_count": len(actual_toolcalls),
            "unexpected_count": unexpected_toolcalls,
            "results": toolcall_results,
            "overall_score": overall_score
        }
    
    def _compare_toolcall_arguments(
        self,
        expected: Dict[str, Any],
        actual: Dict[str, Any]
    ) -> bool:
        """Toolcall引数を比較
        
        Args:
            expected: 期待される引数
            actual: 実際の引数
            
        Returns:
            一致するかどうか
        """
        # 期待される引数のすべてのキーが実際の引数に存在し、値が一致するか確認
        for key, expected_value in expected.items():
            if key not in actual:
                return False
            if actual[key] != expected_value:
                return False
        return True
    
    def evaluate_conversation_quality(
        self,
        logger_instance: UnifiedLogger,
        user_texts: List[Optional[str]],
        bot_texts: List[Optional[str]]
    ) -> Dict[str, Any]:
        """LLMによる対話全体の品質評価
        
        Args:
            logger_instance: UnifiedLoggerインスタンス（イベントリストを取得）
            user_texts: ユーザー発話のテキストリスト（STT結果）
            bot_texts: ボット発話のテキストリスト（STT結果）
            
        Returns:
            評価結果の辞書:
            {
                "backchannel_score": float,  # 0.0-1.0
                "tone_consistency_score": float,  # 0.0-1.0
                "omotenashi_score": int,  # 1-5
                "error": Optional[str]
            }
        """
        if not self.llm_conversation_evaluator:
            return {
                "backchannel_score": None,
                "tone_consistency_score": None,
                "omotenashi_score": None,
                "error": "LLM conversation evaluator not initialized"
            }
        
        try:
            events = logger_instance.events
            result = self.llm_conversation_evaluator.evaluate_conversation(
                events=events,
                user_texts=user_texts,
                bot_texts=bot_texts
            )
            return result
        except Exception as e:
            logger.error(f"[Evaluator] LLM対話評価エラー: {e}", exc_info=True)
            return {
                "backchannel_score": None,
                "tone_consistency_score": None,
                "omotenashi_score": None,
                "error": str(e)
            }

