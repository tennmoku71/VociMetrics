"""VAD (Voice Activity Detection) モジュール"""

import asyncio
import numpy as np
from typing import Optional, Callable
import logging

try:
    import torch
    import onnxruntime
except ImportError:
    torch = None
    onnxruntime = None

# silero-vadのインポート
load_silero_vad = None
get_speech_timestamps = None

try:
    from silero_vad import load_silero_vad, get_speech_timestamps
except ImportError:
    pass

logger = logging.getLogger(__name__)


class VADDetector:
    """音声活動検出（VAD）クラス
    
    silero-vadを使用してリアルタイムで音声の開始/終了を検出します。
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        on_speech_start: Optional[Callable] = None,
        on_speech_end: Optional[Callable] = None
    ):
        """VAD検出器を初期化
        
        Args:
            sample_rate: サンプリングレート（Hz）
            threshold: 音声検出の閾値（0.0-1.0）
            min_speech_duration_ms: 最小音声継続時間（ミリ秒）
            min_silence_duration_ms: 最小無音継続時間（ミリ秒）
            on_speech_start: 音声開始時のコールバック
            on_speech_end: 音声終了時のコールバック
        """
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        
        # 音声状態
        self.is_speaking = False
        self.speech_start_time: Optional[float] = None
        self.silence_start_time: Optional[float] = None
        
        # VADモデルの初期化
        self.model = None
        self._init_model()
        
    def _init_model(self):
        """VADモデルを初期化"""
        try:
            if load_silero_vad is None or get_speech_timestamps is None:
                logger.warning("[VADDetector] silero-vad not available. Using fallback VAD.")
                self.model = None
                self.get_speech_timestamps = None
                return
            
            # silero-vadモデルをロード
            try:
                self.model = load_silero_vad(onnx=False)
                self.get_speech_timestamps = get_speech_timestamps
                logger.info("[VADDetector] silero-vad model loaded successfully")
            except Exception as e:
                logger.warning(f"[VADDetector] Failed to load silero-vad model: {e}")
                self.model = None
                self.get_speech_timestamps = None
                
        except Exception as e:
            logger.error(f"[VADDetector] Failed to initialize VAD model: {e}", exc_info=True)
            self.model = None
            self.get_speech_timestamps = None
    
    def process_audio_frame(self, audio_data: np.ndarray, timestamp: float) -> bool:
        """音声フレームを処理して音声活動を検出
        
        Args:
            audio_data: 音声データ（numpy配列、16bit PCM）
            timestamp: タイムスタンプ（秒）
            
        Returns:
            音声が検出された場合True
        """
        if self.model is None or self.get_speech_timestamps is None:
            # フォールバック: 簡易的な音量ベースの検出
            return self._fallback_detection(audio_data, timestamp)
        
        try:
            # 音声データを正規化（-1.0 ～ 1.0）
            if audio_data.dtype == np.int16:
                audio_normalized = audio_data.astype(np.float32) / 32768.0
            else:
                audio_normalized = audio_data.astype(np.float32)
            
            # VADモデルで検出
            # silero-vadは通常、長い音声チャンクを処理するため、
            # 短いフレームの場合はバッファリングが必要
            # ここでは簡易的にフレームごとに処理
            speech_timestamps = self.get_speech_timestamps(
                audio_normalized,
                self.model,
                sampling_rate=self.sample_rate,
                threshold=self.threshold,
                min_speech_duration_ms=self.min_speech_duration_ms,
                min_silence_duration_ms=self.min_silence_duration_ms
            )
            
            # 音声が検出されたかチェック
            has_speech = len(speech_timestamps) > 0
            
            # 状態遷移の処理
            if has_speech and not self.is_speaking:
                # 音声開始
                self.is_speaking = True
                self.speech_start_time = timestamp
                self.silence_start_time = None
                if self.on_speech_start:
                    self.on_speech_start(timestamp)
                logger.debug(f"[VADDetector] Speech started at {timestamp:.3f}s")
                
            elif not has_speech and self.is_speaking:
                # 無音が続いているかチェック
                if self.silence_start_time is None:
                    self.silence_start_time = timestamp
                else:
                    silence_duration = (timestamp - self.silence_start_time) * 1000
                    if silence_duration >= self.min_silence_duration_ms:
                        # 音声終了
                        self.is_speaking = False
                        speech_end_time = self.silence_start_time
                        self.speech_start_time = None
                        self.silence_start_time = None
                        if self.on_speech_end:
                            self.on_speech_end(speech_end_time)
                        logger.debug(f"[VADDetector] Speech ended at {speech_end_time:.3f}s")
            elif has_speech and self.is_speaking:
                # 音声継続中
                self.silence_start_time = None
                
            return has_speech
            
        except Exception as e:
            logger.error(f"[VADDetector] Error processing audio frame: {e}", exc_info=True)
            return False
    
    def _fallback_detection(self, audio_data: np.ndarray, timestamp: float) -> bool:
        """フォールバック検出（音量ベース）"""
        # RMS（Root Mean Square）で音量を計算
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float32) / 32768.0
        else:
            audio_float = audio_data.astype(np.float32)
            
        rms = np.sqrt(np.mean(audio_float ** 2))
        
        # 閾値ベースの検出（簡易版）
        has_speech = rms > 0.01  # 適宜調整が必要
        
        # 状態遷移の処理
        if has_speech and not self.is_speaking:
            self.is_speaking = True
            self.speech_start_time = timestamp
            if self.on_speech_start:
                self.on_speech_start(timestamp)
        elif not has_speech and self.is_speaking:
            if self.silence_start_time is None:
                self.silence_start_time = timestamp
            else:
                silence_duration = (timestamp - self.silence_start_time) * 1000
                if silence_duration >= self.min_silence_duration_ms:
                    self.is_speaking = False
                    speech_end_time = self.silence_start_time
                    self.speech_start_time = None
                    self.silence_start_time = None
                    if self.on_speech_end:
                        self.on_speech_end(speech_end_time)
        elif has_speech and self.is_speaking:
            self.silence_start_time = None
            
        return has_speech
    
    def reset(self):
        """VAD状態をリセット"""
        self.is_speaking = False
        self.speech_start_time = None
        self.silence_start_time = None

