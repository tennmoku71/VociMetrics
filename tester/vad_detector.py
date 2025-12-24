"""VAD (Voice Activity Detection) モジュール"""

import asyncio
import numpy as np
from typing import Optional, Callable
import logging

# webrtcvadのインポート
try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    WEBRTCVAD_AVAILABLE = False
    webrtcvad = None

logger = logging.getLogger(__name__)


class VADDetector:
    """音声活動検出（VAD）クラス
    
    webrtcvadを使用してリアルタイムで音声の開始/終了を検出します。
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
            sample_rate: サンプリングレート（Hz）。8000, 16000, 32000, 48000をサポート
            threshold: 音声検出の閾値（0.0-1.0）。webrtcvadのaggressiveness（0-3）に変換
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
        
        # webrtcvadの初期化
        self.vad = None
        # webrtcvadは10ms、20ms、30msのフレーム長のみをサポート
        # 20msフレームを使用（10msより安定性が高い）
        self.frame_duration_ms = 20  # 20msフレーム
        self.frame_size = int(sample_rate * self.frame_duration_ms / 1000)  # サンプル数
        self._init_model()
    
    def _init_model(self):
        """VADモデルを初期化"""
        try:
            if not WEBRTCVAD_AVAILABLE or webrtcvad is None:
                logger.warning("[VADDetector] webrtcvad not available. Using fallback VAD.")
                self.vad = None
                return
            
            # webrtcvadを初期化
            # aggressiveness: 0=最も敏感、3=最も厳格
            # threshold (0.0-1.0) を 0-3 にマッピング
            aggressiveness = int(self.threshold * 3)
            aggressiveness = max(0, min(3, aggressiveness))  # 0-3の範囲に制限
            
            try:
                self.vad = webrtcvad.Vad(aggressiveness)
                
                # サンプルレートの検証
                if self.sample_rate not in [8000, 16000, 32000, 48000]:
                    logger.warning(f"[VADDetector] Unsupported sample rate: {self.sample_rate}. Using fallback VAD.")
                    self.vad = None
                    return
                
                logger.info(f"[VADDetector] webrtcvad initialized successfully (aggressiveness={aggressiveness}, sample_rate={self.sample_rate}Hz)")
            except Exception as e:
                logger.warning(f"[VADDetector] Failed to initialize webrtcvad: {e}")
                self.vad = None
                
        except Exception as e:
            logger.error(f"[VADDetector] Failed to initialize VAD model: {e}", exc_info=True)
            self.vad = None
    
    def process_audio_frame(self, audio_data: np.ndarray, timestamp: float) -> bool:
        """音声フレームを処理して音声活動を検出
        
        Args:
            audio_data: 音声データ（numpy配列、16bit PCM）
            timestamp: タイムスタンプ（秒）
            
        Returns:
            音声が検出された場合True
        """
        if self.vad is None:
            # フォールバック: 簡易的な音量ベースの検出
            return self._fallback_detection(audio_data, timestamp)
        
        try:
            # webrtcvadは10ms、20ms、30msのフレームサイズをサポート
            # 入力データを適切なサイズに分割して処理
            has_speech = False
            
            # フレームサイズに合わせてデータを分割
            num_frames = len(audio_data) // self.frame_size
            if num_frames == 0:
                # フレームサイズ未満の場合は、パディングまたはスキップ
                return False
            
            # 各フレームを処理
            for i in range(num_frames):
                start_idx = i * self.frame_size
                end_idx = start_idx + self.frame_size
                frame = audio_data[start_idx:end_idx]
                
                # int16のバイト列に変換（webrtcvadの要件）
                if frame.dtype != np.int16:
                    frame = frame.astype(np.int16)
                
                frame_bytes = frame.tobytes()
                
                # webrtcvadで検出
                try:
                    frame_has_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
                    if frame_has_speech:
                        has_speech = True
                        break  # 1フレームでも音声が検出されればTrue
                except Exception as e:
                    logger.warning(f"[VADDetector] Error in webrtcvad.is_speech: {e}")
                    # エラーが発生した場合はフォールバック
                    return self._fallback_detection(audio_data, timestamp)
            
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

