"""10秒間の音声ストリームトラック（無音 + MP3）"""

import asyncio
import numpy as np
from aiortc import MediaStreamTrack
from aiortc.contrib.media import MediaPlayer
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AudioStreamTrack(MediaStreamTrack):
    """10秒間の音声ストリーム（無音 → MP3 → 無音）"""
    
    kind = "audio"
    
    def __init__(self, audio_file: str, duration_seconds: float = 10.0, speech_start_second: float = 3.0):
        """音声ストリームトラックを初期化
        
        Args:
            audio_file: 音声ファイルのパス
            duration_seconds: 全体の長さ（秒）
            speech_start_second: 音声を開始する時刻（秒）
        """
        super().__init__()
        self.audio_file = Path(audio_file)
        self.duration_seconds = duration_seconds
        self.speech_start_second = speech_start_second
        self.sample_rate = 48000  # WebRTCの標準サンプリングレート
        self.channels = 1  # モノラル
        
        # 音声プレーヤーを初期化
        self.player = None
        if self.audio_file.exists():
            self.player = MediaPlayer(str(self.audio_file))
            logger.info(f"[AudioStreamTrack] Loaded audio file: {audio_file}")
        else:
            logger.warning(f"[AudioStreamTrack] Audio file not found: {audio_file}")
        
        self.start_time = None
        self.frame_count = 0
        
    async def recv(self):
        """音声フレームを生成"""
        if self.start_time is None:
            self.start_time = asyncio.get_event_loop().time()
        
        current_time = asyncio.get_event_loop().time()
        elapsed = current_time - self.start_time
        
        # 10秒経過したら終了
        if elapsed >= self.duration_seconds:
            raise StopAsyncIteration()
        
        # 音声開始時刻より前は無音を送信
        if elapsed < self.speech_start_second:
            # 無音フレームを生成（20ms分）
            frame_duration = 0.02  # 20ms
            samples = int(self.sample_rate * frame_duration)
            silence = np.zeros((samples,), dtype=np.float32)
            
            # AudioFrame形式に変換
            from aiortc import AudioFrame
            frame = AudioFrame.from_ndarray(
                silence.reshape(1, -1),  # (channels, samples)
                layout="mono",
                sample_rate=self.sample_rate
            )
            frame.pts = self.frame_count
            frame.sample_rate = self.sample_rate
            self.frame_count += samples
            return frame
        
        # 音声開始時刻以降はMP3から取得
        elif self.player and self.player.audio:
            try:
                frame = await self.player.audio.recv()
                # タイムスタンプを調整
                frame.pts = self.frame_count
                self.frame_count += int(self.sample_rate * 0.02)  # 20ms分
                return frame
            except StopAsyncIteration:
                # MP3が終了したら無音を送信
                frame_duration = 0.02
                samples = int(self.sample_rate * frame_duration)
                silence = np.zeros((samples,), dtype=np.float32)
                
                from aiortc import AudioFrame
                frame = AudioFrame.from_ndarray(
                    silence.reshape(1, -1),
                    layout="mono",
                    sample_rate=self.sample_rate
                )
                frame.pts = self.frame_count
                frame.sample_rate = self.sample_rate
                self.frame_count += samples
                return frame
        
        # フォールバック: 無音
        frame_duration = 0.02
        samples = int(self.sample_rate * frame_duration)
        silence = np.zeros((samples,), dtype=np.float32)
        
        from aiortc import AudioFrame
        frame = AudioFrame.from_ndarray(
            silence.reshape(1, -1),
            layout="mono",
            sample_rate=self.sample_rate
        )
        frame.pts = self.frame_count
        frame.sample_rate = self.sample_rate
        self.frame_count += samples
        return frame

