"""WebRTC Client: aiortcによる音声送受信"""

import asyncio
import json
from typing import Optional, Dict, Any
from pathlib import Path
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaPlayer, MediaRecorder
from aiortc.rtcrtpreceiver import RTCRtpReceiver
from aiortc.mediastreams import MediaStreamError
import aiohttp
import logging
import numpy as np
from tester.vad_detector import VADDetector

logger = logging.getLogger(__name__)


class WebRTCClient:
    """WebRTCクライアント（音声送受信実装）"""
    
    def __init__(self, config: Dict[str, Any], logger_instance=None, server_url: str = "http://localhost:8080"):
        self.config = config
        self.logger = logger_instance
        self.server_url = server_url
        self.pc: Optional[RTCPeerConnection] = None
        self.player: Optional[MediaPlayer] = None
        self.recorder: Optional[MediaRecorder] = None
        
        # VAD設定
        webrtc_config = config.get("webrtc", {})
        sample_rate = webrtc_config.get("sample_rate", 16000)
        
        # VAD検出器を初期化
        self.vad_detector = VADDetector(
            sample_rate=sample_rate,
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
            on_speech_start=self._on_bot_speech_start,
            on_speech_end=self._on_bot_speech_end
        )
        
        self.vad_task: Optional[asyncio.Task] = None
        self.audio_buffer = []
        self.start_time: Optional[float] = None
        
    async def connect(self, audio_file_path: Optional[str] = None):
        """WebRTC接続を確立し、音声ファイルを送信"""
        logger.info("[WebRTCClient] Initializing connection...")
        
        # ICEサーバーの設定
        ice_servers_config = self.config.get("ice_servers", [])
        ice_servers = [RTCIceServer(urls=[s["urls"][0]]) for s in ice_servers_config]
        configuration = RTCConfiguration(iceServers=ice_servers)
        self.pc = RTCPeerConnection(configuration=configuration)
        
        # 音声ファイルを送信する場合
        if audio_file_path and Path(audio_file_path).exists():
            logger.info(f"[WebRTCClient] Loading audio file: {audio_file_path}")
            self.player = MediaPlayer(audio_file_path)
            self.pc.addTrack(self.player.audio)
            
            if self.logger:
                self.logger.log_user_speech_start(text=f"Audio file: {audio_file_path}")
        
        # 受信した音声を記録
        output_dir = Path("data/reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.recorder = MediaRecorder(str(output_dir / "received_audio.wav"))
        
        @self.pc.on("track")
        async def on_track(track):
            logger.info(f"[WebRTCClient] Received track: {track.kind}")
            if track.kind == "audio":
                self.recorder.addTrack(track)
                await self.recorder.start()
                
                # VAD処理を開始
                self.start_time = asyncio.get_event_loop().time()
                self.vad_task = asyncio.create_task(
                    self._process_audio_track_for_vad(track)
                )
        
        @self.pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            logger.info(f"[WebRTCClient] ICE connection state: {self.pc.iceConnectionState}")
        
        # オファーを作成
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)
        
        # サーバーにオファーを送信してアンサーを受信
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.server_url}/offer",
                json={
                    "sdp": self.pc.localDescription.sdp,
                    "type": self.pc.localDescription.type
                }
            ) as resp:
                answer_data = await resp.json()
                answer = RTCSessionDescription(
                    sdp=answer_data["sdp"],
                    type=answer_data["type"]
                )
                await self.pc.setRemoteDescription(answer)
        
        logger.info("[WebRTCClient] Connection established")
        
    async def wait_for_connection(self, timeout: float = 10.0):
        """接続が確立されるまで待機"""
        start_time = asyncio.get_event_loop().time()
        while self.pc.iceConnectionState not in ["connected", "completed"]:
            if asyncio.get_event_loop().time() - start_time > timeout:
                raise TimeoutError("Connection timeout")
            await asyncio.sleep(0.1)
        logger.info("[WebRTCClient] Connection established")
        
    async def _process_audio_track_for_vad(self, track):
        """音声トラックを処理してVAD検出を行う"""
        try:
            sample_rate = self.config.get("webrtc", {}).get("sample_rate", 16000)
            frame_count = 0
            
            logger.info("[WebRTCClient] Starting VAD processing...")
            
            while True:
                try:
                    # 音声フレームを受信
                    frame = await track.recv()
                    frame_count += 1
                    
                    if frame_count % 100 == 0:
                        logger.debug(f"[WebRTCClient] Processed {frame_count} audio frames")
                    
                    # フレームから音声データを取得
                    # aiortcのAudioFrameはto_ndarray()メソッドを持つ
                    try:
                        audio_data = frame.to_ndarray()
                    except AttributeError:
                        # フォールバック: frameがndarrayの場合
                        audio_data = np.array(frame)
                    
                    # ステレオの場合はモノラルに変換
                    if len(audio_data.shape) > 1:
                        if audio_data.shape[0] > 1:
                            # (channels, samples) -> (samples,)
                            audio_data = np.mean(audio_data, axis=0)
                        elif audio_data.shape[1] > 1:
                            # (samples, channels) -> (samples,)
                            audio_data = np.mean(audio_data, axis=1)
                    
                    # float32からint16に変換（必要に応じて）
                    if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                        # -1.0～1.0の範囲を-32768～32767に変換
                        audio_data = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
                    elif audio_data.dtype != np.int16:
                        audio_data = audio_data.astype(np.int16)
                    
                    # タイムスタンプを計算
                    current_time = asyncio.get_event_loop().time()
                    if self.start_time is None:
                        self.start_time = current_time
                    relative_time = current_time - self.start_time
                    
                    # VADで検出
                    has_speech = self.vad_detector.process_audio_frame(audio_data, relative_time)
                    
                    if has_speech and frame_count % 50 == 0:
                        logger.debug(f"[WebRTCClient] Speech detected at {relative_time:.3f}s")
                    
                except asyncio.CancelledError:
                    break
                except (StopAsyncIteration, MediaStreamError):
                    # トラックが終了した
                    logger.info("[WebRTCClient] Audio track ended")
                    break
                except Exception as e:
                    logger.warning(f"[WebRTCClient] Error processing audio frame: {e}")
                    # エラーが続く場合は終了
                    error_count = getattr(self, '_error_count', 0)
                    error_count += 1
                    self._error_count = error_count
                    if error_count > 3:
                        logger.error("[WebRTCClient] Too many errors, stopping VAD processing")
                        break
                    await asyncio.sleep(0.1)
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[WebRTCClient] VAD processing error: {e}", exc_info=True)
        finally:
            logger.info(f"[WebRTCClient] VAD processing stopped (processed {frame_count} frames)")
    
    def _on_bot_speech_start(self, timestamp: float):
        """ボット発話開始時のコールバック"""
        logger.info(f"[WebRTCClient] Bot speech detected (VAD) at {timestamp:.3f}s")
        if self.logger:
            self.logger.log_bot_speech_start()
    
    def _on_bot_speech_end(self, timestamp: float):
        """ボット発話終了時のコールバック"""
        logger.info(f"[WebRTCClient] Bot speech ended (VAD) at {timestamp:.3f}s")
        if self.logger:
            self.logger.log_bot_speech_end()
    
    async def close(self):
        """接続を閉じる"""
        # VADタスクを停止
        if self.vad_task:
            self.vad_task.cancel()
            try:
                await self.vad_task
            except asyncio.CancelledError:
                pass
        
        if self.recorder:
            await self.recorder.stop()
            logger.info("[WebRTCClient] Recording stopped")
            
        if self.pc:
            await self.pc.close()
            self.pc = None
            logger.info("[WebRTCClient] Connection closed")

