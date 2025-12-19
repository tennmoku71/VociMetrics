"""WebRTC Client: aiortcによる音声送受信"""

import asyncio
import json
from typing import Optional, Dict, Any
from pathlib import Path
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaPlayer, MediaRecorder
import aiohttp
import logging

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
                
                if self.logger:
                    self.logger.log_bot_speech_start()
        
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
        
    async def close(self):
        """接続を閉じる"""
        if self.recorder:
            await self.recorder.stop()
            logger.info("[WebRTCClient] Recording stopped")
            
        if self.pc:
            await self.pc.close()
            self.pc = None
            logger.info("[WebRTCClient] Connection closed")

