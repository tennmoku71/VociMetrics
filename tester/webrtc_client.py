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
    
    def __init__(self, config: Dict[str, Any], logger_instance=None, orchestrator=None, server_url: str = "http://localhost:8080"):
        self.config = config
        self.logger = logger_instance
        self.orchestrator = orchestrator  # Orchestratorへの参照
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
        self.audio_track: Optional[Any] = None  # カスタムトラックへの参照
        
    async def connect(self, audio_file_path: Optional[str] = None):
        """WebRTC接続を確立し、音声ファイルを送信"""
        logger.info("[WebRTCClient] Initializing connection...")
        
        # ICEサーバーの設定
        ice_servers_config = self.config.get("ice_servers", [])
        ice_servers = [RTCIceServer(urls=[s["urls"][0]]) for s in ice_servers_config]
        configuration = RTCConfiguration(iceServers=ice_servers)
        self.pc = RTCPeerConnection(configuration=configuration)
        
        # 動的に音声ファイルを送信できるカスタムトラックを作成
        # 双方向通信を確立するため、常に音声トラックを追加
        try:
            from aiortc import MediaStreamTrack
            # aiortcのバージョンによっては、AudioFrameのインポート方法が異なる
            try:
                from aiortc import AudioFrame
            except ImportError:
                from aiortc.mediastreams import AudioFrame
            
            class DynamicAudioTrack(MediaStreamTrack):
                """動的に音声ファイルを読み込めるカスタムトラック"""
                kind = "audio"
                
                def __init__(self, sample_rate: int = 48000):
                    super().__init__()
                    self.sample_rate = sample_rate
                    self.frame_count = 0
                    self.pts_samples = 0  # ptsはサンプル数で表現（累積）
                    self.current_player: Optional[MediaPlayer] = None
                    self.current_audio_file: Optional[str] = None
                    self._lock = asyncio.Lock()
                    
                async def set_audio_file(self, audio_file_path: str):
                    """音声ファイルを設定（動的に切り替え可能）"""
                    async with self._lock:
                        logger.info(f"[DynamicAudioTrack] set_audio_file called with: {audio_file_path}")
                        audio_path = Path(audio_file_path)
                        if not audio_path.exists():
                            # 相対パスを試す
                            if audio_path.name == "hello.wav" or "hello" in audio_path.name:
                                # 48000HzのWAVファイルを優先
                                audio_path_48k = Path("tests/hello_48k.wav")
                                if audio_path_48k.exists():
                                    audio_path = audio_path_48k
                                else:
                                    audio_path = Path("tests/hello.wav")
                            elif "recorded" in audio_path.name:
                                audio_path = Path("data/reports") / audio_path.name
                            else:
                                audio_path = Path("tests") / audio_path.name
                        
                        if not audio_path.exists():
                            logger.warning(f"[DynamicAudioTrack] Audio file not found: {audio_file_path} (tried: {audio_path})")
                            return False
                        
                        logger.info(f"[DynamicAudioTrack] Found audio file: {audio_path}")
                        
                        # 既存のプレーヤーを閉じる
                        if self.current_player:
                            try:
                                if hasattr(self.current_player, 'audio') and self.current_player.audio:
                                    self.current_player.audio.stop()
                            except:
                                pass
                            self.current_player = None
                        
                        # 新しいプレーヤーを作成
                        try:
                            # MediaPlayerをオプションなしで作成（フレームのサンプリングレートは後で確認）
                            self.current_player = MediaPlayer(str(audio_path))
                            self.current_audio_file = str(audio_path)
                            # フレームカウントをリセット（新しい音声ファイルを読み込むため）
                            self.frame_count = 0
                            self.pts_samples = 0
                            logger.info(f"[DynamicAudioTrack] Loaded audio file: {audio_path}, player={self.current_player}, audio={self.current_player.audio if self.current_player else None}, frame_count reset to 0")
                            return True
                        except Exception as e:
                            logger.error(f"[DynamicAudioTrack] Failed to load audio file: {e}", exc_info=True)
                            return False
                
                async def recv(self):
                    """音声フレームを生成（48000Hz固定）"""
                    async with self._lock:
                        if self.current_player and self.current_player.audio:
                            try:
                                original_frame = await self.current_player.audio.recv()
                                audio_data = original_frame.to_ndarray()
                                
                                # モノラルに変換
                                if len(audio_data.shape) > 1:
                                    if audio_data.shape[0] > 1:
                                        audio_data = np.mean(audio_data, axis=0)
                                    elif audio_data.shape[1] > 1:
                                        audio_data = audio_data[0]
                                
                                # int16に変換（AudioFrame.from_ndarray()はint16を期待）
                                if audio_data.dtype != np.int16:
                                    audio_data_float = audio_data.astype(np.float32)
                                    audio_data = (np.clip(audio_data_float, -1.0, 1.0) * 32767).astype(np.int16)
                                
                                # AudioFrameを作成（48000Hz、モノラル）
                                frame = AudioFrame.from_ndarray(
                                    audio_data.reshape(1, -1),
                                    layout="mono"
                                )
                                
                                # ptsは累積サンプル数で管理（シンプルに）
                                frame.pts = self.pts_samples
                                frame.sample_rate = self.sample_rate
                                
                                # 累積サンプル数を更新
                                num_samples = len(audio_data)
                                self.pts_samples += num_samples
                                self.frame_count += 1
                                
                                return frame
                            except StopAsyncIteration:
                                # 音声ファイルが終了したら無音を送信
                                logger.info(f"[DynamicAudioTrack] Audio file finished: {self.current_audio_file}")
                                self.current_player = None
                                self.current_audio_file = None
                                # 無音を送信して続行
                                return await self._create_silence_frame()
                            except MediaStreamError:
                                # MediaStreamErrorは音声ファイルが終了したことを示す
                                logger.debug(f"[DynamicAudioTrack] MediaStreamError: Audio file ended: {self.current_audio_file}")
                                self.current_player = None
                                self.current_audio_file = None
                                # 無音を送信して続行
                                return await self._create_silence_frame()
                            except Exception as e:
                                # その他のエラーは最初の1回だけログ出力
                                if not hasattr(self, '_error_logged'):
                                    logger.warning(f"[DynamicAudioTrack] Error receiving frame: {e}")
                                    self._error_logged = True
                                # エラー時は無音を送信
                                return await self._create_silence_frame()
                        else:
                            # 音声ファイルが設定されていない場合は無音を送信
                            if self.frame_count == 0 or self.frame_count % 100 == 0:
                                logger.info(f"[DynamicAudioTrack] No audio file set, sending silence (frame {self.frame_count})")
                            return await self._create_silence_frame()
                
                async def _create_silence_frame(self):
                    """無音フレームを作成"""
                    frame_duration = 0.02  # 20ms
                    samples = int(self.sample_rate * frame_duration)
                    silence = np.zeros((samples,), dtype=np.int16)
                    
                    try:
                        frame = AudioFrame.from_ndarray(
                            silence.reshape(1, -1),  # (channels, samples)
                            layout="mono"
                        )
                        frame.pts = self.pts_samples  # ptsはサンプル数で表現
                        frame.sample_rate = self.sample_rate
                        self.pts_samples += samples  # 累積サンプル数を更新
                        self.frame_count += 1
                        return frame
                    except Exception as e:
                        logger.error(f"[DynamicAudioTrack] Failed to create silence frame: {e}")
                        await asyncio.sleep(0.02)
                        return await self.recv()
            
            # カスタムトラックを作成（初期は無音）
            self.audio_track = DynamicAudioTrack(sample_rate=48000)
            self.pc.addTrack(self.audio_track)
            logger.info("[WebRTCClient] Dynamic audio track added for bidirectional connection")
            
            # 初期音声ファイルが指定されている場合は設定
            if audio_file_path:
                await self.audio_track.set_audio_file(audio_file_path)
                if self.logger:
                    self.logger.log_user_speech_start(text=f"Audio file: {audio_file_path}")
        except Exception as e:
            logger.warning(f"[WebRTCClient] Failed to add audio track: {e}", exc_info=True)
        
        # 受信した音声を記録（サーバー側で記録するため、クライアント側では記録しない）
        # 注意: サーバー側で記録するため、クライアント側のMediaRecorderは無効化
        # 必要に応じて、別のファイル名で記録する場合は以下を有効化
        # output_dir = Path("data/reports")
        # output_dir.mkdir(parents=True, exist_ok=True)
        # self.recorder = MediaRecorder(str(output_dir / "client_received_audio.wav"))
        self.recorder = None
        
        @self.pc.on("track")
        async def on_track(track):
            logger.info(f"[WebRTCClient] Received track: {track.kind}")
            if track.kind == "audio":
                # MediaRecorderは無効化（サーバー側で記録するため）
                # if self.recorder:
                #     self.recorder.addTrack(track)
                #     await self.recorder.start()
                
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
                    
                    # デバッグ: 最初の数フレームをログ出力
                    if frame_count <= 10:
                        logger.info(f"[WebRTCClient] Received frame {frame_count} from track")
                    
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
                    
                    # デバッグ: 音声検出をログ出力
                    if has_speech:
                        logger.info(f"[WebRTCClient] Speech detected at {relative_time:.3f}s (frame {frame_count})")
                    elif frame_count % 50 == 0:
                        logger.debug(f"[WebRTCClient] No speech detected at {relative_time:.3f}s (frame {frame_count})")
                    
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
        # Orchestratorにイベントを通知
        if self.orchestrator:
            self.orchestrator.notify_event("BOT_SPEECH_START")
    
    def _on_bot_speech_end(self, timestamp: float):
        """ボット発話終了時のコールバック"""
        logger.info(f"[WebRTCClient] Bot speech ended (VAD) at {timestamp:.3f}s")
        if self.logger:
            self.logger.log_bot_speech_end()
        # Orchestratorにイベントを通知
        if self.orchestrator:
            self.orchestrator.notify_event("BOT_SPEECH_END")
    
    async def close(self):
        """接続を閉じる"""
        # MediaRecorderを停止（無効化されている場合はスキップ）
        if self.recorder:
            try:
                await self.recorder.stop()
            except Exception as e:
                logger.warning(f"[WebRTCClient] Failed to stop recorder: {e}")
        
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
    
    async def send_audio_file(self, audio_file_path: str):
        """音声ファイルを送信（動的に切り替え）"""
        if self.audio_track:
            logger.info(f"[WebRTCClient] Sending audio file: {audio_file_path}")
            success = await self.audio_track.set_audio_file(audio_file_path)
            if success and self.logger:
                self.logger.log_user_speech_start(text=f"Audio file: {audio_file_path}")
            return success
        else:
            logger.warning("[WebRTCClient] Audio track not available")
            return False

