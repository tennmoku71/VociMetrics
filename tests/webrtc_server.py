"""テスト用WebRTCサーバー（音声送受信の相手側）"""

import asyncio
import json
import logging
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer, MediaRecorder
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# グローバル変数でピアコネクションを保持
pcs = set()


async def offer(request):
    """WebRTCオファーを受信してアンサーを返す"""
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"[Server] ICE connection state: {pc.iceConnectionState}")
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    # 録音した10秒の音声ファイルを送信
    audio_file = Path("tests/test_recording.wav")
    if not audio_file.exists():
        # フォールバック: hello.mp3を使用
        audio_file = Path("tests/hello.mp3")
    
    if audio_file.exists():
        logger.info(f"[Server] Sending audio file: {audio_file}")
        try:
            # MediaPlayerで直接送信（10秒間）
            player = MediaPlayer(str(audio_file))
            pc.addTrack(player.audio)
            logger.info("[Server] Audio track added to peer connection")
        except Exception as e:
            logger.error(f"[Server] Failed to add audio track: {e}", exc_info=True)
    else:
        logger.warning(f"[Server] Audio file not found: {audio_file}")
    
    # 接続状態の監視と10秒後の切断
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"[Server] Connection state: {pc.connectionState}")
        if pc.connectionState == "connected":
            logger.info("[Server] Connection established, audio should be streaming...")
            
            # 10秒後に接続を閉じる
            async def close_after_duration():
                await asyncio.sleep(10)
                logger.info("[Server] 10 seconds elapsed, closing connection...")
                if pc.connectionState != "closed":
                    await pc.close()
                    pcs.discard(pc)
            
            asyncio.create_task(close_after_duration())
        elif pc.connectionState == "closed":
            logger.info("[Server] Connection closed")

    @pc.on("track")
    async def on_track(track):
        logger.info(f"[Server] Received track: {track.kind}")
        
        if track.kind == "audio":
            # 受信した音声を記録
            recorder = MediaRecorder("data/reports/received_audio.wav")
            recorder.addTrack(track)
            await recorder.start()
            logger.info("[Server] Audio track received, recording...")
            
            # 接続が閉じられたら記録を停止
            @pc.on("connectionstatechange")
            async def on_connectionstatechange_for_recorder():
                if pc.connectionState == "closed":
                    await recorder.stop()
                    logger.info("[Server] Recording stopped")

    # オファーをセット
    await pc.setRemoteDescription(offer)

    # アンサーを生成
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })
    )


async def on_shutdown(app):
    """シャットダウン時にすべての接続を閉じる"""
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


def create_app():
    """アプリケーションを作成"""
    app = web.Application()
    app.router.add_post("/offer", offer)
    app.on_shutdown.append(on_shutdown)
    return app


async def main():
    """メイン実行関数"""
    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, "localhost", 8080)
    await site.start()
    
    logger.info("[Server] WebRTC server started on http://localhost:8080")
    logger.info("[Server] Waiting for connections...")
    
    # 無限に待機
    try:
        await asyncio.Future()
    except KeyboardInterrupt:
        logger.info("[Server] Shutting down...")
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

