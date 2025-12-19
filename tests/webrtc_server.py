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

    @pc.on("track")
    async def on_track(track):
        logger.info(f"[Server] Received track: {track.kind}")
        
        if track.kind == "audio":
            # 受信した音声を記録
            recorder = MediaRecorder("data/reports/received_audio.wav")
            recorder.addTrack(track)
            await recorder.start()
            
            # テスト用の音声ファイルを送信（エコーバック）
            # 実際の実装では、ここで音声処理やAI応答を実装
            logger.info("[Server] Audio track received, recording...")
            
            # 接続が閉じられたら記録を停止
            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
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

