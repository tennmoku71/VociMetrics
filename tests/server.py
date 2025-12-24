import asyncio
import json
import os
import signal
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRecorder

# グローバル変数でシャットダウンフラグを管理
shutdown_event = asyncio.Event()


def signal_handler(signum, frame):
    """シグナルハンドラー（CTRL+C）"""
    print("\nシャットダウンシグナルを受信しました。安全に終了します...")
    shutdown_event.set()


async def run():
    pc = RTCPeerConnection()
    recorder = MediaRecorder("server_received.wav")

    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            print("クライアントからの音声トラックを受信しました")
            recorder.addTrack(track)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"接続状態: {pc.connectionState}")
        if pc.connectionState == "closed":
            print("接続が閉じられました。")
            shutdown_event.set()

    # シグナルハンドラーを設定
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # 1. 空の状態でOfferを作成（相手に送るものはないが、セッションを開始する）
        # クライアントがaddTrackすることを期待して、ダミーのトランシーバーを追加
        pc.addTransceiver("audio", direction="recvonly")
        
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        
        with open("offer.json", "w") as f:
            json.dump({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}, f)
        print("1. offer.json を作成しました。")

        while not os.path.exists("answer.json") and not shutdown_event.is_set():
            await asyncio.sleep(0.5)
        
        if shutdown_event.is_set():
            print("シャットダウンが要求されました。")
            return
        
        with open("answer.json", "r") as f:
            answer = json.load(f)
            await pc.setRemoteDescription(RTCSessionDescription(**answer))

        print("接続完了。受信を開始します... (CTRL+Cで終了)")
        await recorder.start()
        
        try:
            # 無限に待機（シャットダウンシグナルまたは接続終了まで）
            await shutdown_event.wait()
        finally:
            print("録音を停止しています...")
            await recorder.stop()
            await pc.close()
            print("終了。server_received.wav を確認してください。")
            
    except KeyboardInterrupt:
        print("\nキーボード割り込みを受信しました。安全に終了します...")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
    finally:
        try:
            if pc.connectionState != "closed":
                await pc.close()
        except Exception as e:
            print(f"接続を閉じる際にエラーが発生しました: {e}")
        if not shutdown_event.is_set():
            shutdown_event.set()


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nプログラムを終了します。")