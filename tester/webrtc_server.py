import asyncio, json, os
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRecorder

async def run():
    pc = RTCPeerConnection()
    recorder = MediaRecorder("server_received.wav")

    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            print("クライアントからの音声トラックを受信しました")
            recorder.addTrack(track)

    # 1. 空の状態でOfferを作成（相手に送るものはないが、セッションを開始する）
    # クライアントがaddTrackすることを期待して、ダミーのトランシーバーを追加
    pc.addTransceiver("audio", direction="recvonly")
    
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    
    with open("offer.json", "w") as f:
        json.dump({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}, f)
    print("1. offer.json を作成しました。")

    while not os.path.exists("answer.json"):
        await asyncio.sleep(0.5)
    
    with open("answer.json", "r") as f:
        answer = json.load(f)
        await pc.setRemoteDescription(RTCSessionDescription(**answer))

    print("接続完了。10秒間受信します...")
    await recorder.start()
    try:
        await asyncio.sleep(10) 
    finally:
        await recorder.stop()
        await pc.close()
        print("終了。server_received.wav を確認してください。")

if __name__ == "__main__":
    asyncio.run(run())
