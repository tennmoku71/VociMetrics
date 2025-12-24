import asyncio, json, os
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer

async def run():
    pc = RTCPeerConnection()
    
    # 先にOfferを待つ
    print("offer.json を待機中...")
    while not os.path.exists("offer.json"):
        await asyncio.sleep(0.5)
    
    with open("offer.json", "r") as f:
        offer_data = json.load(f)
        await pc.setRemoteDescription(RTCSessionDescription(**offer_data))

    # Offerを受け取った後に、送信したいファイルをトラックに追加
    audio_file = "tests/hello.wav"
    if os.path.exists(audio_file):
        player = MediaPlayer(audio_file)
        pc.addTrack(player.audio)
        print(f"{audio_file} をセットアップしました。")
    else:
        print("ファイルが見つかりません。")
        return

    # Answerを作成
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    with open("answer.json", "w") as f:
        json.dump({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}, f)
    print("2. answer.json を作成しました。")

    await asyncio.sleep(12) # 再生時間に余裕を持つ
    await pc.close()
    print("送信終了。")

if __name__ == "__main__":
    asyncio.run(run())