"""Interactive Voice Evaluator (IVE) - メインエントリーポイント"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from tester.orchestrator import Orchestrator
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer


def setup_logging(test_id: str, logs_dir: str = "logs"):
    """ロギング設定（ファイルとコンソールの両方に出力）"""
    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    
    # ログファイル名（テストIDとタイムスタンプを含む）
    log_filename = logs_path / f"{test_id}.log"
    
    # フォーマット設定
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # ファイルハンドラー
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # コンソールハンドラー
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # ルートロガーに設定
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return log_filename


logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.json") -> dict:
    """設定ファイルを読み込む"""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)


async def main():
    """WebRTCクライアントとして動作（評価ツール）"""
    try:
        # 設定ファイルを読み込み
        config = load_config()
        
        # テストIDを生成（簡易版）
        test_id = f"test_{asyncio.get_event_loop().time():.0f}"
        
        # ロギング設定（テストIDごとにログファイルを作成）
        log_filename = setup_logging(test_id, logs_dir="logs")
        
        logger.info("=" * 60)
        logger.info(f"Interactive Voice Evaluator (IVE) - WebRTC Client")
        logger.info(f"Test ID: {test_id}")
        logger.info(f"Log file: {log_filename}")
        logger.info("=" * 60)
        
        # オーケストレーターを初期化
        orchestrator = Orchestrator(config)
        
        # テストを開始（ロガーを初期化）
        await orchestrator.run_test(test_id, test_type="rule")
        
        # 音声ファイルのパスを取得（コマンドライン引数またはデフォルト）
        audio_file = sys.argv[1] if len(sys.argv) > 1 else "tests/hello.wav"
        audio_path = Path(audio_file)
        
        if not audio_path.exists():
            logger.warning(f"Audio file not found: {audio_file}")
            logger.info("Continuing without audio file...")
            audio_file = None
        else:
            logger.info(f"Using audio file: {audio_file}")
        
        # WebRTCクライアントを起動
        pc = RTCPeerConnection()
        
        # 先にOfferを待つ（サーバーが作成する）
        logger.info("offer.json を待機中...")
        while not os.path.exists("offer.json"):
            await asyncio.sleep(0.5)
        
        with open("offer.json", "r") as f:
            offer_data = json.load(f)
            await pc.setRemoteDescription(RTCSessionDescription(**offer_data))
        logger.info("offer.json を受信しました。")

        # Offerを受け取った後に、送信したいファイルをトラックに追加
        if audio_file and os.path.exists(audio_file):
            player = MediaPlayer(audio_file)
            pc.addTrack(player.audio)
            logger.info(f"{audio_file} をセットアップしました。")
        else:
            logger.warning("音声ファイルが見つかりません。")
            return

        # Answerを作成
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        with open("answer.json", "w") as f:
            json.dump({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}, f)
        logger.info("answer.json を作成しました。")

        # 音声ファイルの長さ分待機
        if audio_file:
            try:
                import soundfile as sf
                audio_data, sample_rate = sf.read(audio_file)
                duration = len(audio_data) / sample_rate
                logger.info(f"音声ファイルの長さ: {duration:.2f}秒、送信完了を待機中...")
                await asyncio.sleep(duration + 1.0)  # 送信完了を待つ（少し余裕を持たせる）
            except Exception as e:
                logger.warning(f"音声ファイルの長さを取得できませんでした: {e}")
                await asyncio.sleep(12)  # デフォルトで12秒待機
        else:
            await asyncio.sleep(12)
        
        logger.info("接続を閉じます...")
        await pc.close()
        
        # タイムラインを保存
        timeline_path = orchestrator.logger.save_timeline()
        
        logger.info("=" * 60)
        logger.info("Client completed successfully!")
        logger.info(f"Timeline saved to: {timeline_path}")
        logger.info(f"Log file: {log_filename}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())

