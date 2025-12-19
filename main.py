"""Interactive Voice Evaluator (IVE) - メインエントリーポイント"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from tester.orchestrator import Orchestrator
from tester.webrtc_client import WebRTCClient


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
    """メイン実行関数"""
    try:
        # 設定ファイルを読み込み
        config = load_config()
        
        # テストIDを生成（簡易版）
        test_id = f"test_{asyncio.get_event_loop().time():.0f}"
        
        # ロギング設定（テストIDごとにログファイルを作成）
        log_filename = setup_logging(test_id, logs_dir="logs")
        
        logger.info("=" * 60)
        logger.info(f"Interactive Voice Evaluator (IVE) - Test: {test_id}")
        logger.info(f"Log file: {log_filename}")
        logger.info("=" * 60)
        logger.info("Loading configuration...")
        logger.info("Configuration loaded successfully")
        
        # 音声ファイルのパスを取得（コマンドライン引数またはデフォルト）
        audio_file = sys.argv[1] if len(sys.argv) > 1 else "tests/hello.mp3"
        audio_path = Path(audio_file)
        
        if not audio_path.exists():
            logger.warning(f"Audio file not found: {audio_file}")
            logger.info("Continuing without audio file...")
            audio_file = None
        else:
            logger.info(f"Using audio file: {audio_file}")
        
        # オーケストレーターを初期化
        orchestrator = Orchestrator(config)
        
        logger.info(f"Starting test: {test_id}")
        logger.info("Make sure the WebRTC server is running: python tests/webrtc_server.py")
        
        # テストを開始（ロガーを初期化）
        await orchestrator.run_test(test_id, test_type="rule")
        
        # WebRTCクライアントを初期化（ロガーを渡す）
        webrtc_config = config.get("webrtc", {})
        webrtc_client = WebRTCClient(
            webrtc_config,
            logger_instance=orchestrator.logger,
            server_url="http://localhost:8080"
        )
        
        # WebRTC接続を確立（音声ファイルを送信）
        await webrtc_client.connect(audio_file_path=str(audio_path) if audio_file else None)
        
        # 接続が確立されるまで待機
        try:
            await webrtc_client.wait_for_connection(timeout=10.0)
        except TimeoutError:
            logger.warning("Connection timeout, but continuing...")
        
        # サーバーからの音声ストリームを10秒間待機
        logger.info("Waiting for 10-second audio stream from server...")
        await asyncio.sleep(12)  # 10秒のストリーム + バッファ
        
        # 接続を閉じる
        await webrtc_client.close()
        
        # タイムラインを保存
        timeline_path = orchestrator.logger.save_timeline()
        
        logger.info("=" * 60)
        logger.info("Test completed successfully!")
        logger.info(f"Timeline saved to: {timeline_path}")
        logger.info(f"Received audio saved to: data/reports/received_audio.wav")
        logger.info(f"Log file: {log_filename}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())

