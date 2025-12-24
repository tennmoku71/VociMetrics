"""Interactive Voice Evaluator (IVE) - メインエントリーポイント
WebSocketクライアントとして動作（評価ツール）
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from tester.orchestrator import Orchestrator
from tester.vad_detector import VADDetector
import numpy as np
import soundfile as sf
import aiohttp


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
    """WebSocketクライアントとして動作（評価ツール）"""
    try:
        # 設定ファイルを読み込み
        config = load_config()
        
        # テストIDを生成（簡易版）
        test_id = f"test_{asyncio.get_event_loop().time():.0f}"
        
        # ロギング設定（テストIDごとにログファイルを作成）
        log_filename = setup_logging(test_id, logs_dir="logs")
        
        logger.info("=" * 60)
        logger.info(f"Interactive Voice Evaluator (IVE) - WebSocket Client")
        logger.info(f"Test ID: {test_id}")
        logger.info(f"Log file: {log_filename}")
        logger.info("=" * 60)
        
        # オーケストレーターを初期化
        orchestrator = Orchestrator(config)
        
        # テストを開始（ロガーを初期化）
        await orchestrator.run_test(test_id, test_type="rule")
        
        # VAD検出器を初期化（サーバーからの音声を検出）
        websocket_config = config.get("websocket", {})
        input_sample_rate = websocket_config.get("sample_rate", 24000)  # WebSocketで受信する音声のサンプルレート
        vad_sample_rate = 16000  # webrtcvadが期待するサンプルレート
        
        # 音声開始/終了のコールバック
        def on_bot_speech_start(timestamp: float):
            """ボット（サーバー）の音声開始を記録"""
            logger.info(f"[VAD] Bot speech started at {timestamp:.3f}s")
            orchestrator.logger.log_bot_speech_start()
        
        def on_bot_speech_end(timestamp: float):
            """ボット（サーバー）の音声終了を記録"""
            logger.info(f"[VAD] Bot speech ended at {timestamp:.3f}s")
            orchestrator.logger.log_bot_speech_end()
        
        vad_detector = VADDetector(
            sample_rate=vad_sample_rate,  # 16000Hzで初期化
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
            on_speech_start=on_bot_speech_start,
            on_speech_end=on_bot_speech_end
        )
        
        # 音声ファイルのパスを取得（コマンドライン引数またはデフォルト）
        audio_file = sys.argv[1] if len(sys.argv) > 1 else "tests/hello.wav"
        audio_path = Path(audio_file)
        
        if not audio_path.exists():
            logger.warning(f"Audio file not found: {audio_file}")
            logger.info("Continuing without audio file...")
            audio_file = None
        else:
            logger.info(f"Using audio file: {audio_file}")
        
        # WebSocketサーバーに接続
        server_url = websocket_config.get("server_url", "ws://localhost:8765/ws")
        logger.info(f"Connecting to WebSocket server: {server_url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(server_url) as ws:
                logger.info("WebSocket接続が確立されました")
                
                # サーバーからの音声を受信してVADで処理するタスク
                async def receive_audio():
                    """サーバーからの音声を受信してVADで処理"""
                    start_time = asyncio.get_event_loop().time()
                    # VAD用のバッファ（リサンプリング用）
                    vad_buffer = np.array([], dtype=np.int16)
                    vad_chunk_size = 320  # 16000Hzで20ms = 320サンプル
                    
                    # scipyのインポート（リサンプリング用）
                    try:
                        from scipy import signal
                        has_scipy = True
                    except ImportError:
                        has_scipy = False
                        logger.warning("scipyがインストールされていません。VADの精度が低下する可能性があります。")
                    
                    try:
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.BINARY:
                                # バイナリメッセージ（音声データ）
                                # PCM形式（int16）を想定
                                audio_chunk = np.frombuffer(msg.data, dtype=np.int16)
                                
                                # VADバッファに追加（24000Hz）
                                vad_buffer = np.concatenate([vad_buffer, audio_chunk])
                                
                                # 24000Hzから16000Hzにリサンプリング（320サンプル以上になったら）
                                min_samples_for_resample = int(input_sample_rate * 0.02)  # 20ms分（24000Hzで480サンプル）
                                
                                if len(vad_buffer) >= min_samples_for_resample and has_scipy:
                                    # リサンプリング: 24000Hz → 16000Hz
                                    num_output_samples = int(len(vad_buffer) * vad_sample_rate / input_sample_rate)
                                    resampled_chunk = signal.resample(vad_buffer, num_output_samples).astype(np.int16)
                                    
                                    # VADバッファが一定の長さに達したら処理
                                    while len(resampled_chunk) >= vad_chunk_size:
                                        vad_chunk = resampled_chunk[:vad_chunk_size]
                                        resampled_chunk = resampled_chunk[vad_chunk_size:]
                                        
                                        # タイムスタンプを計算（相対時間）
                                        current_time = asyncio.get_event_loop().time()
                                        relative_time = current_time - start_time
                                        
                                        # VADで処理
                                        vad_detector.process_audio_frame(vad_chunk, relative_time)
                                    
                                    # 残りのデータを保持
                                    vad_buffer = resampled_chunk.astype(np.int16) if len(resampled_chunk) > 0 else np.array([], dtype=np.int16)
                                elif not has_scipy:
                                    # scipyがない場合は、そのまま処理（精度は低下）
                                    if len(vad_buffer) >= vad_chunk_size:
                                        vad_chunk = vad_buffer[:vad_chunk_size]
                                        vad_buffer = vad_buffer[vad_chunk_size:]
                                        
                                        # タイムスタンプを計算（相対時間）
                                        current_time = asyncio.get_event_loop().time()
                                        relative_time = current_time - start_time
                                        
                                        # VADで処理
                                        vad_detector.process_audio_frame(vad_chunk, relative_time)
                                
                            elif msg.type == aiohttp.WSMsgType.TEXT:
                                # テキストメッセージ（制御用）
                                try:
                                    data = json.loads(msg.data)
                                    logger.info(f"受信メッセージ: {data}")
                                except json.JSONDecodeError:
                                    logger.warning(f"無効なJSONメッセージ: {msg.data}")
                                    
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                logger.error(f"WebSocketエラー: {ws.exception()}")
                                break
                                
                    except Exception as e:
                        logger.info(f"[VAD] Audio reception ended: {e}")
                
                # 音声受信タスクを開始
                receive_task = asyncio.create_task(receive_audio())
                logger.info("[VAD] Started receiving audio and VAD processing...")
                
                # 音声送信の設定
                total_duration_seconds = 10.0  # 総送信時間（10秒）
                chunk_size = int(input_sample_rate * 0.01)  # 10ms
                chunk_duration = 0.01  # 10ms
                total_chunks = int(total_duration_seconds / chunk_duration)  # 総チャンク数
                
                # 無音チャンク（0で埋める）
                silence_chunk = np.zeros(chunk_size, dtype=np.int16)
                
                # 音声ファイルを送信
                audio_data = None
                audio_duration = 0.0
                
                if audio_file:
                    try:
                        audio_data, sr = sf.read(audio_file)
                        
                        # モノラルに変換
                        if len(audio_data.shape) > 1:
                            audio_data = np.mean(audio_data, axis=1)
                        
                        # int16に変換
                        if audio_data.dtype != np.int16:
                            audio_data = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
                        
                        # サンプルレートを24000Hzにリサンプリング（必要に応じて）
                        if sr != input_sample_rate:
                            try:
                                from scipy import signal
                                num_samples = int(len(audio_data) * input_sample_rate / sr)
                                audio_data = signal.resample(audio_data, num_samples).astype(np.int16)
                                logger.info(f"サンプルレートを{sr}Hzから{input_sample_rate}Hzにリサンプリングしました")
                            except ImportError:
                                logger.warning(f"scipyがインストールされていないため、リサンプリングをスキップします。"
                                             f"音声ファイルのサンプルレート({sr}Hz)が期待値({input_sample_rate}Hz)と異なる場合、"
                                             f"音声の速度が正しくない可能性があります。")
                                # リサンプリングできない場合は警告のみ
                        
                        audio_duration = len(audio_data) / input_sample_rate
                        logger.info(f"音声ファイルを送信します: {audio_file} ({audio_duration:.2f}秒)")
                        
                    except Exception as e:
                        logger.error(f"音声ファイルの読み込みエラー: {e}", exc_info=True)
                        audio_data = None
                
                # 10秒間、音声または無音を送信
                audio_chunks = []
                if audio_data is not None:
                    # 音声データをチャンクに分割
                    for i in range(0, len(audio_data), chunk_size):
                        chunk = audio_data[i:i+chunk_size]
                        # チャンクサイズに満たない場合は0でパディング
                        if len(chunk) < chunk_size:
                            padded_chunk = np.zeros(chunk_size, dtype=np.int16)
                            padded_chunk[:len(chunk)] = chunk
                            chunk = padded_chunk
                        audio_chunks.append(chunk)
                
                audio_chunk_count = len(audio_chunks)
                logger.info(f"音声送信を開始します（総時間: {total_duration_seconds}秒、音声: {audio_duration:.2f}秒、無音: {total_duration_seconds - audio_duration:.2f}秒）")
                
                for chunk_idx in range(total_chunks):
                    if chunk_idx < audio_chunk_count:
                        # 音声チャンクを送信
                        await ws.send_bytes(audio_chunks[chunk_idx].tobytes())
                    else:
                        # 無音チャンクを送信
                        await ws.send_bytes(silence_chunk.tobytes())
                    
                    await asyncio.sleep(chunk_duration)  # 10ms待機
                
                logger.info("音声送信が完了しました（10秒間）")
                
                # 受信タスクをキャンセル
                receive_task.cancel()
                try:
                    await receive_task
                except asyncio.CancelledError:
                    pass
                
                logger.info("WebSocket接続を閉じます...")
        
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
