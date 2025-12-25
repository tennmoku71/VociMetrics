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
from evaluator.evaluator import Evaluator
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
        
        # 評価エンジンを初期化
        evaluator = Evaluator(config)
        
        # VAD検出器を初期化（サーバーからの音声を検出）
        websocket_config = config.get("websocket", {})
        input_sample_rate = websocket_config.get("sample_rate", 24000)  # WebSocketで受信する音声のサンプルレート
        vad_sample_rate = 16000  # webrtcvadが期待するサンプルレート
        
        # .convoファイルのパスを取得（コマンドライン引数またはデフォルト）
        convo_file = sys.argv[1] if len(sys.argv) > 1 else "data/scenarios/sample.convo"
        convo_path = Path(convo_file)
        
        if not convo_path.exists():
            # scenarios_dirからの相対パスを試す
            scenarios_dir = config.get("scenarios_dir", "data/scenarios")
            convo_path = Path(scenarios_dir) / convo_file
            if not convo_path.exists():
                logger.error(f"Convo file not found: {convo_file}")
                logger.info("Please specify a valid .convo file path")
                sys.exit(1)
        
        logger.info(f"Using convo file: {convo_path}")
        
        # WebSocket接続とVAD検出器を初期化（シナリオ実行中に使用）
        ws_connection = None
        vad_detector = None
        
        # クライアント側VAD検知時間分析用のタイムスタンプ（現在のイベント用）
        client_vad_timestamps = {
            "first_audio_received": None,  # 最初の音声データを受信した時刻
            "first_frame_processed": None,  # 最初のVADフレームを処理した時刻
            "speech_start_detected": None,  # VADが音声開始を検出した時刻
            "event_logged": None,  # BOT_SPEECH_STARTイベントを記録した時刻
        }
        
        # クライアント側VAD検知時間分析用の履歴（各BOT_SPEECH_STARTイベントごとに保存）
        client_vad_history = []
        
        # 音声開始/終了のコールバック（orchestratorにイベントを通知）
        def on_bot_speech_start(timestamp: float):
            """ボット（サーバー）の音声開始を記録"""
            nonlocal client_vad_timestamps, client_vad_history
            current_time = asyncio.get_event_loop().time()
            
            # 音声開始検出時刻を記録
            if client_vad_timestamps.get("speech_start_detected") is None:
                client_vad_timestamps["speech_start_detected"] = current_time
            
            logger.info(f"[VAD] Bot speech started at {timestamp:.3f}s")
            orchestrator.logger.log_bot_speech_start()
            orchestrator.notify_event("BOT_SPEECH_START")
            
            # イベント記録時刻を記録
            client_vad_timestamps["event_logged"] = current_time
            
            # 現在のタイムスタンプを履歴に保存（コピーして保存）
            history_entry = {
                "first_audio_received": client_vad_timestamps.get("first_audio_received"),
                "first_frame_processed": client_vad_timestamps.get("first_frame_processed"),
                "speech_start_detected": client_vad_timestamps.get("speech_start_detected"),
                "event_logged": client_vad_timestamps.get("event_logged"),
            }
            client_vad_history.append(history_entry)
            
            # タイムスタンプをリセット（次の検出に備える）
            client_vad_timestamps = {
                "first_audio_received": None,
                "first_frame_processed": None,
                "speech_start_detected": None,
                "event_logged": None,
            }
        
        def on_bot_speech_end(timestamp: float):
            """ボット（サーバー）の音声終了を記録"""
            logger.info(f"[VAD] Bot speech ended at {timestamp:.3f}s")
            orchestrator.logger.log_bot_speech_end()
            orchestrator.notify_event("BOT_SPEECH_END")
        
        # VAD検出器を初期化
        vad_detector = VADDetector(
            sample_rate=vad_sample_rate,  # 16000Hzで初期化
            threshold=1.0,  # 閾値を高くして、より厳格に検出
            min_speech_duration_ms=300,  # 最小音声継続時間を長くして、短いノイズを無視
            min_silence_duration_ms=300,  # 無音継続時間を長くして、途中の無音で誤検出しないようにする
            on_speech_start=on_bot_speech_start,
            on_speech_end=on_bot_speech_end
        )
        
        # WebSocketサーバーに接続
        server_url = websocket_config.get("server_url", "ws://localhost:8765/ws")
        logger.info(f"Connecting to WebSocket server: {server_url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(server_url) as ws:
                logger.info("WebSocket接続が確立されました")
                ws_connection = ws
                
                # サーバーからの音声を受信してVADで処理するタスク
                async def receive_audio():
                    """サーバーからの音声を受信してVADで処理"""
                    nonlocal client_vad_timestamps
                    # orchestratorのstart_timeと統一
                    start_time = orchestrator.logger.start_time
                    # VAD用のバッファ（リサンプリング用）
                    vad_buffer = np.array([], dtype=np.int16)
                    vad_chunk_size = 320  # 16000Hzで20ms = 320サンプル
                    # 累積サンプル数（24000Hz）を追跡してタイムスタンプを計算
                    total_samples_received = 0
                    
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
                                
                                # 最初の音声データ受信時刻を記録（最初のチャンクを受信した時点）
                                if client_vad_timestamps["first_audio_received"] is None:
                                    client_vad_timestamps["first_audio_received"] = asyncio.get_event_loop().time()
                                
                                # 受信した音声を記録（録音が有効な場合のみ）
                                if recording_enabled:
                                    recorded_bot_audio.append(audio_chunk.copy())
                                
                                # 累積サンプル数を更新（24000Hz）
                                chunk_samples = len(audio_chunk)
                                total_samples_received += chunk_samples
                                
                                # VADバッファに追加（24000Hz）
                                vad_buffer = np.concatenate([vad_buffer, audio_chunk])
                                
                                # 24000Hzから16000Hzにリサンプリング（320サンプル以上になったら）
                                min_samples_for_resample = int(input_sample_rate * 0.02)  # 20ms分（24000Hzで480サンプル）
                                
                                if len(vad_buffer) >= min_samples_for_resample and has_scipy:
                                    # リサンプリング: 24000Hz → 16000Hz
                                    num_output_samples = int(len(vad_buffer) * vad_sample_rate / input_sample_rate)
                                    resampled_chunk = signal.resample(vad_buffer, num_output_samples).astype(np.int16)
                                    
                                    # VADバッファが一定の長さに達したら処理
                                    processed_samples = 0  # 処理したサンプル数（16000Hz）
                                    while len(resampled_chunk) >= vad_chunk_size:
                                        vad_chunk = resampled_chunk[:vad_chunk_size]
                                        resampled_chunk = resampled_chunk[vad_chunk_size:]
                                        
                                        # タイムスタンプを計算（累積サンプル数から）
                                        # 処理済みサンプル数（24000Hz）を計算
                                        processed_samples_24k = int(processed_samples * input_sample_rate / vad_sample_rate)
                                        # このチャンクの開始時刻（24000Hzでの累積サンプル数から）
                                        chunk_start_samples_24k = total_samples_received - len(vad_buffer) + processed_samples_24k
                                        timestamp = chunk_start_samples_24k / input_sample_rate
                                        
                                        # VADで処理（音声開始を検出する可能性があるフレーム）
                                        is_speaking_before = vad_detector.is_speaking
                                        
                                        # VADで処理
                                        has_speech = vad_detector.process_audio_frame(vad_chunk, timestamp)
                                        
                                        # 音声開始を検出したフレームの処理時刻を記録
                                        if not is_speaking_before and vad_detector.is_speaking:
                                            if client_vad_timestamps["first_frame_processed"] is None:
                                                client_vad_timestamps["first_frame_processed"] = asyncio.get_event_loop().time()
                                        
                                        processed_samples += vad_chunk_size
                                    
                                    # 残りのデータを保持
                                    vad_buffer = resampled_chunk.astype(np.int16) if len(resampled_chunk) > 0 else np.array([], dtype=np.int16)
                                elif not has_scipy:
                                    # scipyがない場合は、そのまま処理（精度は低下）
                                    if len(vad_buffer) >= vad_chunk_size:
                                        vad_chunk = vad_buffer[:vad_chunk_size]
                                        vad_buffer = vad_buffer[vad_chunk_size:]
                                        
                                        # タイムスタンプを計算（累積サンプル数から）
                                        chunk_start_samples_24k = total_samples_received - len(vad_buffer)
                                        timestamp = chunk_start_samples_24k / input_sample_rate
                                        
                                        # VADで処理（音声開始を検出する可能性があるフレーム）
                                        is_speaking_before = vad_detector.is_speaking
                                        
                                        # VADで処理
                                        has_speech = vad_detector.process_audio_frame(vad_chunk, timestamp)
                                        
                                        # 音声開始を検出したフレームの処理時刻を記録
                                        if not is_speaking_before and vad_detector.is_speaking:
                                            if client_vad_timestamps["first_frame_processed"] is None:
                                                client_vad_timestamps["first_frame_processed"] = asyncio.get_event_loop().time()
                                
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
                        pass
                
                # 音声送信用の変数
                user_audio_chunks = None  # 送信する音声チャンク（.convoファイルに従って設定される）
                audio_send_complete_event = asyncio.Event()  # 音声送信完了を通知するイベント
                valid_loop = True  # ループ継続フラグ
                
                # 音声録音用のバッファ
                recorded_user_audio = []  # 送信した音声（ユーザー音声）
                recorded_bot_audio = []  # 受信した音声（ボット音声）
                recording_start_time = asyncio.get_event_loop().time()  # 録音開始時刻
                recording_enabled = True  # 録音有効フラグ
                
                async def audio_sender_task():
                    """音声送信タスク（常に動作し、デフォルトは無音、.convoに従って音声を送信）"""
                    chunk_size = int(input_sample_rate * 0.01)  # 10ms
                    chunk_duration = 0.01  # 10ms
                    silence_chunk = np.zeros(chunk_size, dtype=np.int16)
                    
                    while valid_loop:
                        nonlocal user_audio_chunks
                        
                        # 音声チャンクがある場合は音声を送信、ない場合は無音を送信
                        if user_audio_chunks and len(user_audio_chunks) > 0:
                            # 音声チャンクを送信
                            chunk = user_audio_chunks.pop(0)
                            await ws.send_bytes(chunk.tobytes())
                            # 送信した音声を記録（録音が有効な場合のみ）
                            if recording_enabled:
                                recorded_user_audio.append(chunk.copy())
                            if len(user_audio_chunks) == 0:
                                # すべての音声チャンクを送信完了
                                user_audio_chunks = None
                                audio_send_complete_event.set()  # 送信完了を通知
                        else:
                            # 無音を送信（デフォルト）
                            await ws.send_bytes(silence_chunk.tobytes())
                            # 無音も記録（タイミングを合わせるため、録音が有効な場合のみ）
                            if recording_enabled:
                                recorded_user_audio.append(silence_chunk.copy())
                        
                        await asyncio.sleep(chunk_duration)  # 10ms待機
                
                # 音声受信タスクを開始
                receive_task = asyncio.create_task(receive_audio())
                logger.info("[VAD] Started receiving audio and VAD processing...")
                
                # 音声送信タスクを開始（常に動作し、デフォルトは無音）
                audio_sender = asyncio.create_task(audio_sender_task())
                
                # 音声送信関数（シナリオ実行時に使用）
                async def send_audio_file(audio_file_path: str):
                    """音声ファイルを送信する関数（音声ファイルの長さ分だけ送信）"""
                    nonlocal user_audio_chunks
                    audio_path = Path(audio_file_path)
                    if not audio_path.exists():
                        logger.warning(f"Audio file not found: {audio_file_path}")
                        return
                    
                    try:
                        audio_data, sr = sf.read(audio_path)
                        
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
                        
                        # 音声データをチャンクに分割
                        chunk_size = int(input_sample_rate * 0.01)  # 10ms
                        user_audio_chunks = []
                        for i in range(0, len(audio_data), chunk_size):
                            chunk = audio_data[i:i+chunk_size]
                            # チャンクサイズに満たない場合は0でパディング
                            if len(chunk) < chunk_size:
                                padded_chunk = np.zeros(chunk_size, dtype=np.int16)
                                padded_chunk[:len(chunk)] = chunk
                                chunk = padded_chunk
                            user_audio_chunks.append(chunk)
                        
                        audio_duration = len(audio_data) / input_sample_rate
                        logger.info(f"音声ファイルを送信します: {audio_file_path} ({audio_duration:.2f}秒)")
                        
                        # 音声送信完了イベントをリセット
                        audio_send_complete_event.clear()
                        
                        # 音声送信が完了するまで待機
                        await audio_send_complete_event.wait()
                        
                    except Exception as e:
                        logger.error(f"音声ファイルの読み込みエラー: {e}", exc_info=True)
                
                # シナリオを実行
                await orchestrator.run_scenario(
                    convo_file=str(convo_path),
                    audio_sender=send_audio_file
                )
                
                # シナリオ実行完了後、3秒余裕を持たせてから終了
                # （シナリオ内の最後のWAIT_FOR_BOT_SPEECH_ENDは既に完了している）
                await asyncio.sleep(3.0)  # 3秒待機
                
                # 録音を停止
                recording_enabled = False
                
                # ループフラグを無効化
                valid_loop = False
                
                # 音声送信タスクをキャンセル
                audio_sender.cancel()
                try:
                    await audio_sender
                except asyncio.CancelledError:
                    pass
                
                # 受信タスクをキャンセル
                receive_task.cancel()
                try:
                    await receive_task
                except asyncio.CancelledError:
                    pass
                
                # 録音した音声を2チャンネル（ステレオ）WAVファイルとして保存
                if recorded_user_audio or recorded_bot_audio:
                    try:
                        # 録音データを結合
                        user_audio = np.concatenate(recorded_user_audio) if recorded_user_audio else np.array([], dtype=np.int16)
                        bot_audio = np.concatenate(recorded_bot_audio) if recorded_bot_audio else np.array([], dtype=np.int16)
                        
                        # 長さを揃える（短い方に無音を追加）
                        max_length = max(len(user_audio), len(bot_audio))
                        if len(user_audio) < max_length:
                            silence_padding = np.zeros(max_length - len(user_audio), dtype=np.int16)
                            user_audio = np.concatenate([user_audio, silence_padding])
                        if len(bot_audio) < max_length:
                            silence_padding = np.zeros(max_length - len(bot_audio), dtype=np.int16)
                            bot_audio = np.concatenate([bot_audio, silence_padding])
                        
                        # 2チャンネル（ステレオ）に結合: 左チャンネル=ユーザー音声、右チャンネル=ボット音声
                        stereo_audio = np.column_stack([user_audio, bot_audio])
                        
                        # float32形式に変換（-1.0～1.0の範囲）
                        stereo_audio_float = stereo_audio.astype(np.float32) / 32767.0
                        
                        # WAVファイルとして保存
                        output_dir = Path(config.get("logging", {}).get("output_dir", "data/reports"))
                        output_dir.mkdir(parents=True, exist_ok=True)
                        output_file = output_dir / f"{test_id}_recording.wav"
                        sf.write(str(output_file), stereo_audio_float, input_sample_rate)
                        
                        logger.info(f"録音を保存しました: {output_file}")
                        logger.info(f"  左チャンネル（ユーザー音声）: {len(user_audio)/input_sample_rate:.2f}秒")
                        logger.info(f"  右チャンネル（ボット音声）: {len(bot_audio)/input_sample_rate:.2f}秒")
                        
                    except Exception as e:
                        logger.error(f"録音の保存エラー: {e}", exc_info=True)
                
                # 評価処理（STTと評価指標の計算）
                if recorded_user_audio or recorded_bot_audio:
                    try:
                        user_audio = np.concatenate(recorded_user_audio) if recorded_user_audio else np.array([], dtype=np.int16)
                        bot_audio = np.concatenate(recorded_bot_audio) if recorded_bot_audio else np.array([], dtype=np.int16)
                        
                        # STTで音声を認識
                        stt_results = evaluator.evaluate_recording(
                            user_audio=user_audio,
                            bot_audio=bot_audio,
                            sample_rate=input_sample_rate,
                            logger_instance=orchestrator.logger
                        )
                        
                        # 評価指標を計算
                        metrics = evaluator.calculate_metrics(orchestrator.logger)
                        
                        logger.info("=" * 60)
                        logger.info("評価結果:")
                        logger.info(f"  Response Latency: {metrics.get('response_latency_ms')}ms (閾値: {config.get('evaluation', {}).get('response_latency_threshold_ms', 800)}ms) {'✓' if metrics.get('response_latency_ok') else '✗'}")
                        logger.info(f"  Think Time: {metrics.get('think_time_ms')}ms (閾値: {config.get('evaluation', {}).get('think_time_threshold_ms', 500)}ms) {'✓' if metrics.get('think_time_ok') else '✗'}")
                        logger.info(f"  User Speech Duration: {metrics.get('user_speech_duration_ms')}ms")
                        logger.info(f"  Bot Speech Duration: {metrics.get('bot_speech_duration_ms')}ms")
                        logger.info("=" * 60)
                        
                    except Exception as e:
                        logger.error(f"評価処理エラー: {e}", exc_info=True)
        
        # 応答時間分析を出力（タイムラインから計算）
        events = orchestrator.logger.events
        user_speech_ends = [e for e in events if e.get("type") == "USER_SPEECH_END"]
        bot_speech_starts = [e for e in events if e.get("type") == "BOT_SPEECH_START"]
        
        if user_speech_ends and bot_speech_starts:
            # 各USER_SPEECH_ENDに対応するBOT_SPEECH_STARTを探して応答時間を計算
            response_latencies = []
            bot_idx = 0
            for user_end in user_speech_ends:
                user_end_time = user_end.get("time", 0)
                
                # 対応するBOT_SPEECH_STARTを探す（USER_SPEECH_ENDの後に最初に来るもの）
                for j in range(bot_idx, len(bot_speech_starts)):
                    if bot_speech_starts[j].get("time", 0) > user_end_time:
                        bot_start_time = bot_speech_starts[j].get("time", 0)
                        response_latency = bot_start_time - user_end_time
                        response_latencies.append(response_latency)
                        bot_idx = j + 1
                        break
            
            if response_latencies:
                avg_response_latency = sum(response_latencies) / len(response_latencies)
                logger.info("=" * 60)
                logger.info("応答時間分析（Response Latency）:")
                logger.info(f"  平均応答時間: {avg_response_latency:.2f}ms (サンプル数: {len(response_latencies)})")
                logger.info("=" * 60)
        
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
