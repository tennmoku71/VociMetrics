"""WebSocketサーバー（VADを使用したモック対話システム）
クライアントからの音声を受信し、VADで音声終了を検出したら応答音声を送信

応答時間について:
固定値として決まっている要素のみを使用した理論値（正値）:
- サーバー側: min_silence_duration_ms=800ms（無音継続時間）
  - ユーザーが話し終わってから、VADが無音を検知するまでに800msの無音が必要
- クライアント側: frame_duration_ms=20ms（VADフレーム長）
  - 20ms分の音声データが溜まるまでの時間（main.py側のVAD検知遅延）

理論値（正値）: 800ms + 20ms = 820ms

"""

import asyncio
import json
import os
import signal
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
import soundfile as sf
from aiohttp import web, WSMsgType
from tester.vad_detector import VADDetector
from evaluator.tts_engine import create_tts_engine

# グローバル変数でシャットダウンフラグを管理
shutdown_event = asyncio.Event()
shutdown_requested = False


def signal_handler(signum, frame):
    """シグナルハンドラー（CTRL+C）"""
    global shutdown_requested
    if not shutdown_requested:
        shutdown_requested = True
        print("\nシャットダウンシグナルを受信しました。安全に終了します...")
        shutdown_event.set()
    else:
        # 2回目のCTRL+Cで強制終了
        print("\n強制終了します...")
        os._exit(1)


async def websocket_handler(request):
    """WebSocketハンドラー"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    print("クライアントが接続しました")
    
    # config.jsonを読み込んでTTSエンジンを初期化
    config_path = project_root / "config.json"
    tts_engine = None
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            tts_engine = create_tts_engine(config)
        except Exception as e:
            print(f"[WARNING] TTSエンジンの初期化に失敗しました: {e}")
    
    # VAD検出器を初期化
    input_sample_rate = 24000  # WebSocketで受信する音声のサンプルレート（OpenAI Realtime API標準）
    vad_sample_rate = 16000  # webrtcvadが期待するサンプルレート
    should_send_response = False
    
    # 応答音声データ（立ち下がり検出時に設定される）
    response_audio_chunks = None
    
    async def audio_sender_task():
        """音声送信タスク（常に動作し、デフォルトは無音、条件満たすと音声を送信）"""
        chunk_size = int(input_sample_rate * 0.01)  # 10ms
        chunk_duration = 0.01  # 10ms
        silence_chunk = np.zeros(chunk_size, dtype=np.int16)
        
        while valid_loop:
            nonlocal response_audio_chunks, should_send_response
            
            # 接続が閉じられている場合はループを終了
            if ws.closed:
                break
            
            # 応答音声を送信する必要がある場合
            if should_send_response and response_audio_chunks is None:
                # 応答音声ファイルを読み込む
                response_audio_file = "tests/response.wav"
                
                # 音声ファイルが存在しない場合、TTSで自動生成
                if not os.path.exists(response_audio_file):
                    if tts_engine:
                        print(f"[TTS] 音声ファイルが見つかりません。TTSで自動生成します: {response_audio_file}")
                        default_text = "はい、承知いたしました。"
                        # ディレクトリが存在しない場合は作成
                        os.makedirs(os.path.dirname(response_audio_file), exist_ok=True)
                        if tts_engine.synthesize(default_text, response_audio_file):
                            print(f"[TTS] 音声ファイルを生成しました: {response_audio_file}")
                        else:
                            print(f"[ERROR] TTSでの音声生成に失敗しました。無音を送信します。")
                            should_send_response = False
                            continue
                    else:
                        print(f"[WARNING] 音声ファイルが見つかりません: {response_audio_file} (TTSエンジンも利用できません)")
                        should_send_response = False
                        continue
                
                if os.path.exists(response_audio_file):
                    # 音声ファイルを読み込む
                    audio_data, sr = sf.read(response_audio_file)
                    
                    # モノラルに変換
                    if len(audio_data.shape) > 1:
                        audio_data = np.mean(audio_data, axis=1)
                    
                    # int16に変換
                    if audio_data.dtype != np.int16:
                        audio_data = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
                    
                    # サンプルレートを24000Hzにリサンプリング（必要に応じて）
                    if sr != input_sample_rate:
                        if has_scipy:
                            num_samples = int(len(audio_data) * input_sample_rate / sr)
                            audio_data = signal.resample(audio_data, num_samples).astype(np.int16)
                        else:
                            pass
                    
                    # 音声データをチャンクに分割
                    response_audio_chunks = []
                    for i in range(0, len(audio_data), chunk_size):
                        chunk = audio_data[i:i+chunk_size]
                        # チャンクサイズに満たない場合は0でパディング
                        if len(chunk) < chunk_size:
                            padded_chunk = np.zeros(chunk_size, dtype=np.int16)
                            padded_chunk[:len(chunk)] = chunk
                            chunk = padded_chunk
                        response_audio_chunks.append(chunk)
                    
                    audio_duration = len(audio_data) / input_sample_rate
                    current_time = asyncio.get_event_loop().time()
                    relative_time = current_time - start_time
                    print(f"[応答] 音声送信開始: {relative_time:.3f}s (音声: {audio_duration:.2f}秒)")
                    should_send_response = False  # フラグをリセット
            
            # 応答音声チャンクがある場合は音声を送信、ない場合は無音を送信
            try:
                if response_audio_chunks and len(response_audio_chunks) > 0:
                    # 音声チャンクを送信
                    chunk = response_audio_chunks.pop(0)
                    if not ws.closed:
                        await ws.send_bytes(chunk.tobytes())
                    if len(response_audio_chunks) == 0:
                        # すべての音声チャンクを送信完了
                        current_time = asyncio.get_event_loop().time()
                        relative_time = current_time - start_time
                        print(f"[応答] 音声送信終了: {relative_time:.3f}s")
                        response_audio_chunks = None
                else:
                    # 無音を送信（デフォルト）
                    if not ws.closed:
                        await ws.send_bytes(silence_chunk.tobytes())
            except (ConnectionError, BrokenPipeError, OSError) as e:
                # 接続が切断された場合はループを終了
                print(f"[INFO] 接続が切断されました（音声送信タスク）: {e}")
                break
            except Exception as e:
                # その他のエラーはログに記録して続行
                print(f"[WARNING] 音声送信エラー: {e}")
            
            await asyncio.sleep(chunk_duration)  # 10ms待機
    
    # VAD検出のコールバック
    def on_speech_start(timestamp: float):
        """音声開始を検出（立ち上がり）"""
        current_time = asyncio.get_event_loop().time()
        relative_time = current_time - start_time
        print(f"[VAD] 立ち上がり: {timestamp:.3f}s (現在時刻: {relative_time:.3f}s)")
    
    def on_speech_end(timestamp: float):
        """音声終了を検出（立ち下がり）"""
        nonlocal should_send_response, response_audio_chunks
        current_time = asyncio.get_event_loop().time()
        relative_time = current_time - start_time
        # VADのis_speaking状態を確認（Falseの時のみ応答を開始）
        if vad_detector.is_speaking:
            print(f"[VAD] 立ち下がり: {timestamp:.3f}s (現在時刻: {relative_time:.3f}s, is_speaking=Trueのため無視)")
            return
        # 応答音声送信中は無視
        if response_audio_chunks is not None and len(response_audio_chunks) > 0:
            print(f"[VAD] 立ち下がり: {timestamp:.3f}s (現在時刻: {relative_time:.3f}s, 応答送信中のため無視)")
            return
        print(f"[VAD] 立ち下がり: {timestamp:.3f}s (現在時刻: {relative_time:.3f}s, is_speaking=False、応答開始)")
        should_send_response = True
    
    vad_detector = VADDetector(
        sample_rate=vad_sample_rate,  # 16000Hzで初期化
        threshold=1.0,  # 閾値を高くして、より厳格に検出
        min_speech_duration_ms=300,  # 最小音声継続時間を長くして、短いノイズを無視
        min_silence_duration_ms=0,  # 遅延を0にして、割り込み検出を確実にする
        on_speech_start=on_speech_start,
        on_speech_end=on_speech_end
    )
    
    # VADモデルの状態を確認
    if vad_detector.vad is None:
        print("[WARNING] VADモデルがロードされていません。フォールバックVADを使用します。")
    else:
        print("[INFO] VADモデルが正常にロードされました。")
    
    start_time = asyncio.get_event_loop().time()
    
    # VAD用のバッファ（一定の長さのチャンクにまとめる）
    # webrtcvadは10ms、20ms、30msのフレーム長のみをサポート
    # 20msフレームを使用（16000Hzで320サンプル）
    vad_chunk_size = 320  # VAD処理用のチャンクサイズ（サンプル数、16000Hz、20ms）
    vad_buffer = np.array([], dtype=np.int16)  # リサンプリング後のデータを保持
    
    # リサンプリング用の一時バッファ（48000Hzのデータを蓄積）
    resample_buffer = np.array([], dtype=np.int16)
    
    # scipyのインポート（リサンプリング用）
    try:
        from scipy import signal
        has_scipy = True
    except ImportError:
        has_scipy = False
        print("[WARNING] scipyがインストールされていません。VADの精度が低下する可能性があります。")
    
    # ループ継続フラグ
    valid_loop = True
    
    # シャットダウン監視タスク
    async def monitor_shutdown():
        """シャットダウンシグナルを監視"""
        await shutdown_event.wait()
        nonlocal valid_loop
        valid_loop = False
        print("シャットダウンシグナルを受信しました。接続を閉じます...")
        if not ws.closed:
            await ws.close()
    
    shutdown_task = asyncio.create_task(monitor_shutdown())
    
    # 音声送信タスクを開始（常に動作し、デフォルトは無音）
    audio_sender = asyncio.create_task(audio_sender_task())
    
    try:
        async for msg in ws:
            # ループ継続フラグをチェック
            if not valid_loop:
                break
                
            if msg.type == WSMsgType.TEXT:
                # テキストメッセージ（制御用）
                try:
                    data = json.loads(msg.data)
                    print(f"受信メッセージ: {data}")
                except json.JSONDecodeError:
                    print(f"無効なJSONメッセージ: {msg.data}")
                    
            elif msg.type == WSMsgType.BINARY:
                # バイナリメッセージ（音声データ）
                # PCM形式（int16）を想定
                audio_chunk = np.frombuffer(msg.data, dtype=np.int16)
                
                # リサンプリング用バッファに追加（24000Hz）
                resample_buffer = np.concatenate([resample_buffer, audio_chunk])
                
                # タイムスタンプを計算（相対時間）
                current_time = asyncio.get_event_loop().time()
                relative_time = current_time - start_time
                
                # 24000Hzから16000Hzにリサンプリング（十分なデータが蓄積されたら）
                # リサンプリング効率を上げるため、ある程度まとめて処理
                min_samples_for_resample = int(input_sample_rate * 0.02)  # 20ms分（24000Hzで480サンプル）
                
                if len(resample_buffer) >= min_samples_for_resample and has_scipy:
                    # リサンプリング: 24000Hz → 16000Hz
                    num_output_samples = int(len(resample_buffer) * vad_sample_rate / input_sample_rate)
                    resampled_chunk = signal.resample(resample_buffer, num_output_samples).astype(np.int16)
                    
                    # VADバッファに追加（16000Hz）
                    vad_buffer = np.concatenate([vad_buffer, resampled_chunk])
                    
                    # リサンプリングバッファをクリア
                    resample_buffer = np.array([], dtype=np.int16)
                    
                    # VADバッファが一定の長さに達したら処理
                    while len(vad_buffer) >= vad_chunk_size:
                        # VAD処理用のチャンクを取得（16000Hz）
                        vad_chunk = vad_buffer[:vad_chunk_size]
                        vad_buffer = vad_buffer[vad_chunk_size:]
                        
                        # VADで処理（立ち上がり/立ち下がりはコールバックで出力）
                        vad_detector.process_audio_frame(vad_chunk, relative_time)
                elif not has_scipy:
                    # scipyがない場合は、そのまま処理（精度は低下）
                    vad_buffer = np.concatenate([vad_buffer, audio_chunk])
                    
                    # VADバッファが一定の長さに達したら処理
                    while len(vad_buffer) >= vad_chunk_size:
                        # VAD処理用のチャンクを取得
                        vad_chunk = vad_buffer[:vad_chunk_size]
                        vad_buffer = vad_buffer[vad_chunk_size:]
                        
                        # VADで処理（立ち上がり/立ち下がりはコールバックで出力）
                        vad_detector.process_audio_frame(vad_chunk, relative_time)
                
                # should_send_responseフラグはaudio_sender_task内で処理される
                # ここでは何もしない（VADコールバックでフラグを設定するだけ）
                                
            elif msg.type == WSMsgType.ERROR:
                print(f"WebSocketエラー: {ws.exception()}")
                break
            elif msg.type == WSMsgType.CLOSE:
                print("WebSocket接続が閉じられました")
                break
                
    except (ConnectionError, BrokenPipeError, OSError) as e:
        # 接続が切断された場合のエラー（正常な切断も含む）
        if not ws.closed:
            print(f"WebSocket接続が切断されました: {e}")
        else:
            print("WebSocket接続が正常に閉じられました")
    except Exception as e:
        print(f"WebSocket接続エラー: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # ループフラグを無効化
        valid_loop = False
        
        # 音声送信タスクをキャンセル
        if 'audio_sender' in locals() and audio_sender is not None and not audio_sender.done():
            audio_sender.cancel()
            try:
                await audio_sender
            except asyncio.CancelledError:
                pass
        
        # シャットダウン監視タスクをキャンセル
        shutdown_task.cancel()
        try:
            await shutdown_task
        except asyncio.CancelledError:
            pass
        
        print("クライアントが切断しました")
        
        # 残りのリサンプリングバッファを処理
        if len(resample_buffer) > 0 and has_scipy:
            num_output_samples = int(len(resample_buffer) * vad_sample_rate / input_sample_rate)
            if num_output_samples > 0:
                resampled_chunk = signal.resample(resample_buffer, num_output_samples).astype(np.int16)
                vad_buffer = np.concatenate([vad_buffer, resampled_chunk])
        
        # 残りのVADバッファを処理
        if len(vad_buffer) > 0:
            final_time = asyncio.get_event_loop().time()
            final_relative_time = final_time - start_time
            vad_detector.process_audio_frame(vad_buffer, final_relative_time)
        
    return ws


async def init_app():
    """アプリケーションを初期化"""
    app = web.Application()
    app.router.add_get('/ws', websocket_handler)
    return app


async def run():
    """WebSocketサーバーを起動"""
    # シグナルハンドラーを設定
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    app = await init_app()
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, 'localhost', 8765)
    
    try:
        await site.start()
        print("WebSocketサーバーを起動しました: ws://localhost:8765/ws")
        print("(CTRL+Cで終了)")
        
        # シャットダウンシグナルまで待機（定期的にチェック）
        while not shutdown_event.is_set():
            try:
                # 0.1秒ごとにチェック
                await asyncio.wait_for(shutdown_event.wait(), timeout=0.1)
                break
            except asyncio.TimeoutError:
                # タイムアウトは正常（継続してチェック）
                continue
        
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"エラー: ポート8765は既に使用されています。")
            print("既存のサーバープロセスを終了してから再試行してください。")
            sys.exit(1)
        else:
            print(f"エラーが発生しました: {e}")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nキーボード割り込みを受信しました。安全に終了します...")
        shutdown_event.set()
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        sys.exit(1)
    finally:
        print("WebSocketサーバーを終了しています...")
        try:
            # クリーンアップ（タイムアウト付き）
            await asyncio.wait_for(runner.cleanup(), timeout=2.0)
        except asyncio.TimeoutError:
            print("クリーンアップがタイムアウトしました。強制終了します...")
            os._exit(1)
        except Exception as e:
            print(f"クリーンアップ中にエラーが発生しました: {e}")
        print("WebSocketサーバーを終了しました")


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nプログラムを終了します。")
        sys.exit(0)
    except Exception as e:
        print(f"予期しないエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

