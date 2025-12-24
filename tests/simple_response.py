"""WebSocketサーバー（VADを使用したモック対話システム）
クライアントからの音声を受信し、VADで音声終了を検出したら応答音声を送信
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

import numpy as np
import soundfile as sf
from aiohttp import web, WSMsgType
from tester.vad_detector import VADDetector

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
    
    # VAD検出器を初期化
    input_sample_rate = 24000  # WebSocketで受信する音声のサンプルレート（OpenAI Realtime API標準）
    vad_sample_rate = 16000  # silero-vadが期待するサンプルレート
    response_sent = False
    should_send_response = False
    
    # VAD検出のコールバック
    def on_speech_start(timestamp: float):
        """音声開始を検出（立ち上がり）"""
        print(f"[VAD] 立ち上がり: {timestamp:.3f}s")
    
    def on_speech_end(timestamp: float):
        """音声終了を検出（立ち下がり）"""
        nonlocal should_send_response
        print(f"[VAD] 立ち下がり: {timestamp:.3f}s")
        should_send_response = True
    
    vad_detector = VADDetector(
        sample_rate=vad_sample_rate,  # 16000Hzで初期化
        threshold=0.2,  # 閾値をさらに下げる（0.3 → 0.2）
        min_speech_duration_ms=250,
        min_silence_duration_ms=100,
        on_speech_start=on_speech_start,
        on_speech_end=on_speech_end
    )
    
    # VADモデルの状態を確認
    if vad_detector.vad is None:
        print("[WARNING] VADモデルがロードされていません。フォールバックVADを使用します。")
    else:
        print("[INFO] VADモデルが正常にロードされました。")
    
    # 受信した音声データを保存（デバッグ用）
    received_audio_data = []
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
                
                # デバッグログは削除（必要に応じて有効化）
                
                received_audio_data.append(audio_chunk)
                
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
                
                # 応答音声を送信する必要があるかチェック
                if should_send_response and not response_sent:
                    response_sent = True
                    response_audio_file = "tests/hello_48k.wav"
                    if os.path.exists(response_audio_file):
                        print(f"応答音声を送信します: {response_audio_file}")
                        
                        # 音声ファイルを読み込んで送信
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
                                print(f"サンプルレートを{sr}Hzから{input_sample_rate}Hzにリサンプリングしました")
                            else:
                                print(f"[WARNING] scipyがインストールされていないため、リサンプリングをスキップします。")
                        
                        # チャンクに分けて送信（10msごと）
                        chunk_size = int(input_sample_rate * 0.01)  # 10ms
                        for i in range(0, len(audio_data), chunk_size):
                            chunk = audio_data[i:i+chunk_size]
                            await ws.send_bytes(chunk.tobytes())
                            await asyncio.sleep(0.01)  # 10ms待機
                        
                        print("応答音声の送信が完了しました")
                    else:
                        print(f"応答音声ファイルが見つかりません: {response_audio_file}")
                                
            elif msg.type == WSMsgType.ERROR:
                print(f"WebSocketエラー: {ws.exception()}")
                break
            elif msg.type == WSMsgType.CLOSE:
                print("WebSocket接続が閉じられました")
                break
                
    except Exception as e:
        print(f"WebSocket接続エラー: {e}")
    finally:
        # ループフラグを無効化
        valid_loop = False
        
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
        
        # 受信した音声を保存（デバッグ用）
        if received_audio_data:
            all_received = np.concatenate(received_audio_data)
            output_file = "server_received.wav"
            sf.write(output_file, all_received.astype(np.float32) / 32767.0, input_sample_rate)
            print(f"受信した音声を保存しました: {output_file}")
    
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

