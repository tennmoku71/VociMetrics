"""WebSocketサーバー（モック対話システム）
クライアントからの音声を受信し、音声終了を検出したら応答音声を送信
"""

import asyncio
import json
import os
import signal
import sys
import numpy as np
import soundfile as sf
from aiohttp import web, WSMsgType
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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
    
    # 音声検出の状態管理
    audio_detected = False
    audio_ended = False
    response_sent = False
    
    # 受信した音声データを保存
    received_audio_data = []
    sample_rate = 24000  # OpenAI Realtime API標準のサンプルレート
    
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
                received_audio_data.append(audio_chunk)
                
                # 最新の500ms分の音声強度を計算
                recent_samples = int(sample_rate * 0.5)  # 500ms
                all_audio = np.concatenate(received_audio_data) if received_audio_data else np.array([], dtype=np.int16)
                
                if len(all_audio) > recent_samples:
                    recent_audio = all_audio[-recent_samples:]
                else:
                    recent_audio = all_audio
                
                # 音声強度（RMS）を計算
                if len(recent_audio) > 0:
                    audio_float = recent_audio.astype(np.float32) / 32767.0
                    rms = np.sqrt(np.mean(audio_float ** 2))
                    amplitude = rms * 32767
                    
                    # 音声が検出されたかチェック（閾値は適宜調整）
                    if amplitude > 100:  # 閾値を100に設定
                        if not audio_detected:
                            print(f"音声検出: 強度={amplitude:.0f}")
                            audio_detected = True
                            audio_ended = False
                    else:
                        # 音声が終了したかチェック
                        if audio_detected and not audio_ended:
                            # 少し待ってから再確認（誤検出を防ぐ）
                            await asyncio.sleep(0.2)
                            
                            # 再度チェック
                            if len(all_audio) > recent_samples:
                                recent_audio_check = all_audio[-recent_samples:]
                            else:
                                recent_audio_check = all_audio
                            
                            if len(recent_audio_check) > 0:
                                audio_float_check = recent_audio_check.astype(np.float32) / 32767.0
                                rms_check = np.sqrt(np.mean(audio_float_check ** 2))
                                amplitude_check = rms_check * 32767
                                
                                if amplitude_check <= 100:
                                    print(f"音声終了: 強度={amplitude_check:.0f}")
                                    audio_ended = True
                                    
                                    # 応答音声を送信
                                    if not response_sent:
                                        response_sent = True
                                        response_audio_file = "tests/hello_48k.wav"
                                        
                                        # 音声ファイルが存在しない場合、TTSで自動生成
                                        if not os.path.exists(response_audio_file):
                                            if tts_engine:
                                                print(f"[TTS] 音声ファイルが見つかりません。TTSで自動生成します: {response_audio_file}")
                                                default_text = "こんにちは。"
                                                # ディレクトリが存在しない場合は作成
                                                os.makedirs(os.path.dirname(response_audio_file), exist_ok=True)
                                                if tts_engine.synthesize(default_text, response_audio_file):
                                                    print(f"[TTS] 音声ファイルを生成しました: {response_audio_file}")
                                                else:
                                                    print(f"[ERROR] TTSでの音声生成に失敗しました。")
                                                    continue
                                            else:
                                                print(f"[WARNING] 音声ファイルが見つかりません: {response_audio_file} (TTSエンジンも利用できません)")
                                                continue
                                        
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
                                            if sr != sample_rate:
                                                try:
                                                    from scipy import signal
                                                    num_samples = int(len(audio_data) * sample_rate / sr)
                                                    audio_data = signal.resample(audio_data, num_samples).astype(np.int16)
                                                    print(f"サンプルレートを{sr}Hzから{sample_rate}Hzにリサンプリングしました")
                                                except ImportError:
                                                    print(f"[WARNING] scipyがインストールされていないため、リサンプリングをスキップします。")
                                            
                                            # チャンクに分けて送信（10msごと）
                                            chunk_size = int(sample_rate * 0.01)  # 10ms
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
        
        # 受信した音声を保存（デバッグ用）
        if received_audio_data:
            all_received = np.concatenate(received_audio_data)
            output_file = "server_received.wav"
            sf.write(output_file, all_received.astype(np.float32) / 32767.0, sample_rate)
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
