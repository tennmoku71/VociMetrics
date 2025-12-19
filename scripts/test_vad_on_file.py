"""録音ファイルに対してVAD検出をテストするスクリプト"""

import numpy as np
import soundfile as sf
import logging
from pathlib import Path
import argparse
import sys

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from tester.vad_detector import VADDetector
from tester.orchestrator import UnifiedLogger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_vad_on_file(audio_file: str, sample_rate: int = 16000):
    """録音ファイルに対してVAD検出を実行
    
    Args:
        audio_file: 音声ファイルのパス
        sample_rate: サンプリングレート（Hz）
    """
    audio_path = Path(audio_file)
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_file}")
        return
    
    logger.info("=" * 60)
    logger.info("VAD Detection Test")
    logger.info("=" * 60)
    logger.info(f"Audio file: {audio_file}")
    
    # 音声ファイルを読み込み
    logger.info("Loading audio file...")
    audio_data, file_sr = sf.read(str(audio_path))
    
    # サンプリングレートを変換（必要に応じて）
    if file_sr != sample_rate:
        logger.info(f"Resampling from {file_sr} Hz to {sample_rate} Hz...")
        from scipy import signal
        num_samples = int(len(audio_data) * sample_rate / file_sr)
        audio_data = signal.resample(audio_data, num_samples)
    
    # モノラルに変換
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # int16に変換
    if audio_data.dtype != np.int16:
        audio_data = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
    
    logger.info(f"Audio loaded: {len(audio_data)} samples, duration: {len(audio_data)/sample_rate:.2f}s")
    
    # VAD検出器を初期化
    speech_events = []
    
    def on_speech_start(timestamp: float):
        logger.info(f"🎤 Speech started at {timestamp:.3f}s")
        speech_events.append({"time": timestamp, "type": "SPEECH_START"})
    
    def on_speech_end(timestamp: float):
        logger.info(f"🔇 Speech ended at {timestamp:.3f}s")
        speech_events.append({"time": timestamp, "type": "SPEECH_END"})
    
    vad_detector = VADDetector(
        sample_rate=sample_rate,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100,
        on_speech_start=on_speech_start,
        on_speech_end=on_speech_end
    )
    
    # ロガーを初期化
    logger_instance = UnifiedLogger(output_dir="data/reports")
    logger_instance.start_test("vad_file_test", test_type="rule")
    
    # VADで処理
    logger.info("Processing audio with VAD...")
    frame_duration_ms = 30  # 30msフレーム
    frame_size = int(sample_rate * frame_duration_ms / 1000)
    total_frames = len(audio_data) // frame_size
    
    logger.info(f"Processing {total_frames} frames...")
    
    for i in range(total_frames):
        start_idx = i * frame_size
        end_idx = min(start_idx + frame_size, len(audio_data))
        frame = audio_data[start_idx:end_idx]
        
        # タイムスタンプを計算
        timestamp = (i * frame_duration_ms) / 1000.0
        
        # VADで検出
        has_speech = vad_detector.process_audio_frame(frame, timestamp)
        
        # ロガーに記録
        if has_speech and not vad_detector.is_speaking:
            logger_instance.log_bot_speech_start()
        elif not has_speech and vad_detector.is_speaking:
            logger_instance.log_bot_speech_end()
    
    # タイムラインを保存
    timeline_path = logger_instance.save_timeline()
    
    # 結果サマリー
    logger.info("=" * 60)
    logger.info("VAD Detection Results:")
    logger.info("=" * 60)
    if speech_events:
        for event in speech_events:
            logger.info(f"  {event['type']} at {event['time']:.3f}s")
    else:
        logger.info("  No speech detected")
    logger.info("=" * 60)
    logger.info(f"Timeline saved to: {timeline_path}")


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="Test VAD on recorded audio file")
    parser.add_argument("audio_file", type=str, help="Audio file path")
    parser.add_argument("--sample-rate", "-r", type=int, default=16000, help="Sample rate (default: 16000)")
    
    args = parser.parse_args()
    
    test_vad_on_file(args.audio_file, args.sample_rate)


if __name__ == "__main__":
    main()

