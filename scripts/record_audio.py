"""マイクから音声を録音するシンプルなスクリプト"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import logging
from pathlib import Path
import argparse
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def record_audio(duration: float = 10.0, sample_rate: int = 16000, output_file: str = None):
    """マイクから音声を録音
    
    Args:
        duration: 録音時間（秒）
        sample_rate: サンプリングレート（Hz）
        output_file: 出力ファイルパス（指定しない場合は自動生成）
    """
    # 出力ファイルのパスを決定
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("data/reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = str(output_dir / f"recorded_{timestamp}.wav")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 利用可能なオーディオデバイスを表示
    logger.info("=" * 60)
    logger.info("Audio Recording")
    logger.info("=" * 60)
    devices = sd.query_devices()
    default_input = sd.default.device[0]
    logger.info(f"Default input device: {devices[default_input]['name']}")
    logger.info(f"Sample rate: {sample_rate} Hz")
    logger.info(f"Duration: {duration} seconds")
    logger.info("=" * 60)
    
    # 録音開始
    logger.info(f"Recording... (speak for {duration} seconds)")
    
    try:
        # 音声を録音
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32
        )
        sd.wait()  # 録音完了まで待機
        
        logger.info("Recording completed!")
        
        # 音声ファイルとして保存
        sf.write(str(output_path), audio_data, sample_rate)
        logger.info(f"Audio saved to: {output_path}")
        
        # 統計情報を表示
        max_amplitude = np.max(np.abs(audio_data))
        rms = np.sqrt(np.mean(audio_data ** 2))
        logger.info(f"Audio stats: max_amplitude={max_amplitude:.4f}, rms={rms:.4f}")
        
        return str(output_path)
        
    except KeyboardInterrupt:
        logger.info("\nRecording interrupted by user")
        return None
    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        raise


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(
        description="Record audio from microphone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 10秒間録音（デフォルト）
  python scripts/record_audio.py
  
  # 5秒間録音
  python scripts/record_audio.py --duration 5
  
  # 出力ファイルを指定
  python scripts/record_audio.py --duration 10 --output my_recording.wav
        """
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=10.0,
        help="Recording duration in seconds (default: 10.0)"
    )
    parser.add_argument(
        "--sample-rate", "-r",
        type=int,
        default=16000,
        help="Sample rate in Hz (default: 16000)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    record_audio(
        duration=args.duration,
        sample_rate=args.sample_rate,
        output_file=args.output
    )


if __name__ == "__main__":
    main()

