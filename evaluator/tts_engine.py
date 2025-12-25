"""TTS（Text-to-Speech）エンジン"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path
import tempfile
import os

logger = logging.getLogger(__name__)


class TTSEngine:
    """TTSエンジンの基底クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.language = config.get("language", "ja")
        self.sample_rate = config.get("sample_rate", 24000)
    
    def synthesize(self, text: str, output_path: str) -> bool:
        """
        テキストを音声ファイルに変換します。
        
        Args:
            text: 変換するテキスト
            output_path: 出力先の音声ファイルパス
            
        Returns:
            成功した場合True、失敗した場合False
        """
        raise NotImplementedError


class GTTSEngine(TTSEngine):
    """gTTS（Google Text-to-Speech）を使用したTTSエンジン"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        try:
            from gtts import gTTS
            self.gTTS = gTTS
            logger.debug("[TTSEngine] GTTSEngine initialized.")
        except ImportError:
            logger.warning("[TTSEngine] gttsがインストールされていません。pip install gttsでインストールしてください。")
            self.gTTS = None
    
    def synthesize(self, text: str, output_path: str) -> bool:
        """gTTSでテキストを音声ファイルに変換"""
        if self.gTTS is None:
            logger.error("[TTSEngine] gTTS is not available.")
            return False
        
        try:
            # gTTSで音声を生成（MP3形式）
            tts = self.gTTS(text=text, lang=self.language, slow=False)
            
            # 一時ファイルに保存（MP3）
            temp_mp3 = output_path.replace('.wav', '.mp3')
            tts.save(temp_mp3)
            
            # MP3をWAVに変換（pydubを使用）
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_mp3(temp_mp3)
                # サンプルレートを変換
                if self.sample_rate != audio.frame_rate:
                    audio = audio.set_frame_rate(self.sample_rate)
                # WAV形式で保存
                audio.export(output_path, format="wav")
                # 一時MP3ファイルを削除
                os.remove(temp_mp3)
                logger.debug(f"[TTSEngine] Synthesized audio saved to: {output_path}")
                return True
            except ImportError:
                logger.warning("[TTSEngine] pydubがインストールされていません。MP3ファイルのまま保存します。")
                # MP3ファイルをそのまま使用（main.pyで対応が必要）
                return True
            except Exception as e:
                logger.error(f"[TTSEngine] Error converting MP3 to WAV: {e}", exc_info=True)
                return False
                
        except Exception as e:
            logger.error(f"[TTSEngine] Error synthesizing speech: {e}", exc_info=True)
            return False


def create_tts_engine(config: Dict[str, Any]) -> Optional[TTSEngine]:
    """設定に基づいてTTSエンジンを作成"""
    tts_config = config.get("tts", {})
    engine_type = tts_config.get("engine", "gtts").lower()
    
    if engine_type == "gtts":
        return GTTSEngine(tts_config)
    else:
        logger.warning(f"不明なTTSエンジンタイプ: {engine_type}。TTS機能は無効です。")
        return None

