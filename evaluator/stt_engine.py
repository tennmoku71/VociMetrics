"""STT (Speech-to-Text) エンジンの抽象基底クラスと実装"""

import logging
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import soundfile as sf
from pathlib import Path
import tempfile
import os

logger = logging.getLogger(__name__)


class STTEngine(ABC):
    """STTエンジンの抽象基底クラス"""
    
    @abstractmethod
    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> Optional[str]:
        """音声データをテキストに変換
        
        Args:
            audio_data: 音声データ（numpy配列、float32またはint16）
            sample_rate: サンプリングレート（Hz）
            
        Returns:
            認識されたテキスト（認識失敗時はNone）
        """
        pass
    
    @abstractmethod
    def transcribe_file(self, audio_file: str) -> Optional[str]:
        """音声ファイルをテキストに変換
        
        Args:
            audio_file: 音声ファイルのパス
            
        Returns:
            認識されたテキスト（認識失敗時はNone）
        """
        pass
    
    def _prepare_audio(self, audio_data: np.ndarray, sample_rate: int, target_sample_rate: int = 16000) -> np.ndarray:
        """音声データを準備（モノラル化、リサンプリング、正規化）
        
        Args:
            audio_data: 音声データ
            sample_rate: 現在のサンプリングレート
            target_sample_rate: 目標サンプリングレート
            
        Returns:
            準備された音声データ（float32、-1.0～1.0の範囲）
        """
        # モノラルに変換
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # float32に変換（-1.0～1.0の範囲）
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32767.0
        elif audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = np.clip(audio_data, -1.0, 1.0)
        
        # リサンプリング（必要に応じて）
        if sample_rate != target_sample_rate:
            try:
                from scipy import signal
                num_samples = int(len(audio_data) * target_sample_rate / sample_rate)
                audio_data = signal.resample(audio_data, num_samples).astype(np.float32)
            except ImportError:
                logger.warning("scipyがインストールされていません。リサンプリングをスキップします。")
        
        return audio_data


class SpeechRecognitionEngine(STTEngine):
    """SpeechRecognitionライブラリを使用したSTTエンジン（Google Web Speech API）"""
    
    def __init__(self, language: str = "ja-JP"):
        """SpeechRecognitionエンジンを初期化
        
        Args:
            language: 言語コード（デフォルト: ja-JP）
        """
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.language = language
            logger.debug(f"[STT] SpeechRecognition engine initialized (language: {language})")
        except ImportError:
            raise ImportError("speech_recognitionがインストールされていません。pip install SpeechRecognitionでインストールしてください。")
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> Optional[str]:
        """音声データをテキストに変換"""
        try:
            import speech_recognition as sr
            
            # 音声データを準備
            audio_data = self._prepare_audio(audio_data, sample_rate, target_sample_rate=16000)
            
            # 一時ファイルに保存
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name
                sf.write(tmp_path, audio_data, 16000)
                
                try:
                    # 音声ファイルを読み込み
                    with sr.AudioFile(tmp_path) as source:
                        audio = self.recognizer.record(source)
                    
                    # Google Web Speech APIで認識
                    text = self.recognizer.recognize_google(audio, language=self.language)
                    return text
                finally:
                    # 一時ファイルを削除
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
        except sr.UnknownValueError:
            logger.debug("[STT] SpeechRecognition could not understand audio")
            return None
        except sr.RequestError as e:
            logger.error(f"[STT] SpeechRecognition request error: {e}")
            return None
        except Exception as e:
            logger.error(f"[STT] SpeechRecognition error: {e}", exc_info=True)
            return None
    
    def transcribe_file(self, audio_file: str) -> Optional[str]:
        """音声ファイルをテキストに変換"""
        try:
            audio_data, sample_rate = sf.read(audio_file)
            return self.transcribe(audio_data, sample_rate)
        except Exception as e:
            logger.error(f"[STT] Error reading audio file {audio_file}: {e}", exc_info=True)
            return None


class VoskEngine(STTEngine):
    """Voskライブラリを使用したSTTエンジン（オフライン対応）"""
    
    def __init__(self, model_path: Optional[str] = None, language: str = "ja"):
        """Voskエンジンを初期化
        
        Args:
            model_path: Voskモデルのパス（Noneの場合はデフォルトモデルを使用）
            language: 言語コード（デフォルト: ja）
        """
        try:
            import vosk
            import json
            
            if model_path is None:
                # デフォルトモデルパス（ユーザーが設定する必要がある）
                model_path = os.path.expanduser("~/vosk-models/ja")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Voskモデルが見つかりません: {model_path}\n"
                    f"モデルをダウンロードしてください: https://alphacephei.com/vosk/models"
                )
            
            self.model = vosk.Model(model_path)
            self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
            self.language = language
            logger.debug(f"[STT] Vosk engine initialized (model: {model_path})")
        except ImportError:
            raise ImportError("voskがインストールされていません。pip install voskでインストールしてください。")
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> Optional[str]:
        """音声データをテキストに変換"""
        try:
            import json
            
            # 音声データを準備（16kHz、int16形式）
            audio_data = self._prepare_audio(audio_data, sample_rate, target_sample_rate=16000)
            audio_data_int16 = (audio_data * 32767.0).astype(np.int16)
            
            # Voskで認識
            self.recognizer.SetWords(True)
            self.recognizer.AcceptWaveform(audio_data_int16.tobytes())
            result = self.recognizer.FinalResult()
            
            result_dict = json.loads(result)
            text = result_dict.get("text", "").strip()
            
            return text if text else None
        except Exception as e:
            logger.error(f"[STT] Vosk error: {e}", exc_info=True)
            return None
    
    def transcribe_file(self, audio_file: str) -> Optional[str]:
        """音声ファイルをテキストに変換"""
        try:
            audio_data, sample_rate = sf.read(audio_file)
            return self.transcribe(audio_data, sample_rate)
        except Exception as e:
            logger.error(f"[STT] Error reading audio file {audio_file}: {e}", exc_info=True)
            return None


def create_stt_engine(engine_type: str, **kwargs) -> STTEngine:
    """STTエンジンを作成
    
    Args:
        engine_type: エンジンタイプ（"speechrecognition" または "vosk"）
        **kwargs: エンジン固有のパラメータ
        
    Returns:
        STTエンジンインスタンス
    """
    if engine_type.lower() == "speechrecognition" or engine_type.lower() == "google":
        language = kwargs.get("language", "ja-JP")
        return SpeechRecognitionEngine(language=language)
    elif engine_type.lower() == "vosk":
        model_path = kwargs.get("model_path", None)
        language = kwargs.get("language", "ja")
        return VoskEngine(model_path=model_path, language=language)
    else:
        raise ValueError(f"Unknown STT engine type: {engine_type}")

