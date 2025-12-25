"""音声品質評価モジュール - SNR、ノイズプロファイリング、STT Confidence、幻聴率の計算"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq

logger = logging.getLogger(__name__)


class SoundEvaluator:
    """音声品質評価エンジン"""
    
    def __init__(self, config: Dict[str, Any]):
        """音声品質評価エンジンを初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        sound_config = config.get("evaluation", {}).get("sound", {})
        self.snr_threshold_db = sound_config.get("snr_threshold_db", 15.0)
        self.false_trigger_threshold = sound_config.get("false_trigger_threshold", 0.1)
    
    def calculate_snr(
        self,
        audio: np.ndarray,
        sample_rate: int,
        vad_on_samples: List[Tuple[int, int]],
        vad_off_samples: List[Tuple[int, int]]
    ) -> Dict[str, Any]:
        """SNR (Signal-to-Noise Ratio) を計算
        
        VADのON（発話）とOFF（背景音）のパワー比による音声の明瞭度を評価
        
        Args:
            audio: 音声データ（int16）
            sample_rate: サンプルレート
            vad_on_samples: VAD ON区間のリスト [(start_sample, end_sample), ...]
            vad_off_samples: VAD OFF区間のリスト [(start_sample, end_sample), ...]
        
        Returns:
            SNR評価結果の辞書
        """
        if len(vad_on_samples) == 0 or len(vad_off_samples) == 0:
            return {
                "snr_db": None,
                "signal_power": None,
                "noise_power": None,
                "snr_ok": False
            }
        
        # 信号パワー（VAD ON区間の平均パワー）
        signal_powers = []
        for start, end in vad_on_samples:
            segment = audio[start:end].astype(np.float32)
            power = np.mean(segment ** 2)
            signal_powers.append(power)
        
        signal_power = np.mean(signal_powers) if signal_powers else 0.0
        
        # ノイズパワー（VAD OFF区間の平均パワー）
        noise_powers = []
        for start, end in vad_off_samples:
            segment = audio[start:end].astype(np.float32)
            power = np.mean(segment ** 2)
            noise_powers.append(power)
        
        noise_power = np.mean(noise_powers) if noise_powers else 0.0
        
        # SNRを計算（dB）
        if noise_power == 0:
            snr_db = float('inf') if signal_power > 0 else None
        else:
            snr_db = 10.0 * np.log10(signal_power / noise_power)
        
        snr_ok = snr_db is not None and snr_db >= self.snr_threshold_db
        
        return {
            "snr_db": float(snr_db) if snr_db is not None and not np.isinf(snr_db) else None,
            "signal_power": float(signal_power),
            "noise_power": float(noise_power),
            "snr_ok": snr_ok
        }
    
    def analyze_noise_profile(
        self,
        audio: np.ndarray,
        sample_rate: int,
        vad_off_samples: List[Tuple[int, int]]
    ) -> Dict[str, Any]:
        """ノイズ・プロファイリング（スペクトル解析）
        
        OFF区間のスペクトル解析によるノイズの種類（定常、低域、高域）の特定
        
        Args:
            audio: 音声データ（int16）
            sample_rate: サンプルレート
            vad_off_samples: VAD OFF区間のリスト [(start_sample, end_sample), ...]
        
        Returns:
            ノイズプロファイル評価結果の辞書
        """
        if len(vad_off_samples) == 0:
            return {
                "noise_type": "unknown",
                "low_freq_power": None,
                "mid_freq_power": None,
                "high_freq_power": None,
                "spectral_flatness": None
            }
        
        # すべてのOFF区間を結合
        noise_segments = []
        for start, end in vad_off_samples:
            segment = audio[start:end].astype(np.float32)
            noise_segments.append(segment)
        
        if not noise_segments:
            return {
                "noise_type": "unknown",
                "low_freq_power": None,
                "mid_freq_power": None,
                "high_freq_power": None,
                "spectral_flatness": None
            }
        
        noise_audio = np.concatenate(noise_segments)
        
        # FFTでスペクトル解析
        fft_result = fft(noise_audio)
        freqs = fftfreq(len(noise_audio), 1.0 / sample_rate)
        power_spectrum = np.abs(fft_result) ** 2
        
        # 周波数帯域を定義
        nyquist = sample_rate / 2.0
        low_freq_end = nyquist * 0.2  # 0-20% (低域)
        mid_freq_start = nyquist * 0.2
        mid_freq_end = nyquist * 0.7  # 20-70% (中域)
        high_freq_start = nyquist * 0.7  # 70-100% (高域)
        
        # 各周波数帯域のパワーを計算
        low_mask = (freqs >= 0) & (freqs <= low_freq_end)
        mid_mask = (freqs > mid_freq_start) & (freqs <= mid_freq_end)
        high_mask = (freqs > high_freq_start) & (freqs <= nyquist)
        
        low_freq_power = np.mean(power_spectrum[low_mask]) if np.any(low_mask) else 0.0
        mid_freq_power = np.mean(power_spectrum[mid_mask]) if np.any(mid_mask) else 0.0
        high_freq_power = np.mean(power_spectrum[high_mask]) if np.any(high_mask) else 0.0
        
        total_power = low_freq_power + mid_freq_power + high_freq_power
        if total_power > 0:
            low_ratio = low_freq_power / total_power
            mid_ratio = mid_freq_power / total_power
            high_ratio = high_freq_power / total_power
        else:
            low_ratio = mid_ratio = high_ratio = 0.0
        
        # スペクトル平坦度を計算（定常性の指標）
        # 幾何平均 / 算術平均
        positive_power = power_spectrum[power_spectrum > 0]
        if len(positive_power) > 0:
            geometric_mean = np.exp(np.mean(np.log(positive_power)))
            arithmetic_mean = np.mean(power_spectrum)
            spectral_flatness = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0.0
        else:
            spectral_flatness = 0.0
        
        # ノイズタイプを判定
        if spectral_flatness > 0.7:
            noise_type = "stationary"  # 定常ノイズ（ホワイトノイズなど）
        elif low_ratio > 0.5:
            noise_type = "low_freq"  # 低域ノイズ
        elif high_ratio > 0.4:
            noise_type = "high_freq"  # 高域ノイズ
        else:
            noise_type = "mixed"  # 混合ノイズ
        
        return {
            "noise_type": noise_type,
            "low_freq_power": float(low_freq_power),
            "mid_freq_power": float(mid_freq_power),
            "high_freq_power": float(high_freq_power),
            "spectral_flatness": float(spectral_flatness),
            "low_ratio": float(low_ratio),
            "mid_ratio": float(mid_ratio),
            "high_ratio": float(high_ratio)
        }
    
    def calculate_stt_confidence(
        self,
        stt_results: List[Dict[str, Any]],
        noise_power: Optional[float]
    ) -> Dict[str, Any]:
        """STT Confidence (認識信頼度) を計算
        
        ノイズ強度に対して、STTエンジンがどれだけ確信を持って文字起こしできたか
        
        Args:
            stt_results: STT結果のリスト [{"text": "...", "confidence": ...}, ...]
            noise_power: ノイズパワー（Noneの場合は計算できない）
        
        Returns:
            STT Confidence評価結果の辞書
        """
        if not stt_results:
            return {
                "average_confidence": None,
                "confidence_ok": False,
                "noise_impact": None
            }
        
        # STTエンジンがconfidenceを返している場合
        confidences = []
        for result in stt_results:
            confidence = result.get("confidence")
            if confidence is not None:
                confidences.append(float(confidence))
        
        if not confidences:
            # confidenceが利用できない場合は、テキストの長さや存在を基に推定
            # ここでは簡易的な実装として、テキストが存在する場合は0.5、存在しない場合は0.0とする
            return {
                "average_confidence": None,
                "confidence_ok": False,
                "noise_impact": None,
                "note": "STT engine does not provide confidence scores"
            }
        
        average_confidence = np.mean(confidences)
        
        # ノイズの影響を評価（ノイズパワーが高いほどconfidenceが低くなる傾向）
        noise_impact = None
        if noise_power is not None:
            # ノイズパワーを正規化（0-1の範囲に）
            # 実際の値は音声のパワーに依存するため、簡易的な実装
            normalized_noise = min(1.0, noise_power / 1e6)  # 適切な閾値で正規化
            noise_impact = normalized_noise
        
        # confidenceが0.7以上なら良好と判定
        confidence_ok = average_confidence >= 0.7
        
        return {
            "average_confidence": float(average_confidence),
            "confidence_ok": confidence_ok,
            "noise_impact": float(noise_impact) if noise_impact is not None else None
        }
    
    def calculate_false_trigger_rate(
        self,
        stt_results: List[Dict[str, Any]],
        vad_off_samples: List[Tuple[int, int]],
        sample_rate: int
    ) -> Dict[str, Any]:
        """幻聴率 (False Trigger Rate) を計算
        
        ユーザーが無音またはノイズのみの状態（VAD-OFF）で、STTが誤って文字を出力した頻度
        
        Args:
            stt_results: STT結果のリスト [{"text": "...", "start_time": ..., "end_time": ...}, ...]
            vad_off_samples: VAD OFF区間のリスト [(start_sample, end_sample), ...]
            sample_rate: サンプルレート
        
        Returns:
            幻聴率評価結果の辞書
        """
        # VAD OFF区間の総時間を計算
        total_vad_off_duration = sum((end - start) / sample_rate for start, end in vad_off_samples)
        
        if total_vad_off_duration == 0:
            return {
                "false_trigger_rate": 0.0,
                "false_trigger_count": 0,
                "false_trigger_ok": True
            }
        
        # STT結果がVAD OFF区間内にあるかチェック
        false_triggers = 0
        for stt_result in stt_results:
            stt_text = stt_result.get("text", "").strip()
            if not stt_text:  # 空のテキストは無視
                continue
            
            stt_start_time = stt_result.get("start_time")
            stt_end_time = stt_result.get("end_time")
            
            if stt_start_time is None:
                continue
            
            # STT結果がVAD OFF区間内にあるかチェック
            for start_sample, end_sample in vad_off_samples:
                off_start_time = start_sample / sample_rate
                off_end_time = end_sample / sample_rate
                
                # STT結果の開始時刻がVAD OFF区間内にあるか
                if off_start_time <= stt_start_time <= off_end_time:
                    false_triggers += 1
                    break
        
        # 幻聴率 = 幻聴回数 / VAD OFF区間の総時間（回/秒）
        false_trigger_rate = false_triggers / total_vad_off_duration if total_vad_off_duration > 0 else 0.0
        
        false_trigger_ok = false_trigger_rate <= self.false_trigger_threshold
        
        return {
            "false_trigger_rate": float(false_trigger_rate),
            "false_trigger_count": false_triggers,
            "false_trigger_ok": false_trigger_ok
        }
    
    def evaluate_sound(
        self,
        bot_audio: np.ndarray,
        sample_rate: int,
        vad_on_samples: List[Tuple[int, int]],
        vad_off_samples: List[Tuple[int, int]],
        stt_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """音声品質を総合評価
        
        Args:
            bot_audio: ボット音声データ（int16）
            sample_rate: サンプルレート
            vad_on_samples: VAD ON区間のリスト [(start_sample, end_sample), ...]
            vad_off_samples: VAD OFF区間のリスト [(start_sample, end_sample), ...]
            stt_results: STT結果のリスト
        
        Returns:
            音声品質評価結果の辞書
        """
        # SNRを計算
        snr_result = self.calculate_snr(bot_audio, sample_rate, vad_on_samples, vad_off_samples)
        
        # ノイズプロファイリング
        noise_profile = self.analyze_noise_profile(bot_audio, sample_rate, vad_off_samples)
        
        # STT Confidence
        stt_confidence = self.calculate_stt_confidence(stt_results, snr_result.get("noise_power"))
        
        # 幻聴率
        false_trigger_rate = self.calculate_false_trigger_rate(
            stt_results, vad_off_samples, sample_rate
        )
        
        # 総合スコアを計算（各項目の重み付き平均）
        score = 0.0
        max_score = 0.0
        
        # SNR (30%)
        if snr_result.get("snr_db") is not None:
            snr_db = snr_result["snr_db"]
            if snr_db >= self.snr_threshold_db:
                score += 30.0
            else:
                # 線形減点
                score += max(0.0, 30.0 * (snr_db / self.snr_threshold_db))
            max_score += 30.0
        
        # STT Confidence (30%)
        if stt_confidence.get("average_confidence") is not None:
            conf = stt_confidence["average_confidence"]
            score += 30.0 * conf
            max_score += 30.0
        
        # 幻聴率 (20%)
        if false_trigger_rate.get("false_trigger_ok"):
            score += 20.0
        else:
            ftr = false_trigger_rate["false_trigger_rate"]
            threshold = self.false_trigger_threshold
            if ftr <= threshold * 2:
                score += max(0.0, 20.0 * (1.0 - (ftr - threshold) / threshold))
            max_score += 20.0
        
        # ノイズプロファイリング (20%) - 定常ノイズが検出された場合は減点
        if noise_profile.get("noise_type") == "stationary":
            score += 20.0  # 定常ノイズは期待される（ホワイトノイズテストの場合）
        elif noise_profile.get("noise_type") != "unknown":
            score += 15.0  # その他のノイズタイプ
        max_score += 20.0
        
        final_score = (score / max_score * 100.0) if max_score > 0 else 0.0
        
        return {
            "snr": snr_result,
            "noise_profile": noise_profile,
            "stt_confidence": stt_confidence,
            "false_trigger_rate": false_trigger_rate,
            "score": final_score
        }

