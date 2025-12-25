"""Botium形式の.convoファイルパーサー"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Speaker(Enum):
    """発話者"""
    USER = "me"
    BOT = "bot"


class ScenarioAction:
    """シナリオアクション（タスクキューアイテム）"""
    
    def __init__(
        self,
        action_type: str,
        speaker: Optional[Speaker] = None,
        text: Optional[str] = None,
        audio_file: Optional[str] = None,
        wait_for: Optional[str] = None,
        delay_ms: int = 0,
        toolcall_name: Optional[str] = None,
        toolcall_arguments: Optional[Dict[str, Any]] = None
    ):
        """アクションを初期化
        
        Args:
            action_type: アクションタイプ（USER_SPEECH_START, BOT_SPEECH_START, API_WAIT, etc.）
            speaker: 発話者（USER/BOT）
            text: 発話テキスト
            audio_file: 音声ファイルパス（ルールテスト用）
            wait_for: 待機するイベントタイプ
            delay_ms: 遅延時間（ミリ秒）
            toolcall_name: 期待されるtoolcall名
            toolcall_arguments: 期待されるtoolcall引数
        """
        self.action_type = action_type
        self.speaker = speaker
        self.text = text
        self.audio_file = audio_file
        self.wait_for = wait_for
        self.delay_ms = delay_ms
        self.toolcall_name = toolcall_name
        self.toolcall_arguments = toolcall_arguments
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        result = {
            "action_type": self.action_type,
            "delay_ms": self.delay_ms
        }
        if self.speaker:
            result["speaker"] = self.speaker.value
        if self.text:
            result["text"] = self.text
        if self.audio_file:
            result["audio_file"] = self.audio_file
        if self.wait_for:
            result["wait_for"] = self.wait_for
        if self.toolcall_name:
            result["toolcall_name"] = self.toolcall_name
        if self.toolcall_arguments:
            result["toolcall_arguments"] = self.toolcall_arguments
        return result


class ConvoParser:
    """Botium形式の.convoファイルパーサー"""
    
    def __init__(self, scenarios_dir: str = "data/scenarios"):
        self.scenarios_dir = Path(scenarios_dir)
        self.scenarios_dir.mkdir(parents=True, exist_ok=True)
    
    def parse(self, convo_file: str) -> List[ScenarioAction]:
        """.convoファイルをパースしてタスクキューを生成
        
        現時点での形式:
        #me <audio_file_path>
        #bot [speechStart]
        #bot [speechEnd]
        
        Args:
            convo_file: .convoファイルのパス
            
        Returns:
            シナリオアクションのリスト
        """
        convo_path = Path(convo_file)
        if not convo_path.exists():
            # scenarios_dirからの相対パスを試す
            convo_path = self.scenarios_dir / convo_file
            if not convo_path.exists():
                raise FileNotFoundError(f"Convo file not found: {convo_file}")
        
        logger.debug(f"Parsing convo file: {convo_path}")
        
        with open(convo_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        actions = []
        lines = content.split("\n")
        
        for line in lines:
            line = line.strip()
            
            # コメント行をスキップ（#me, #bot, #toolcallは除外）
            if line.startswith("#") and not line.startswith("#me") and not line.startswith("#bot") and not line.startswith("#toolcall"):
                continue
            
            # ユーザー発話: #me <audio_file_path> または #me <text>
            if line.startswith("#me"):
                parts = line.split(None, 1)  # 最初の空白で分割
                if len(parts) > 1:
                    content = parts[1].strip()
                    # ファイルパスかテキストかを判定
                    # 1. 拡張子がある場合はファイルパスとみなす
                    # 2. ファイルが存在する場合はファイルパスとみなす
                    # 3. それ以外はテキストとみなす
                    content_path = Path(content)
                    has_extension = content_path.suffix.lower() in ['.wav', '.mp3', '.m4a', '.ogg', '.flac']
                    file_exists = content_path.exists()
                    
                    # scenarios_dirからの相対パスも確認
                    if not file_exists and not content_path.is_absolute():
                        relative_path = self.scenarios_dir.parent / content_path
                        file_exists = relative_path.exists()
                        if file_exists:
                            content = str(relative_path)
                    
                    if has_extension or file_exists:
                        # 音声ファイルパス
                        actions.append(ScenarioAction(
                            action_type="USER_SPEECH_START",
                            speaker=Speaker.USER,
                            audio_file=content
                        ))
                    else:
                        # テキスト（TTSで変換する必要がある）
                        actions.append(ScenarioAction(
                            action_type="USER_SPEECH_START",
                            speaker=Speaker.USER,
                            text=content
                        ))
                    actions.append(ScenarioAction(
                        action_type="USER_SPEECH_END",
                        speaker=Speaker.USER
                    ))
                else:
                    logger.warning(f"Invalid #me line (missing content): {line}")
            
            # ボット発話: #bot [speechStart] または #bot [speechEnd] または #bot <expected_text>
            elif line.startswith("#bot"):
                parts = line.split(None, 1)
                if len(parts) > 1:
                    content = parts[1].strip()
                    if content == "[speechStart]":
                        actions.append(ScenarioAction(
                            action_type="WAIT_FOR_BOT_SPEECH_START",
                            speaker=Speaker.BOT,
                            wait_for="BOT_SPEECH_START"
                        ))
                    elif content == "[speechEnd]":
                        actions.append(ScenarioAction(
                            action_type="WAIT_FOR_BOT_SPEECH_END",
                            speaker=Speaker.BOT,
                            wait_for="BOT_SPEECH_END"
                        ))
                    else:
                        # 期待テキストとして扱う
                        # speechStartとspeechEndの両方を待機し、テキスト比較も行う
                        actions.append(ScenarioAction(
                            action_type="WAIT_FOR_BOT_SPEECH_START",
                            speaker=Speaker.BOT,
                            wait_for="BOT_SPEECH_START",
                            text=content  # 期待テキスト
                        ))
                        actions.append(ScenarioAction(
                            action_type="WAIT_FOR_BOT_SPEECH_END",
                            speaker=Speaker.BOT,
                            wait_for="BOT_SPEECH_END",
                            text=content  # 期待テキスト（比較用）
                        ))
                else:
                    logger.warning(f"Invalid #bot line (missing content): {line}")
            
            # Toolcall期待値: #toolcall <name> <arguments_json>
            elif line.startswith("#toolcall"):
                parts = line.split(None, 2)
                if len(parts) >= 2:
                    toolcall_name = parts[1].strip()
                    toolcall_arguments = {}
                    if len(parts) >= 3:
                        # JSON形式の引数をパース
                        try:
                            import json
                            toolcall_arguments = json.loads(parts[2].strip())
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in toolcall arguments: {parts[2]}")
                            toolcall_arguments = {}
                    
                    actions.append(ScenarioAction(
                        action_type="WAIT_FOR_TOOLCALL",
                        wait_for="TOOLCALL",
                        toolcall_name=toolcall_name,
                        toolcall_arguments=toolcall_arguments
                    ))
                else:
                    logger.warning(f"Invalid #toolcall line (missing name): {line}")
        
        logger.debug(f"Parsed {len(actions)} actions from convo file")
        return actions
    
    
    def get_available_scenarios(self) -> List[str]:
        """利用可能なシナリオファイルのリストを取得"""
        scenarios = []
        for file in self.scenarios_dir.glob("*.convo"):
            scenarios.append(file.name)
        return scenarios

