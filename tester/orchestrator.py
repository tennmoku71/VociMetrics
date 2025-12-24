"""Orchestrator: 全体の進行・時系列ログ管理"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import logging

try:
    from parser.convo_parser import ConvoParser, ScenarioAction
except ImportError:
    # パーサーモジュールがインポートできない場合のフォールバック
    ConvoParser = None
    ScenarioAction = None

logger = logging.getLogger(__name__)


class UnifiedLogger:
    """Unified Event Timeline Logger
    
    音声パケットの送受信とAPIフックの時刻を単一の時計で記録する仕組み
    """
    
    def __init__(self, output_dir: str = "data/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events: List[Dict[str, Any]] = []
        self.start_time: Optional[float] = None
        self.test_id: Optional[str] = None
        
    def start_test(self, test_id: str, test_type: str = "rule"):
        """テスト開始を記録"""
        self.test_id = test_id
        self.start_time = asyncio.get_event_loop().time()
        self.events = []
        
        self.log_event({
            "type": "TEST_START",
            "test_id": test_id,
            "test_type": test_type,
            "timestamp": datetime.now().isoformat()
        })
        
    def log_event(self, event_data: Dict[str, Any]):
        """イベントを記録
        
        Args:
            event_data: イベントデータ（timeフィールドは相対時間（ミリ秒）で自動計算）
        """
        if self.start_time is None:
            raise RuntimeError("Test not started. Call start_test() first.")
            
        current_time = asyncio.get_event_loop().time()
        relative_time_ms = int((current_time - self.start_time) * 1000)
        
        event = {
            "time": relative_time_ms,
            **event_data
        }
        self.events.append(event)
        
    def log_user_speech_start(self, text: Optional[str] = None):
        """ユーザー発話開始を記録"""
        self.log_event({
            "type": "USER_SPEECH_START",
            "text": text
        })
        
    def log_user_speech_end(self):
        """ユーザー発話終了を記録"""
        self.log_event({
            "type": "USER_SPEECH_END"
        })
        
    def log_bot_speech_start(self, text: Optional[str] = None):
        """ボット発話開始を記録"""
        self.log_event({
            "type": "BOT_SPEECH_START",
            "text": text
        })
        
    def log_bot_speech_end(self):
        """ボット発話終了を記録"""
        self.log_event({
            "type": "BOT_SPEECH_END"
        })
        
    def log_api_call_start(self, function: str, args: Dict[str, Any]):
        """API呼び出し開始を記録"""
        self.log_event({
            "type": "API_CALL_START",
            "function": function,
            "args": args
        })
        
    def log_api_call_end(self, status: str, result: Optional[Any] = None):
        """API呼び出し終了を記録"""
        self.log_event({
            "type": "API_CALL_END",
            "status": status,
            "result": result
        })
        
    def save_timeline(self) -> Path:
        """時系列ログをJSONファイルに保存"""
        if self.test_id is None:
            raise RuntimeError("Test not started. Call start_test() first.")
            
        timeline_data = {
            "test_metadata": {
                "id": self.test_id,
                "eval_status": "pending",
                "created_at": datetime.now().isoformat()
            },
            "events": self.events
        }
        
        filename = f"{self.test_id}_timeline.json"
        filepath = self.output_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(timeline_data, f, ensure_ascii=False, indent=2)
            
        return filepath


class Orchestrator:
    """テスト実行のオーケストレーター"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = UnifiedLogger(
            output_dir=config.get("logging", {}).get("output_dir", "data/reports")
        )
        if ConvoParser:
            self.convo_parser = ConvoParser(
                scenarios_dir=config.get("scenarios_dir", "data/scenarios")
            )
        else:
            self.convo_parser = None
            logger.warning("[Orchestrator] ConvoParser not available")
        self.event_waiters: Dict[str, asyncio.Event] = {}
        
    async def run_test(self, test_id: str, test_type: str = "rule"):
        """テストを開始（ロガーを初期化）"""
        self.logger.start_test(test_id, test_type)
        print(f"[Orchestrator] Test started: {test_id} (type: {test_type})")
    
    async def run_scenario(
        self,
        convo_file: str,
        audio_sender: Optional[Callable] = None,
        event_handler: Optional[Callable] = None
    ):
        """シナリオを実行
        
        Args:
            convo_file: .convoファイルのパス
            audio_sender: 音声ファイルを送信する関数（audio_file_pathを受け取る）
            event_handler: イベントハンドラー（event_type, dataを受け取る）
        """
        if not self.convo_parser:
            raise RuntimeError("ConvoParser not available")
        
        # シナリオをパース
        actions = self.convo_parser.parse(convo_file)
        logger.info(f"[Orchestrator] Executing scenario with {len(actions)} actions")
        
        # イベント待機用のイベントを作成
        self.event_waiters = {
            "BOT_SPEECH_START": asyncio.Event(),
            "BOT_SPEECH_END": asyncio.Event(),
            "USER_SPEECH_END": asyncio.Event()
        }
        
        # 各アクションを実行
        for i, action in enumerate(actions):
            logger.info(f"[Orchestrator] Executing action {i+1}/{len(actions)}: {action.action_type}")
            
            # 遅延がある場合は待機
            if action.delay_ms > 0:
                await asyncio.sleep(action.delay_ms / 1000.0)
            
            # アクションタイプに応じて処理
            if action.action_type == "USER_SPEECH_START":
                # ユーザー発話開始: 音声ファイルを送信
                if audio_sender and action.audio_file:
                    logger.info(f"[Orchestrator] Sending user speech: {action.audio_file}")
                    self.logger.log_user_speech_start(text=f"Audio file: {action.audio_file}")
                    await audio_sender(action.audio_file)
                    self.logger.log_user_speech_end()
                    self.event_waiters["USER_SPEECH_END"].set()
                    
            elif action.action_type == "WAIT_FOR_BOT_SPEECH_START":
                # ボット発話開始を待機（VAD検出）
                logger.info(f"[Orchestrator] Waiting for bot speech start...")
                # イベントをリセット
                self.event_waiters["BOT_SPEECH_START"].clear()
                try:
                    await asyncio.wait_for(
                        self.event_waiters["BOT_SPEECH_START"].wait(),
                        timeout=15.0  # タイムアウトを15秒に延長
                    )
                    logger.info("[Orchestrator] Bot speech detected!")
                except asyncio.TimeoutError:
                    logger.warning("[Orchestrator] Timeout waiting for bot speech start (15s)")
                    # タイムアウトしても続行
                    
            elif action.action_type == "WAIT_FOR_BOT_SPEECH_END":
                # ボット発話終了を待機（VAD検出）
                logger.info("[Orchestrator] Waiting for bot speech end...")
                # イベントをリセット
                self.event_waiters["BOT_SPEECH_END"].clear()
                try:
                    await asyncio.wait_for(
                        self.event_waiters["BOT_SPEECH_END"].wait(),
                        timeout=15.0  # タイムアウトを15秒に延長
                    )
                    logger.info("[Orchestrator] Bot speech ended!")
                except asyncio.TimeoutError:
                    logger.warning("[Orchestrator] Timeout waiting for bot speech end (15s)")
                    # タイムアウトしても続行
        
        logger.info("[Orchestrator] Scenario execution completed")
    
    def notify_event(self, event_type: str):
        """イベントを通知（VAD検出などから呼ばれる）"""
        if event_type in self.event_waiters:
            self.event_waiters[event_type].set()
            logger.debug(f"[Orchestrator] Event notified: {event_type}")

