"""VociMetrics - GUI Application
StreamlitベースのGUIアプリケーション
"""

import os
import warnings

# Streamlitの警告とウェルカムメッセージを抑制
os.environ["STREAMLIT_LOGGER_LEVEL"] = "error"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")

import streamlit as st
import json
import subprocess
import sys
from pathlib import Path
import time
import re
from typing import Dict, Any, Optional
import threading
import queue
import logging

# Streamlitの警告ログを抑制
logging.getLogger("streamlit.runtime.scriptrunner.script_runner").setLevel(logging.CRITICAL)
logging.getLogger("streamlit.runtime.caching").setLevel(logging.CRITICAL)
logging.getLogger("streamlit.runtime.metrics_util").setLevel(logging.CRITICAL)

# デバッグログを有効化（一時的）
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ページ設定
st.set_page_config(
    page_title="VociMetrics",
    page_icon="🎤",
    layout="wide"
)

# プロジェクトルートを取得
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_FILE = PROJECT_ROOT / "config.json"
CONFIG_OPTIONS_FILE = PROJECT_ROOT / "config_options.json"
TEMP_CONFIG_FILE = PROJECT_ROOT / "_config.gui.json"
TEMP_SCENARIO_FILE = PROJECT_ROOT / "_scenario.gui.convo"
SCENARIOS_DIR = PROJECT_ROOT / "scenarios"
MAIN_SCRIPT = PROJECT_ROOT / "main.py"


def inject_tooltip_css():
    """ツールチップ用のCSSスタイルを注入"""
    css = """
    <style>
    .tooltip-icon {
        display: inline-flex;
        justify-content: center;
        align-items: center;
        width: 18px;
        height: 18px;
        border: 1px solid #999;
        border-radius: 50%;
        font-size: 12px;
        font-family: sans-serif;
        color: #666;
        cursor: help;
        position: relative;
        margin-left: 4px;
        vertical-align: middle;
    }
    
    .tooltip-icon:hover {
        background-color: #f0f0f0;
        border-color: #333;
    }
    
    .tooltip-icon::after {
        content: attr(data-tooltip);
        position: absolute;
        bottom: 150%;
        left: 50%;
        transform: translateX(-50%);
        background-color: #333;
        color: #fff;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 11px;
        white-space: normal;
        width: 250px;
        opacity: 0;
        visibility: hidden;
        transition: all 0.2s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        z-index: 1000;
        text-align: left;
        line-height: 1.4;
    }
    
    .tooltip-icon:hover::after {
        opacity: 1;
        visibility: visible;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def load_config() -> Dict[str, Any]:
    """config.jsonを読み込む"""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")
    
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config_options() -> Dict[str, Any]:
    """config_options.jsonを読み込む"""
    if not CONFIG_OPTIONS_FILE.exists():
        return {}
    
    with open(CONFIG_OPTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config: Dict[str, Any], use_temp: bool = False):
    """設定を保存
    
    Args:
        config: 保存する設定辞書
        use_temp: Trueの場合、一時ファイル（_config.gui.json）に保存
    """
    target_file = TEMP_CONFIG_FILE if use_temp else CONFIG_FILE
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def get_scenario_files() -> list:
    """シナリオファイルのリストを取得"""
    if not SCENARIOS_DIR.exists():
        return []
    
    convo_files = list(SCENARIOS_DIR.glob("*.convo"))
    return sorted([f.name for f in convo_files])


def render_config_field(
    config: Dict[str, Any],
    options: Dict[str, Any],
    key_path: list,
    label: str
) -> Any:
    """設定フィールドをレンダリング（再帰的）"""
    if not options:
        return None
    
    # configから現在の値を取得
    current_value = config
    for i, key in enumerate(key_path):
        if isinstance(current_value, dict):
            if i == len(key_path) - 1:
                # 最後のキーの場合は値を取得
                current_value = current_value.get(key)
            else:
                # 途中のキーの場合は辞書を取得
                current_value = current_value.get(key, {})
        else:
            current_value = None
            break
    
    # デバッグログ
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"[render_config_field] key_path={key_path}, current_value={current_value}, type={type(current_value)}")
    
    # optionsから設定オプションを取得
    option = options
    for i, key in enumerate(key_path):
        if isinstance(option, dict):
            prev_option = option
            option = option.get(key)  # Noneを返すように変更（空の辞書ではなく）
            if option is None:
                available_keys = list(prev_option.keys()) if isinstance(prev_option, dict) else 'N/A'
                # logger.debug(f"[render_config_field] Option key '{key}' not found at path {key_path[:i+1]}, available keys: {available_keys}")
                return None
        else:
            # logger.debug(f"[render_config_field] Option path not found: {key_path[:i+1]}, option type: {type(option)}")
            return None
    
    if not isinstance(option, dict) or "type" not in option:
        # logger.debug(f"[render_config_field] Invalid option: {option}, key_path={key_path}, option type: {type(option)}, option keys: {list(option.keys()) if isinstance(option, dict) else 'N/A'}")
        return None
    
    field_type = option.get("type")
    default_value = option.get("default")
    description = option.get("description", "")
    # 現在の値がNoneまたは空の辞書の場合はデフォルト値を使用
    if current_value is None or (isinstance(current_value, dict) and len(current_value) == 0):
        # logger.debug(f"[render_config_field] Using default value: {default_value} (current_value was {current_value})")
        current_value = default_value
    # else:
    #     logger.debug(f"[render_config_field] Using current value: {current_value}")
    
    # ラベルとツールチップアイコンを表示
    display_label = label
    if description:
        # HTMLでツールチップアイコンを表示
        tooltip_html = f"""
        <span class="tooltip-icon" data-tooltip="{description.replace('"', '&quot;')}">?</span>
        """
        label_with_tooltip = f"{label}{tooltip_html}"
        st.markdown(label_with_tooltip, unsafe_allow_html=True)
    else:
        st.write(label)
    
    if field_type == "select":
        options_list = option.get("options", [])
        if options_list:
            try:
                index = options_list.index(current_value) if current_value in options_list else 0
            except (ValueError, TypeError):
                index = 0
            return st.selectbox(display_label, options_list, index=index, key=f"{'.'.join(key_path)}", label_visibility="collapsed")
    
    elif field_type == "number":
        min_val = option.get("min", 0)
        max_val = option.get("max", 100)
        step = option.get("step", 1)
        
        # 型を統一（current_valueがfloatの場合は、min/max/stepもfloatに）
        if current_value is not None:
            if isinstance(current_value, float) or (isinstance(current_value, int) and step != int(step)):
                min_val = float(min_val)
                max_val = float(max_val)
                step = float(step)
                current_value = float(current_value)
            else:
                min_val = int(min_val)
                max_val = int(max_val)
                step = int(step)
                current_value = int(current_value)
        else:
            # デフォルト値の型に合わせる
            if isinstance(default_value, float) or (step != int(step)):
                min_val = float(min_val)
                max_val = float(max_val)
                step = float(step)
                default_value = float(default_value) if default_value is not None else 0.0
            else:
                min_val = int(min_val)
                max_val = int(max_val)
                step = int(step)
                default_value = int(default_value) if default_value is not None else 0
        
        return st.number_input(
            display_label,
            min_value=min_val,
            max_value=max_val,
            value=current_value if current_value is not None else default_value,
            step=step,
            key=f"{'.'.join(key_path)}",
            label_visibility="collapsed"
        )
    
    elif field_type == "checkbox":
        return st.checkbox(
            display_label,
            value=bool(current_value) if current_value is not None else bool(default_value),
            key=f"{'.'.join(key_path)}"
        )
    
    elif field_type == "text":
        return st.text_input(
            display_label,
            value=str(current_value) if current_value is not None else str(default_value) if default_value is not None else "",
            key=f"{'.'.join(key_path)}",
            label_visibility="collapsed"
        )
    
    elif field_type == "password":
        return st.text_input(
            display_label,
            value=str(current_value) if current_value is not None else str(default_value) if default_value is not None else "",
            type="password",
            key=f"{'.'.join(key_path)}",
            label_visibility="collapsed"
        )
    
    elif field_type == "textarea":
        return st.text_area(
            display_label,
            value=str(current_value) if current_value is not None else str(default_value) if default_value is not None else "",
            height=150,
            key=f"{'.'.join(key_path)}",
            label_visibility="collapsed"
        )
    
    return None


def edit_config(config: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """設定を編集可能なUIで表示
    
    Args:
        config: 元の設定（config.jsonから読み込んだ値）
        options: 設定オプション（config_options.jsonから読み込んだ値）
    
    Returns:
        編集された設定（元の値をデフォルトとして保持）
    """
    st.header("📋 Configuration")
    
    # デバッグログ
    import logging
    logger = logging.getLogger(__name__)
    # logger.debug(f"[edit_config] config keys: {list(config.keys())}")
    # logger.debug(f"[edit_config] evaluation config: {config.get('evaluation', {})}")
    
    # 元の設定をデフォルトとして保持（Deep copy）
    edited_config = json.loads(json.dumps(config))
    
    # STT設定
    with st.expander("STT Settings", expanded=False):
        stt_config = edited_config.setdefault("stt", {})
        stt_options = options.get("stt", {})
        
        if "engine" in stt_options:
            value = render_config_field(
                edited_config, options, ["stt", "engine"], "Engine"
            )
            if value is not None:
                stt_config["engine"] = value
            elif "engine" not in stt_config:
                # デフォルト値を設定
                stt_config["engine"] = stt_options["engine"].get("default", "speechrecognition")
        if "language" in stt_options:
            value = render_config_field(
                edited_config, options, ["stt", "language"], "Language"
            )
            if value is not None:
                stt_config["language"] = value
            elif "language" not in stt_config:
                stt_config["language"] = stt_options["language"].get("default", "ja-JP")
    
    # TTS設定
    with st.expander("TTS Settings", expanded=False):
        tts_config = edited_config.setdefault("tts", {})
        tts_options = options.get("tts", {})
        
        if "engine" in tts_options:
            value = render_config_field(
                edited_config, options, ["tts", "engine"], "Engine"
            )
            if value is not None:
                tts_config["engine"] = value
            elif "engine" not in tts_config:
                tts_config["engine"] = tts_options["engine"].get("default", "gtts")
        if "language" in tts_options:
            value = render_config_field(
                edited_config, options, ["tts", "language"], "Language"
            )
            if value is not None:
                tts_config["language"] = value
            elif "language" not in tts_config:
                tts_config["language"] = tts_options["language"].get("default", "ja")
        if "sample_rate" in tts_options:
            sample_rate_value = render_config_field(
                edited_config, options, ["tts", "sample_rate"], "Sample Rate"
            )
            if sample_rate_value is not None:
                tts_config["sample_rate"] = int(sample_rate_value)
            elif "sample_rate" not in tts_config:
                tts_config["sample_rate"] = tts_options["sample_rate"].get("default", 24000)
    
    # 評価設定
    with st.expander("Evaluation Settings", expanded=False):
        eval_config = edited_config.setdefault("evaluation", {})
        eval_options = options.get("evaluation", {})
        
        if "response_latency_threshold_ms" in eval_options:
            value = render_config_field(
                edited_config, options, ["evaluation", "response_latency_threshold_ms"],
                "Response Latency Threshold (ms)"
            )
            if value is not None:
                eval_config["response_latency_threshold_ms"] = int(value)
            elif "response_latency_threshold_ms" not in eval_config:
                eval_config["response_latency_threshold_ms"] = eval_options["response_latency_threshold_ms"].get("default", 800)
        if "toolcall_latency_threshold_ms" in eval_options:
            value = render_config_field(
                edited_config, options, ["evaluation", "toolcall_latency_threshold_ms"],
                "Toolcall Latency Threshold (ms)"
            )
            if value is not None:
                eval_config["toolcall_latency_threshold_ms"] = int(value)
            elif "toolcall_latency_threshold_ms" not in eval_config:
                eval_config["toolcall_latency_threshold_ms"] = eval_options["toolcall_latency_threshold_ms"].get("default", 2000)
        
        # Sound設定
        sound_config = eval_config.setdefault("sound", {})
        sound_options = eval_options.get("sound", {})
        white_noise_config = sound_config.setdefault("white_noise", {})
        white_noise_options = sound_options.get("white_noise", {})
        
        if "enabled" in white_noise_options:
            value = render_config_field(
                edited_config, options, ["evaluation", "sound", "white_noise", "enabled"],
                "White Noise Enabled"
            )
            if value is not None:
                white_noise_config["enabled"] = value
            elif "enabled" not in white_noise_config:
                white_noise_config["enabled"] = white_noise_options["enabled"].get("default", True)
        
        # White Noiseが有効な場合のみSNRを表示
        white_noise_enabled = white_noise_config.get("enabled", True)
        if white_noise_enabled and "snr_db" in white_noise_options:
            value = render_config_field(
                edited_config, options, ["evaluation", "sound", "white_noise", "snr_db"],
                "SNR (dB)"
            )
            if value is not None:
                white_noise_config["snr_db"] = float(value)
            elif "snr_db" not in white_noise_config:
                white_noise_config["snr_db"] = white_noise_options["snr_db"].get("default", 10.0)
        
        # Background Noise設定
        background_noise_config = white_noise_config.setdefault("background_noise", {})
        background_noise_options = white_noise_options.get("background_noise", {})
        
        if "enabled" in background_noise_options:
            value = render_config_field(
                edited_config, options, ["evaluation", "sound", "white_noise", "background_noise", "enabled"],
                "Enable Background Noise"
            )
            if value is not None:
                background_noise_config["enabled"] = value
            elif "enabled" not in background_noise_config:
                background_noise_config["enabled"] = background_noise_options["enabled"].get("default", True)
        
        # Background Noiseが有効な場合のみLevelを表示
        background_noise_enabled = background_noise_config.get("enabled", True)
        if background_noise_enabled and "level" in background_noise_options:
            value = render_config_field(
                edited_config, options, ["evaluation", "sound", "white_noise", "background_noise", "level"],
                "Background Noise Level"
            )
            if value is not None:
                background_noise_config["level"] = float(value)
            elif "level" not in background_noise_config:
                background_noise_config["level"] = background_noise_options["level"].get("default", 0.005)
        if "snr_threshold_db" in sound_options:
            value = render_config_field(
                edited_config, options, ["evaluation", "sound", "snr_threshold_db"],
                "SNR Threshold (dB)"
            )
            if value is not None:
                sound_config["snr_threshold_db"] = float(value)
            elif "snr_threshold_db" not in sound_config:
                sound_config["snr_threshold_db"] = sound_options["snr_threshold_db"].get("default", 15.0)
    
    # テキストマッチング設定
    with st.expander("Text Matching Settings", expanded=False):
        text_matching_config = edited_config.setdefault("text_matching", {})
        text_matching_options = options.get("text_matching", {})
        
        # Match Methodを先に取得
        match_method = text_matching_config.get("match_method")
        if "match_method" in text_matching_options:
            value = render_config_field(
                edited_config, options, ["text_matching", "match_method"],
                "Match Method"
            )
            if value is not None:
                text_matching_config["match_method"] = value
                match_method = value
            elif "match_method" not in text_matching_config:
                match_method = text_matching_options["match_method"].get("default", "llm")
                text_matching_config["match_method"] = match_method
        
        # match_methodに応じて必要な設定項目のみを表示
        if match_method == "edit_distance":
            # Edit Distance設定
            if "edit_distance_threshold" in text_matching_options:
                value = render_config_field(
                    edited_config, options, ["text_matching", "edit_distance_threshold"],
                    "Edit Distance Threshold"
                )
                if value is not None:
                    text_matching_config["edit_distance_threshold"] = float(value)
                elif "edit_distance_threshold" not in text_matching_config:
                    text_matching_config["edit_distance_threshold"] = text_matching_options["edit_distance_threshold"].get("default", 0.8)
        
        elif match_method == "llm":
            # LLM設定
            llm_config = text_matching_config.setdefault("llm", {})
            llm_options = text_matching_options.get("llm", {})
            
            if "model" in llm_options:
                value = render_config_field(
                    edited_config, options, ["text_matching", "llm", "model"],
                    "LLM Model"
                )
                if value is not None:
                    llm_config["model"] = value
                elif "model" not in llm_config:
                    llm_config["model"] = llm_options["model"].get("default", "gpt-4o-mini")
            if "temperature" in llm_options:
                value = render_config_field(
                    edited_config, options, ["text_matching", "llm", "temperature"],
                    "Temperature"
                )
                if value is not None:
                    llm_config["temperature"] = float(value)
                elif "temperature" not in llm_config:
                    llm_config["temperature"] = llm_options["temperature"].get("default", 0.0)
            if "similarity_threshold" in llm_options:
                value = render_config_field(
                    edited_config, options, ["text_matching", "llm", "similarity_threshold"],
                    "Similarity Threshold"
                )
                if value is not None:
                    llm_config["similarity_threshold"] = float(value)
                elif "similarity_threshold" not in llm_config:
                    llm_config["similarity_threshold"] = llm_options["similarity_threshold"].get("default", 0.7)
            if "system_prompt" in llm_options:
                value = render_config_field(
                    edited_config, options, ["text_matching", "llm", "system_prompt"],
                    "System Prompt"
                )
                if value is not None:
                    llm_config["system_prompt"] = value
                elif "system_prompt" not in llm_config:
                    llm_config["system_prompt"] = llm_options["system_prompt"].get("default", "")
            if "user_prompt_template" in llm_options:
                value = render_config_field(
                    edited_config, options, ["text_matching", "llm", "user_prompt_template"],
                    "User Prompt Template"
                )
                if value is not None:
                    llm_config["user_prompt_template"] = value
                elif "user_prompt_template" not in llm_config:
                    llm_config["user_prompt_template"] = llm_options["user_prompt_template"].get("default", "")
        
        # match_method == "exact"の場合は追加設定なし
    
    # 対話LLM設定
    with st.expander("Conversation LLM Settings", expanded=False):
        conv_llm_config = edited_config.setdefault("conversation_llm", {})
        conv_llm_options = options.get("conversation_llm", {})
        
        if "model" in conv_llm_options:
            value = render_config_field(
                edited_config, options, ["conversation_llm", "model"],
                "Model"
            )
            if value is not None:
                conv_llm_config["model"] = value
            elif "model" not in conv_llm_config:
                conv_llm_config["model"] = conv_llm_options["model"].get("default", "gpt-4o-mini")
        if "temperature" in conv_llm_options:
            value = render_config_field(
                edited_config, options, ["conversation_llm", "temperature"],
                "Temperature"
            )
            if value is not None:
                conv_llm_config["temperature"] = float(value)
            elif "temperature" not in conv_llm_config:
                conv_llm_config["temperature"] = conv_llm_options["temperature"].get("default", 0.0)
        if "system_prompt" in conv_llm_options:
            value = render_config_field(
                edited_config, options, ["conversation_llm", "system_prompt"],
                "System Prompt"
            )
            if value is not None:
                conv_llm_config["system_prompt"] = value
            elif "system_prompt" not in conv_llm_config:
                conv_llm_config["system_prompt"] = conv_llm_options["system_prompt"].get("default", "")
        if "user_prompt_template" in conv_llm_options:
            value = render_config_field(
                edited_config, options, ["conversation_llm", "user_prompt_template"],
                "User Prompt Template"
            )
            if value is not None:
                conv_llm_config["user_prompt_template"] = value
            elif "user_prompt_template" not in conv_llm_config:
                conv_llm_config["user_prompt_template"] = conv_llm_options["user_prompt_template"].get("default", "")
    
    # OpenAI設定
    with st.expander("OpenAI Settings", expanded=False):
        openai_config = edited_config.setdefault("openai", {})
        openai_options = options.get("openai", {})
        
        if "api_key" in openai_options:
            value = render_config_field(
                edited_config, options, ["openai", "api_key"],
                "API Key"
            )
            if value is not None:
                openai_config["api_key"] = value
            elif "api_key" not in openai_config:
                openai_config["api_key"] = openai_options["api_key"].get("default", "")
        
        if "api_base_url" in openai_options:
            value = render_config_field(
                edited_config, options, ["openai", "api_base_url"],
                "API Base URL"
            )
            if value is not None:
                openai_config["api_base_url"] = value
            elif "api_base_url" not in openai_config:
                openai_config["api_base_url"] = openai_options["api_base_url"].get("default", "https://api.openai.com/v1")
        
        if "model" in openai_options:
            value = render_config_field(
                edited_config, options, ["openai", "model"],
                "Model"
            )
            if value is not None:
                openai_config["model"] = value
            elif "model" not in openai_config:
                openai_config["model"] = openai_options["model"].get("default", "gpt-4o-mini")
    
    # WebSocket設定
    with st.expander("WebSocket Settings", expanded=False):
        websocket_config = edited_config.setdefault("websocket", {})
        websocket_options = options.get("websocket", {})
        
        if "server_url" in websocket_options:
            value = render_config_field(
                edited_config, options, ["websocket", "server_url"],
                "Server URL"
            )
            if value is not None:
                websocket_config["server_url"] = value
            elif "server_url" not in websocket_config:
                websocket_config["server_url"] = websocket_options["server_url"].get("default", "ws://localhost:8765/ws")
        
        if "sample_rate" in websocket_options:
            value = render_config_field(
                edited_config, options, ["websocket", "sample_rate"],
                "Sample Rate"
            )
            if value is not None:
                websocket_config["sample_rate"] = int(value)
            elif "sample_rate" not in websocket_config:
                websocket_config["sample_rate"] = websocket_options["sample_rate"].get("default", 24000)
        
        # お試し接続ボタン
        if st.button("🔌 Test Connection", use_container_width=True):
            server_url = websocket_config.get("server_url", "ws://localhost:8765/ws")
            try:
                import asyncio
                import aiohttp
                
                async def test_connection():
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.ws_connect(server_url, timeout=aiohttp.ClientTimeout(total=3)) as ws:
                                await ws.send_str("test")
                                return True
                    except Exception as e:
                        return str(e)
                
                result = asyncio.run(test_connection())
                if result is True:
                    st.success(f"✅ Connection successful: {server_url}")
                else:
                    st.error(f"❌ Connection failed: {result}")
            except Exception as e:
                st.error(f"❌ Connection test error: {e}")
    
    # ロギング設定
    with st.expander("Logging Settings", expanded=False):
        logging_config = edited_config.setdefault("logging", {})
        logging_options = options.get("logging", {})
        
        if "level" in logging_options:
            value = render_config_field(
                edited_config, options, ["logging", "level"],
                "Log Level"
            )
            if value is not None:
                logging_config["level"] = value
            elif "level" not in logging_config:
                logging_config["level"] = logging_options["level"].get("default", "INFO")
    
    # 元のconfig.jsonの値を保持（編集されていない項目も含む）
    # 編集された設定と元の設定をマージ
    final_config = json.loads(json.dumps(config))  # 元の設定をベースに
    # 編集された値を上書き
    for key, value in edited_config.items():
        if isinstance(value, dict) and isinstance(final_config.get(key), dict):
            final_config[key].update(value)
        else:
            final_config[key] = value
    
    return final_config
    
    return edited_config


def parse_output_line(line: str) -> Optional[Dict[str, Any]]:
    """出力行をパースして進捗情報を抽出"""
    # プログレスバーの情報を抽出
    progress_match = re.search(r'(\d+)%', line)
    if progress_match:
        return {"type": "progress", "value": int(progress_match.group(1))}
    
    # アクション情報を抽出
    action_match = re.search(r'現在=([^,]+)', line)
    if action_match:
        return {"type": "action", "value": action_match.group(1)}
    
    return None


def run_evaluation(scenario_file: str, config_file: Optional[str] = None, scenario_content: Optional[str] = None) -> tuple[subprocess.Popen, queue.Queue]:
    """評価を実行（subprocessでmain.pyを実行）
    
    Args:
        scenario_file: シナリオファイルのパス（元のファイル名、一時ファイルのパスを決定するために使用）
        config_file: 設定ファイルのパス（Noneの場合はconfig.jsonを使用）
        scenario_content: 編集されたシナリオコンテンツ（Noneの場合は元のファイルを使用）
    """
    # 編集されたコンテンツがある場合は一時ファイルに保存
    actual_scenario_file = scenario_file
    if scenario_content is not None:
        try:
            TEMP_SCENARIO_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(TEMP_SCENARIO_FILE, "w", encoding="utf-8") as f:
                f.write(scenario_content)
            actual_scenario_file = str(TEMP_SCENARIO_FILE)
        except Exception as e:
            # エラーが発生した場合は元のファイルを使用
            print(f"Warning: Failed to save temporary scenario file: {e}", file=sys.stderr)
            actual_scenario_file = scenario_file
    
    cmd = [sys.executable, str(MAIN_SCRIPT), actual_scenario_file]
    if config_file:
        cmd.append(config_file)
    
    # プロセスを起動
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(PROJECT_ROOT)
    )
    
    # 出力を読み取るためのキュー
    output_queue = queue.Queue()
    
    def read_output():
        """プロセスの出力を読み取る"""
        for line in process.stdout:
            output_queue.put(line.strip())
        output_queue.put(None)  # 終了マーカー
    
    # 別スレッドで出力を読み取る
    thread = threading.Thread(target=read_output, daemon=True)
    thread.start()
    
    return process, output_queue


def parse_results(output_lines: list) -> Dict[str, Any]:
    """出力から結果をパース"""
    results = {
        "turntake": {},
        "sound": {},
        "toolcall": {},
        "dialogue": {},
        "conversation_quality": {},
        "files": {}
    }
    
    current_section = None
    
    for line in output_lines:
        # セクション検出
        if "[turntake]" in line:
            current_section = "turntake"
            score_match = re.search(r'Score:\s*([\d.]+)/100', line)
            if score_match:
                results["turntake"]["score"] = float(score_match.group(1))
        elif "[sound]" in line:
            current_section = "sound"
            score_match = re.search(r'Score:\s*([\d.]+)/100', line)
            if score_match:
                results["sound"]["score"] = float(score_match.group(1))
        elif "[toolcall]" in line:
            current_section = "toolcall"
            score_match = re.search(r'Score:\s*([\d.]+)/100', line)
            if score_match:
                results["toolcall"]["score"] = float(score_match.group(1))
        elif "[dialogue]" in line:
            current_section = "dialogue"
            score_match = re.search(r'Score:\s*([\d.]+)/100', line)
            if score_match:
                results["dialogue"]["score"] = float(score_match.group(1))
        elif "[conversation_quality]" in line:
            current_section = "conversation_quality"
            score_match = re.search(r'Score:\s*([\d.]+)/100', line)
            if score_match:
                results["conversation_quality"]["score"] = float(score_match.group(1))
        
        # 詳細情報の抽出
        if current_section == "turntake":
            if "Response Latency:" in line:
                latency_match = re.search(r'Response Latency:\s*([\d.]+)ms', line)
                if latency_match:
                    results["turntake"]["response_latency_ms"] = float(latency_match.group(1))
            elif "Interrupt to Speech End:" in line:
                interrupt_match = re.search(r'Interrupt to Speech End:\s*([\d.]+)ms', line)
                if interrupt_match:
                    results["turntake"]["interrupt_to_speech_end_ms"] = float(interrupt_match.group(1))
        
        elif current_section == "sound":
            if "SNR:" in line:
                snr_match = re.search(r'SNR:\s*([\d.]+)dB', line)
                if snr_match:
                    results["sound"]["snr_db"] = float(snr_match.group(1))
        
        elif current_section == "conversation_quality":
            if "Backchannel Score:" in line:
                backchannel_match = re.search(r'Backchannel Score:\s*([\d.]+)/100', line)
                if backchannel_match:
                    results["conversation_quality"]["backchannel_score"] = float(backchannel_match.group(1))
            elif "Tone Consistency Score:" in line:
                tone_match = re.search(r'Tone Consistency Score:\s*([\d.]+)/100', line)
                if tone_match:
                    results["conversation_quality"]["tone_consistency_score"] = float(tone_match.group(1))
            elif "Omotenashi Score:" in line:
                omotenashi_match = re.search(r'Omotenashi Score:\s*(\d+)/5', line)
                if omotenashi_match:
                    results["conversation_quality"]["omotenashi_score"] = int(omotenashi_match.group(1))
        
        # ファイルパスの抽出
        if "Timeline saved to:" in line:
            timeline_match = re.search(r'Timeline saved to:\s*(.+)', line)
            if timeline_match:
                results["files"]["timeline"] = timeline_match.group(1).strip()
        elif "Log file:" in line:
            log_match = re.search(r'Log file:\s*(.+)', line)
            if log_match:
                results["files"]["log"] = log_match.group(1).strip()
        elif "Recording file:" in line:
            recording_match = re.search(r'Recording file:\s*(.+)', line)
            if recording_match:
                results["files"]["recording"] = recording_match.group(1).strip()
    
    return results


def display_results(results: Dict[str, Any]):
    """結果を表示"""
    st.header("📊 Evaluation Results")
    
    # スコアバーのスタイル
    st.markdown("""
    <style>
    .score-bar-container {
        margin: 0.2rem 0;
        padding: 0.2rem;
        width: 100%;
        box-sizing: border-box;
    }
    .score-label {
        font-weight: bold;
        margin-bottom: 0.5rem;
        font-size: 1rem;
    }
    .score-bar-wrapper {
        background: #f0f0f0;
        border-radius: 10px;
        height: 30px;
        position: relative;
        overflow: hidden;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
        width: 100%;
        box-sizing: border-box;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 10px;
        color: white;
        font-weight: bold;
        font-size: 0.9rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .score-bar-fill.green {
        background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
    }
    .score-bar-fill.orange {
        background: linear-gradient(90deg, #FF9800 0%, #F57C00 100%);
    }
    .score-bar-fill.red {
        background: linear-gradient(90deg, #F44336 0%, #D32F2F 100%);
    }
    .score-bar-fill.gray {
        background: linear-gradient(90deg, #9E9E9E 0%, #757575 100%);
    }
    </style>
    """, unsafe_allow_html=True)
    
    def get_score_color_class(score: Optional[float]) -> str:
        """スコアに応じた色クラスを返す"""
        if score is None:
            return "gray"
        if score >= 80.0:
            return "green"
        elif score >= 60.0:
            return "orange"
        else:
            return "red"
    
    # スコア表示（バー形式）
    turntake_score = results.get("turntake", {}).get("score")
    sound_score = results.get("sound", {}).get("score")
    toolcall_score = results.get("toolcall", {}).get("score")
    dialogue_score = results.get("dialogue", {}).get("score")
    conv_quality_score = results.get("conversation_quality", {}).get("score")
    
    scores = [
        ("Turn-taking", turntake_score),
        ("Sound", sound_score),
        ("Toolcall", toolcall_score),
        ("Dialogue", dialogue_score),
        ("Conversation Quality", conv_quality_score)
    ]
    
    # 各評価項目の説明
    score_descriptions = {
        "Turn-taking": "応答時間や割り込みの適切さを評価します。応答遅延が短く、適切なタイミングで応答できているかを測定します。",
        "Sound": "音声品質を評価します。SNR（信号対雑音比）やノイズプロファイル、STT信頼度を測定します。",
        "Toolcall": "ツールコールの適切さを評価します。期待されるツールコールが正しいタイミングで、適切な引数で実行されているかを測定します。",
        "Dialogue": "対話内容の適切さを評価します。期待される応答テキストと実際の応答テキストの一致度を測定します。",
        "Conversation Quality": "対話全体の品質を評価します。相槌の適切さ、トーンの一貫性、おもてなしスコアを総合的に測定します。"
    }
    
    for label, score in scores:
        color_class = get_score_color_class(score)
        width = min(100, max(0, score)) if score is not None else 0  # 0-100の範囲に制限
        score_text = f"{score:.1f}/100" if score is not None else "N/A"
        description = score_descriptions.get(label, "")
        
        # ラベルとツールチップアイコン
        if description:
            label_with_tooltip = f"""
            <div class='score-label'>
                {label}
                <span class="tooltip-icon" data-tooltip="{description.replace('"', '&quot;')}">?</span>
            </div>
            """
            st.markdown(label_with_tooltip, unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='score-label'>{label}</div>", unsafe_allow_html=True)
        
        score_html = f"""
        <div class="score-bar-container">
            <div class="score-bar-wrapper">
                <div class="score-bar-fill {color_class}" style="width: {width}%; max-width: 100%;">
                    {score_text if width > 15 else ""}
                </div>
            </div>
            {"<div style='text-align: right; margin-top: 0.25rem; font-size: 0.9rem; color: #666;'>" + score_text + "</div>" if width <= 15 else ""}
        </div>
        """
        st.markdown(score_html, unsafe_allow_html=True)
    
    # 詳細情報
    st.subheader("Details")
    
    # Turn-taking
    if results.get("turntake"):
        with st.expander("Turn-taking Details", expanded=False):
            turntake = results["turntake"]
            if "response_latency_ms" in turntake:
                st.write(f"Response Latency: {turntake['response_latency_ms']:.1f}ms")
            if "interrupt_to_speech_end_ms" in turntake:
                st.write(f"Interrupt to Speech End: {turntake['interrupt_to_speech_end_ms']:.1f}ms")
    
    # Sound
    if results.get("sound"):
        with st.expander("Sound Details", expanded=False):
            sound = results["sound"]
            if "snr_db" in sound:
                st.write(f"SNR: {sound['snr_db']:.1f}dB")
    
    # Conversation Quality
    if results.get("conversation_quality"):
        with st.expander("Conversation Quality Details", expanded=False):
            conv_quality = results["conversation_quality"]
            if "backchannel_score" in conv_quality:
                st.write(f"Backchannel Score: {conv_quality['backchannel_score']:.1f}/100")
            if "tone_consistency_score" in conv_quality:
                st.write(f"Tone Consistency Score: {conv_quality['tone_consistency_score']:.1f}/100")
            if "omotenashi_score" in conv_quality:
                st.write(f"Omotenashi Score: {conv_quality['omotenashi_score']}/5")
    
    # ファイルパス
    st.subheader("Output Files")
    files = results.get("files", {})
    if "timeline" in files:
        st.write(f"📄 Timeline: `{files['timeline']}`")
    if "log" in files:
        st.write(f"📝 Log: `{files['log']}`")
    if "recording" in files:
        recording_path = files['recording']
        st.write(f"🎵 Recording: `{recording_path}`")
        
        # オーディオプレーヤーを表示
        recording_file = Path(recording_path)
        if recording_file.exists():
            try:
                        # オーディオファイルを読み込んでプレーヤーに表示
                with open(recording_file, 'rb') as audio_file:
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/wav')
            except Exception as e:
                st.error(f"Could not load audio file: {e}")


def main():
    """メイン関数"""
    # ツールチップ用のCSSを注入
    inject_tooltip_css()
    
    # セッション状態の初期化
    if "config" not in st.session_state:
        try:
            st.session_state.config = load_config()
            # デバッグログ
            import logging
            logger = logging.getLogger(__name__)
            # logger.debug(f"[main] Loaded config: {list(st.session_state.config.keys())}")
            # logger.debug(f"[main] evaluation: {st.session_state.config.get('evaluation', {})}")
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()
    if "config_options" not in st.session_state:
        st.session_state.config_options = load_config_options()
        # デバッグログ
        import logging
        logger = logging.getLogger(__name__)
        # logger.debug(f"[main] Loaded config_options: {list(st.session_state.config_options.keys())}")
    if "scenario_file" not in st.session_state:
        st.session_state.scenario_file = None
    if "results" not in st.session_state:
        st.session_state.results = None
    if "output_lines" not in st.session_state:
        st.session_state.output_lines = []
    if "scenario_content" not in st.session_state:
        st.session_state.scenario_content = ""
    if "last_selected_scenario" not in st.session_state:
        st.session_state.last_selected_scenario = None
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "process" not in st.session_state:
        st.session_state.process = None
    if "output_queue" not in st.session_state:
        st.session_state.output_queue = None
    
    # タイトル
    st.title("🎤 VociMetrics")
    
    # 左右のカラムにpaddingとmarginを設定
    st.markdown("""
    <style>
    div[data-testid="column"] {
        padding: 1.5rem;
        margin: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 左右に分割
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        # 設定画面（編集可能）
        edited_config = edit_config(st.session_state.config, st.session_state.config_options)
        # 編集された設定をセッション状態に保存（右側のカラムで使用）
        st.session_state.edited_config = edited_config
        
        st.divider()
        
        # シナリオファイル選択
        scenario_files = get_scenario_files()
        if not scenario_files:
            st.error("No scenario files found in scenarios directory")
            st.stop()
        
        def load_scenario_content(scenario_name: str):
            """シナリオファイルの内容を読み込む"""
            scenario_file_path = SCENARIOS_DIR / scenario_name
            if scenario_file_path.exists():
                with open(scenario_file_path, "r", encoding="utf-8") as f:
                    return f.read()
            return ""
        
        # シナリオファイル選択とツールチップ
        scenario_help_text = "評価に使用するシナリオファイルを選択します。.convoファイル形式で、ユーザーとボットの対話を定義します。"
        scenario_label_html = f"""
        <div style="display: flex; align-items: center;">
            <span>Select Scenario File</span>
            <span class="tooltip-icon" data-tooltip="{scenario_help_text.replace('"', '&quot;')}">?</span>
        </div>
        """
        st.markdown(scenario_label_html, unsafe_allow_html=True)
        selected_scenario = st.selectbox(
            "",
            scenario_files,
            index=0 if "dialogue.convo" in scenario_files else 0,
            key="scenario_selectbox",
            label_visibility="collapsed"
        )
        
        # ファイル選択が変更された場合は内容を再読み込み
        if st.session_state.last_selected_scenario != selected_scenario:
            st.session_state.scenario_file = selected_scenario
            st.session_state.scenario_content = load_scenario_content(selected_scenario)
            st.session_state.last_selected_scenario = selected_scenario
        
        st.session_state.scenario_file = selected_scenario
        
        # シナリオファイルの内容を編集可能にする
        content_help_text = "シナリオファイルの内容を編集できます。#meでユーザー発話、#botでボット応答を定義します。#interruptで割り込み発話を定義できます。"
        content_label_html = f"""
        <div style="display: flex; align-items: center;">
            <h3>📝 Scenario Content</h3>
            <span class="tooltip-icon" data-tooltip="{content_help_text.replace('"', '&quot;')}" style="margin-left: 8px;">?</span>
        </div>
        """
        st.markdown(content_label_html, unsafe_allow_html=True)
        
        edited_scenario_content = st.text_area(
            "Edit scenario file content",
            value=st.session_state.scenario_content,
            height=400,
            key=f"scenario_content_editor_{selected_scenario}",
            label_visibility="collapsed"
        )
        
        # 内容が変更された場合はセッション状態を更新
        if edited_scenario_content != st.session_state.scenario_content:
            st.session_state.scenario_content = edited_scenario_content
    
    with right_col:
        # 実行/停止ボタン（右側の一番上）
        # ボタンのスタイルをカスタマイズ
        st.markdown("""
        <style>
        div[data-testid="stButton"] > button[kind="primary"],
        div[data-testid="stButton"] > button[kind="secondary"] {
            height: 80px !important;
            font-size: 1.5rem !important;
            font-weight: bold !important;
            padding: 1rem 2rem !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # 実行中かどうかでボタンのラベルと動作を切り替え
        run_button = False
        stop_button = False
        
        if st.session_state.is_running:
            stop_button = st.button("🛑 Stop Evaluation", type="secondary", use_container_width=True)
            if stop_button:
                process = st.session_state.process
                if process:
                    process.terminate()
                st.session_state.is_running = False
                st.rerun()
        else:
            run_button = st.button("▶️ Run Evaluation", type="primary", use_container_width=True)
        
        # 実行ボタンが押された場合の処理
        if run_button:
            if not st.session_state.scenario_file:
                st.error("Please select a scenario file first.")
            else:
                # 左側のカラムで編集された設定を取得（セッション状態から）
                edited_config = st.session_state.get("edited_config", st.session_state.config)
                # 編集された設定をconfig.jsonに保存してから実行
                save_config(edited_config, use_temp=False)
                st.session_state.config = edited_config
                # 一時ファイルにも保存（main.pyで使用）
                save_config(edited_config, use_temp=True)
                st.session_state.is_running = True
                st.session_state.output_lines = []
                st.session_state.progress_value = 0
                st.session_state.current_action = ""
                # 評価を実行（一時設定ファイルと編集されたシナリオコンテンツを使用）
                scenario_content = st.session_state.get("scenario_content", "")
                try:
                    process, output_queue = run_evaluation(st.session_state.scenario_file, str(TEMP_CONFIG_FILE), scenario_content)
                    st.session_state.process = process
                    st.session_state.output_queue = output_queue
                    st.rerun()
                except Exception as e:
                    import traceback
                    print(f"[ERROR] Failed to start evaluation: {e}")
                    print(traceback.format_exc())
                    st.session_state.is_running = False
        
        if st.session_state.is_running:
            # 実行中 - リッチなプログレス表示
            process = st.session_state.process
            output_queue = st.session_state.output_queue
            
            if not process or not output_queue:
                st.warning("⚠️ Process or output queue not initialized. Please try running again.")
                if st.button("🛑 Stop", use_container_width=True):
                    st.session_state.is_running = False
                    st.rerun()
            elif process and output_queue:
                # プロセスが既に終了している場合（エラーなど）
                if process.poll() is not None:
                    # 残りの出力を読み取る
                    output_lines = st.session_state.get("output_lines", [])
                    remaining_lines = []
                    while True:
                        try:
                            line = output_queue.get(timeout=0.1)
                            if line is None:
                                break
                            remaining_lines.append(line)
                        except queue.Empty:
                            break
                    output_lines.extend(remaining_lines)
                    st.session_state.output_lines = output_lines
                    
                    # プロセスが正常終了したか確認
                    if process.returncode != 0:
                        print(f"[ERROR] Evaluation failed with return code {process.returncode}")
                        if output_lines:
                            print("[ERROR] Output:")
                            for line in output_lines:
                                print(f"  {line}")
                        else:
                            print("[ERROR] No output captured. The process may have failed before producing any output.")
                        st.session_state.is_running = False
                        st.rerun()
                    else:
                        # 結果をパース
                        st.session_state.results = parse_results(output_lines)
                        st.session_state.is_running = False
                        st.rerun()
                else:
                    # プロセスが実行中 - リッチなプログレス表示
                    output_lines = st.session_state.get("output_lines", [])
                    progress_value = st.session_state.get("progress_value", 0)
                    current_action = st.session_state.get("current_action", "")
                    
                    # 出力を読み取り（ログは保存せず、プログレス情報のみ抽出）
                    try:
                        line = output_queue.get(timeout=0.1)
                        if line is not None:  # 終了マーカーでない場合
                            # プログレス情報を抽出
                            parsed = parse_output_line(line)
                            if parsed:
                                if parsed["type"] == "progress":
                                    progress_value = parsed["value"]
                                    st.session_state.progress_value = progress_value
                                elif parsed["type"] == "action":
                                    current_action = parsed["value"]
                                    st.session_state.current_action = current_action
                            # 結果パース用に重要な行のみ保存（スコアやファイルパスなど）
                            if any(keyword in line for keyword in ["Score:", "Timeline saved to:", "Log file:", "Recording file:"]):
                                output_lines.append(line)
                                st.session_state.output_lines = output_lines
                    except queue.Empty:
                        pass
                    
                    # リッチなプログレス表示 - HTML/CSSアニメーション
                    # CSSスタイルを先に適用
                    st.markdown("""
                    <style>
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                    @keyframes pulse {
                        0%, 100% { transform: scale(1); opacity: 1; }
                        50% { transform: scale(1.1); opacity: 0.8; }
                    }
                    @keyframes wave {
                        0%, 100% { height: 30px; }
                        50% { height: 100px; }
                    }
                    @keyframes float {
                        0%, 100% { transform: translateY(0px); }
                        50% { transform: translateY(-10px); }
                    }
                    @keyframes gradient {
                        0% { background-position: 0% 50%; }
                        50% { background-position: 100% 50%; }
                        100% { background-position: 0% 50%; }
                    }
                    @keyframes shimmer {
                        0% { background-position: -1000px 0; }
                        100% { background-position: 1000px 0; }
                    }
                    @keyframes bounce {
                        0%, 100% { transform: translateY(0); }
                        25% { transform: translateY(-15px); }
                        50% { transform: translateY(0); }
                        75% { transform: translateY(-8px); }
                    }
                    
                    @keyframes bounce-circle {
                        0% { 
                            transform: translateY(0) scale(1);
                        }
                        20% { 
                            transform: translateY(-40px) scale(1.2);
                        }
                        40% { 
                            transform: translateY(0) scale(1);
                        }
                        60% { 
                            transform: translateY(-20px) scale(1.1);
                        }
                        80% { 
                            transform: translateY(0) scale(1);
                        }
                        100% { 
                            transform: translateY(0) scale(1);
                        }
                    }
                    
                    @keyframes bounce-circle-delayed {
                        0% { 
                            transform: translateY(0) scale(1);
                        }
                        20% { 
                            transform: translateY(-35px) scale(1.15);
                        }
                        40% { 
                            transform: translateY(0) scale(1);
                        }
                        60% { 
                            transform: translateY(-18px) scale(1.08);
                        }
                        80% { 
                            transform: translateY(0) scale(1);
                        }
                        100% { 
                            transform: translateY(0) scale(1);
                        }
                    }
                    
                    .progress-container {
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        background-size: 200% 200%;
                        animation: gradient 5s ease infinite;
                        padding: 2rem;
                        border-radius: 20px;
                        box-shadow: 0 15px 40px rgba(0,0,0,0.3);
                        margin: 1.5rem auto;
                        max-width: 600px;
                        position: relative;
                        overflow: hidden;
                    }
                    
                    .progress-container::before {
                        content: '';
                        position: absolute;
                        top: -50%;
                        left: -50%;
                        width: 200%;
                        height: 200%;
                        background: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px);
                        background-size: 30px 30px;
                        animation: spin 20s linear infinite;
                    }
                    
                    .progress-header {
                        color: white;
                        font-size: 1.8rem;
                        font-weight: bold;
                        margin-bottom: 1.5rem;
                        text-align: center;
                        position: relative;
                        z-index: 1;
                        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                    }
                    
                    .spinning-icon {
                        display: inline-block;
                        animation: spin 2s linear infinite, pulse 2s ease-in-out infinite;
                        font-size: 2.5em;
                        margin-right: 0.5rem;
                    }
                    
                    .wave-container {
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        gap: 6px;
                        height: 120px;
                        margin: 2rem 0;
                        position: relative;
                        z-index: 1;
                    }
                    
                    .wave-bar {
                        width: 6px;
                        background: rgba(255,255,255,0.9);
                        border-radius: 3px;
                        animation: wave 1.2s ease-in-out infinite;
                        box-shadow: 0 0 10px rgba(255,255,255,0.5);
                    }
                    
                    .wave-bar:nth-child(1) { animation-delay: 0s; }
                    .wave-bar:nth-child(2) { animation-delay: 0.1s; }
                    .wave-bar:nth-child(3) { animation-delay: 0.2s; }
                    .wave-bar:nth-child(4) { animation-delay: 0.3s; }
                    .wave-bar:nth-child(5) { animation-delay: 0.4s; }
                    .wave-bar:nth-child(6) { animation-delay: 0.5s; }
                    .wave-bar:nth-child(7) { animation-delay: 0.6s; }
                    .wave-bar:nth-child(8) { animation-delay: 0.7s; }
                    .wave-bar:nth-child(9) { animation-delay: 0.8s; }
                    .wave-bar:nth-child(10) { animation-delay: 0.9s; }
                    
                    .progress-bar-container {
                        background: rgba(255,255,255,0.25);
                        border-radius: 15px;
                        padding: 0.8rem;
                        margin: 1.5rem 0;
                        position: relative;
                        z-index: 1;
                        backdrop-filter: blur(10px);
                    }
                    
                    .progress-bar-fill {
                        background: #00f260;
                        height: 35px;
                        border-radius: 12px;
                        transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        color: white;
                        font-weight: bold;
                        font-size: 1rem;
                        box-shadow: 0 4px 15px rgba(0,242,96,0.4);
                    }
                    
                    .status-text {
                        color: white;
                        font-size: 1.2rem;
                        margin-top: 1.5rem;
                        text-align: center;
                        font-weight: 500;
                        position: relative;
                        z-index: 1;
                        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
                        animation: float 3s ease-in-out infinite;
                    }
                    
                    .dots-container {
                        display: inline-flex;
                        gap: 8px;
                        margin-left: 10px;
                    }
                    
                    .dot {
                        width: 8px;
                        height: 8px;
                        background: white;
                        border-radius: 50%;
                        animation: bounce 1.4s ease-in-out infinite;
                    }
                    
                    .dot:nth-child(1) { animation-delay: 0s; }
                    .dot:nth-child(2) { animation-delay: 0.2s; }
                    .dot:nth-child(3) { animation-delay: 0.4s; }
                    
                    .bouncing-circles {
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        gap: 20px;
                        margin: 2rem 0;
                        position: relative;
                        z-index: 1;
                    }
                    
                    .bouncing-circle {
                        width: 60px;
                        height: 60px;
                        border-radius: 50%;
                        background: radial-gradient(circle at 30% 30%, rgba(255,255,255,1) 0%, rgba(255,255,255,0.7) 50%, rgba(255,255,255,0.4) 100%);
                        box-shadow: 0 8px 25px rgba(255,255,255,0.6),
                                    0 0 40px rgba(255,255,255,0.4),
                                    inset 0 3px 8px rgba(255,255,255,0.9),
                                    inset 0 -3px 8px rgba(0,0,0,0.2);
                        animation: bounce-circle 1.2s cubic-bezier(0.68, -0.55, 0.265, 1.55) infinite;
                        position: relative;
                        will-change: transform;
                    }
                    
                    .bouncing-circle::before {
                        content: '';
                        position: absolute;
                        top: 50%;
                        left: 50%;
                        transform: translate(-50%, -50%);
                        width: 30px;
                        height: 30px;
                        border-radius: 50%;
                        background: radial-gradient(circle, rgba(255,255,255,0.8) 0%, rgba(255,255,255,0.2) 100%);
                    }
                    
                    .bouncing-circle:nth-child(1) { 
                        animation: bounce-circle 1.2s cubic-bezier(0.68, -0.55, 0.265, 1.55) infinite;
                        animation-delay: 0s;
                    }
                    .bouncing-circle:nth-child(2) { 
                        animation: bounce-circle-delayed 1.2s cubic-bezier(0.68, -0.55, 0.265, 1.55) infinite;
                        animation-delay: 0.2s;
                    }
                    .bouncing-circle:nth-child(3) { 
                        animation: bounce-circle 1.2s cubic-bezier(0.68, -0.55, 0.265, 1.55) infinite;
                        animation-delay: 0.4s;
                    }
                    .bouncing-circle:nth-child(4) { 
                        animation: bounce-circle-delayed 1.2s cubic-bezier(0.68, -0.55, 0.265, 1.55) infinite;
                        animation-delay: 0.6s;
                    }
                    .bouncing-circle:nth-child(5) { 
                        animation: bounce-circle 1.2s cubic-bezier(0.68, -0.55, 0.265, 1.55) infinite;
                        animation-delay: 0.8s;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # プログレスコンテナ
                    progress_width = max(5, progress_value)
                    
                    # HTMLを1行にまとめて、st.markdownで表示（音波バーのみ）
                    progress_html = (
                        '<div class="progress-container">'
                        '<div class="progress-header">Evaluation in Progress</div>'
                        '<div class="wave-container">'
                        '<div class="wave-bar"></div>'
                        '<div class="wave-bar"></div>'
                        '<div class="wave-bar"></div>'
                        '<div class="wave-bar"></div>'
                        '<div class="wave-bar"></div>'
                        '<div class="wave-bar"></div>'
                        '<div class="wave-bar"></div>'
                        '<div class="wave-bar"></div>'
                        '<div class="wave-bar"></div>'
                        '<div class="wave-bar"></div>'
                        '</div>'
                        f'<div class="progress-bar-container">'
                        f'<div class="progress-bar-fill" style="width: {progress_width}%;">{progress_value}%</div>'
                        '</div>'
                        '</div>'
                    )
                    st.markdown(progress_html, unsafe_allow_html=True)
                
                    # 自動更新のため再実行（プロセスが実行中の場合のみ）
                    if process.poll() is None:  # プロセスがまだ実行中
                        time.sleep(0.1)
                        st.rerun()
                    else:
                        # プロセスが終了したが、まだ出力が残っている可能性がある
                        st.rerun()
        
        elif st.session_state.results:
            # 結果表示
            display_results(st.session_state.results)
            
            # クリアボタン
            if st.button("🗑️ Clear Results", use_container_width=True):
                st.session_state.results = None
                st.session_state.output_lines = []
                st.session_state.progress_value = 0
                st.session_state.current_action = ""
                st.rerun()
        else:
            # 待機中（何も表示しない）
            pass


if __name__ == "__main__":
    main()

