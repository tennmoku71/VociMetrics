# VociMetrics

A comprehensive evaluation framework for voice dialogue systems that automatically measures conversation quality, response timing, and system behavior.

## Overview

VociMetrics is a tool for evaluating the quality of voice dialogue systems. It automatically measures the following evaluation metrics:

- **Turn-taking**: Response latency, interrupt handling, speech duration
- **Sound**: SNR (Signal-to-Noise Ratio), noise profiling, STT confidence
- **Toolcall**: Toolcall presence, count, arguments, and latency
- **Dialogue**: Text comparison (exact match, edit distance, LLM-based evaluation)
- **Conversation Quality**: LLM-based overall conversation quality evaluation (backchannel appropriateness, tone consistency, omotenashi score)

## Features

- **WebSocket-based Communication**: Real-time bidirectional audio streaming via WebSocket
- **Voice Activity Detection (VAD)**: Uses `webrtcvad` for accurate speech start/end detection
- **Speech-to-Text (STT)**: Supports multiple STT engines (Google Web Speech API, Vosk)
- **Text-to-Speech (TTS)**: Automatic TTS conversion for scenario files using natural language
- **LLM Evaluation**: OpenAI API integration for semantic text comparison and conversation quality assessment
- **Interactive GUI**: Streamlit-based web interface with tooltips and visual progress indicators
- **Comprehensive Logging**: Timeline JSON, detailed logs, and stereo WAV recordings

## Quick Start

### GUI (Recommended)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the WebSocket server** (in one terminal):
   ```bash
   python tests/simple_response.py
   ```

3. **Launch the GUI** (in another terminal):
   ```bash
   streamlit run gui.py
   ```

4. **Configure and run:**
   - Open the browser (automatically opens)
   - Configure settings in the GUI (including OpenAI API key if needed)
   - Select a scenario file
   - Click "Run Evaluation"

That's it! The evaluation results will be displayed in the GUI and saved in the `reports/` directory.

### CLI

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables** (for LLM evaluation):
   Create a `.env` file:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Start the WebSocket server** (in one terminal):
   ```bash
   python tests/simple_response.py
   ```

4. **Run an evaluation** (in another terminal):
   ```bash
   python main.py scenarios/dialogue.convo
   ```

The evaluation results will be saved in the `reports/` directory.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file and set your OpenAI API key:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

Alternatively, you can set the API key in `config.json` under the `openai` section.

### 3. Configuration

Review and modify `config.json` as needed. Key settings include:

- **WebSocket**: Server URL and sample rate (default: 24000Hz for OpenAI Realtime API compatibility)
- **STT**: Engine type and language
- **TTS**: Engine type and language
- **Evaluation**: Thresholds for response latency, toolcall latency, and sound quality
- **Text Matching**: Method (exact, edit_distance, or llm) and LLM settings

## Usage

### GUI Application (Recommended)

Launch the Streamlit-based GUI:

```bash
streamlit run gui.py
```

The browser will automatically open with the following interface:

1. **Configuration Panel**: Edit settings from `config.json` with helpful tooltips
2. **Scenario Selection**: Choose a scenario file and edit its content
3. **Run Evaluation**: Execute the evaluation and view progress
4. **Results Display**: View scores, metrics, and recorded audio

### Command Line Execution

```bash
# Run default scenario (dialogue.convo)
python main.py

# Specify a scenario file
python main.py scenarios/tool_call.convo

# Use custom config file (for GUI-generated configs)
python main.py scenarios/dialogue.convo _config.gui.json
```

## Server Setup

### Mock Servers (for Testing)

Before running evaluations, you need to start a WebSocket server. For testing purposes, you can use the provided mock servers:

**Simple Response Server:**
```bash
python tests/simple_response.py
```

**Toolcall-Enabled Server:**
```bash
python tests/tool_call.py
```

These mock servers are temporary implementations for testing. For production evaluations, you should connect your actual dialogue system.

### Connecting Your Dialogue System

To evaluate your own dialogue system, you need to implement a WebSocket server that follows the VociMetrics protocol specification below.

## WebSocket Protocol Specification

VociMetrics communicates with dialogue systems via WebSocket. Your dialogue system must implement the following protocol:

### Connection

- **URL**: `ws://localhost:8765/ws` (default, configurable in `config.json`)
- **Protocol**: WebSocket (RFC 6455)

### Audio Streaming (Binary Messages)

**Format:**
- **Encoding**: PCM (Pulse Code Modulation)
- **Sample Rate**: 24000 Hz (OpenAI Realtime API compatible)
- **Bit Depth**: 16-bit signed integer (int16)
- **Channels**: Mono (1 channel)
- **Endianness**: Little-endian
- **Chunk Size**: 10ms chunks (240 samples per chunk)
- **Continuous Streaming**: Both client and server must continuously send audio data, including silence when no speech is present

**Client → Server (User Audio):**
- Send audio chunks continuously at 10ms intervals
- Send silence chunks when user is not speaking
- Audio chunks are sent as binary WebSocket messages

**Server → Client (Bot Audio):**
- Send audio chunks continuously at 10ms intervals
- Send silence chunks when bot is not speaking
- Audio chunks are sent as binary WebSocket messages

**Example (Python):**
```python
import numpy as np
import aiohttp

# Send audio chunk (10ms at 24000Hz = 240 samples)
chunk = np.zeros(240, dtype=np.int16)  # Silence
await ws.send_bytes(chunk.tobytes())

# Or send actual audio data
audio_data = np.array([...], dtype=np.int16)  # Your audio samples
for i in range(0, len(audio_data), 240):
    chunk = audio_data[i:i+240]
    if len(chunk) < 240:
        # Pad with zeros if needed
        padded = np.zeros(240, dtype=np.int16)
        padded[:len(chunk)] = chunk
        chunk = padded
    await ws.send_bytes(chunk.tobytes())
```

### Text Messages (JSON)

Text messages are used for toolcall communication and control.

#### Toolcall Message (Server → Client)

When your dialogue system needs to execute a toolcall (e.g., API call, function call), send a JSON message:

```json
{
  "type": "toolcall",
  "id": "unique_toolcall_id",
  "name": "tool_name",
  "arguments": {
    "arg1": "value1",
    "arg2": "value2"
  }
}
```

**Fields:**
- `type`: Must be `"toolcall"`
- `id`: Unique identifier for this toolcall (string)
- `name`: Name of the tool/function to call (string)
- `arguments`: JSON object containing toolcall arguments

**Example:**
```python
import json

toolcall_message = {
    "type": "toolcall",
    "id": "call_staff_1234567890",
    "name": "call_staff",
    "arguments": {
        "reason": "user_inquiry",
        "priority": "normal"
    }
}
await ws.send_str(json.dumps(toolcall_message))
```

#### Toolcall Result (Client → Server, Optional)

The client may send toolcall results back to the server:

```json
{
  "type": "toolcall_result",
  "id": "unique_toolcall_id",
  "status": "success" | "error",
  "result": {...},  // if status is "success"
  "error": "error_message"  // if status is "error"
}
```

### Implementation Requirements

1. **Continuous Audio Streaming**: Your server must continuously send audio chunks (10ms intervals), even when silent. This ensures proper synchronization and VAD detection.

2. **VAD Detection**: The client uses VAD to detect when user speech ends. Your server should also use VAD to detect when to start responding.

3. **Error Handling**: Handle WebSocket connection errors gracefully. Close connections cleanly on shutdown.

4. **Sample Rate**: Ensure your audio processing pipeline uses 24000Hz sample rate. If your system uses a different rate, resample before sending.

### Example Server Implementation

```python
import asyncio
import json
import numpy as np
from aiohttp import web, WSMsgType

async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    sample_rate = 24000
    chunk_size = int(sample_rate * 0.01)  # 10ms
    
    async def audio_sender():
        """Continuously send audio chunks"""
        while True:
            # Get audio chunk from your TTS/audio system
            # If no audio, send silence
            chunk = get_audio_chunk() or np.zeros(chunk_size, dtype=np.int16)
            await ws.send_bytes(chunk.tobytes())
            await asyncio.sleep(0.01)  # 10ms
    
    async def audio_receiver():
        """Receive audio chunks from client"""
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                audio_chunk = np.frombuffer(msg.data, dtype=np.int16)
                # Process audio with your STT/VAD system
                process_audio(audio_chunk)
            elif msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data.get("type") == "toolcall_result":
                    handle_toolcall_result(data)
    
    # Run both tasks concurrently
    await asyncio.gather(
        audio_sender(),
        audio_receiver()
    )
    
    return ws

app = web.Application()
app.router.add_get('/ws', websocket_handler)
web.run_app(app, port=8765)
```

### Testing Your Implementation

1. Start your dialogue system server
2. Update `config.json` with your server URL if different from default
3. Run VociMetrics evaluation:
   ```bash
   python main.py scenarios/dialogue.convo
   ```
4. Check the evaluation results in `reports/` directory

## Scenario Files (.convo)

Scenario files define the dialogue flow and are placed in the `scenarios/` directory.

### Basic Syntax

```
#me <text_or_audio_file>
#bot <expected_text>
#bot [speechStart]
#bot [speechEnd]
#toolcall <name> <arguments_json>
#interrupt <delay_ms> <text_or_audio_file>
```

### Examples

**Natural Language Text (with TTS):**
```
#me こんにちは、お元気ですか？
#bot こんにちは、元気です。ありがとうございます。

#me 今日はいい天気ですね
#bot そうですね、とてもいい天気です。
```

**Audio Files:**
```
#me tests/hello.wav
#bot [speechStart]
#bot [speechEnd]
```

**Toolcall:**
```
#me この商品について教えてください
#bot 申し訳ございません。詳しい情報はスタッフにお尋ねください。
#toolcall call_staff {"reason": "user_inquiry", "priority": "normal"}
```

**Interrupt:**
```
#me こんにちは
#bot こんにちは、いらっしゃいませ。本日はどのようなご用件でしょうか。
#interrupt 500 すみません、ちょっと待ってください
#bot [speechEnd]
```

### Syntax Details

- **`#me`**: User speech (text is converted to audio via TTS before execution)
- **`#bot`**: Bot response (with expected text or `[speechStart]`/`[speechEnd]` markers)
- **`#toolcall`**: Expected toolcall with name and JSON arguments
- **`#interrupt`**: Interrupt speech after `delay_ms` milliseconds from `BOT_SPEECH_START`

## Output Files

After evaluation execution, the following files are generated in the `reports/` directory:

- **Timeline JSON**: `reports/test_XXXXX_timeline.json` - Chronological event log with timestamps
- **Log File**: `reports/test_XXXXX.log` - Detailed execution log
- **Recording WAV**: `reports/test_XXXXX_recording.wav` - Stereo recording (left: user audio, right: bot audio)

## Evaluation Metrics

### Turn-taking Score

- **Response Latency**: Time from `USER_SPEECH_END` to `BOT_SPEECH_START` (threshold: 800ms default)
- **Interrupt to Speech End**: Time from `USER_INTERRUPT_START` to `BOT_SPEECH_END`
- **User/Bot Speech Duration**: Duration of speech segments

### Sound Score

- **SNR**: Signal-to-Noise Ratio calculated from VAD ON/OFF segments
- **Noise Profiling**: FFT analysis of noise segments (stationary, low-freq, high-freq, mixed)
- **STT Confidence**: Confidence scores from STT engine (if available)

### Toolcall Score

- **Presence**: Whether expected toolcall was sent (30%)
- **Count Match**: Number of toolcalls matches expected (25%)
- **Arguments Match**: Arguments match expected (25%)
- **Latency**: Time from `BOT_SPEECH_END` to `TOOLCALL` (20%)

### Dialogue Score

- **Exact Match**: Character-by-character comparison
- **Edit Distance**: Levenshtein distance-based similarity
- **LLM Evaluation**: Semantic naturalness evaluation (0.0-1.0 score)

### Conversation Quality Score

- **Backchannel Score**: Appropriateness of backchannels and fillers (0.0-1.0)
- **Tone Consistency Score**: Consistency of tone before/after interruptions or API calls (0.0-1.0)
- **Omotenashi Score**: Overall "comfort" score (1-5 integer)

## Configuration

### Key Settings

Configuration is managed through `config.json`. Key configuration areas include:

**VAD Settings** (in server and client code):
- Voice Activity Detection parameters can be adjusted for sensitivity and interrupt handling
- Settings include threshold, minimum speech duration, and minimum silence duration

**WebSocket**:
- Server URL and sample rate configuration
- Default sample rate is 24000Hz for OpenAI Realtime API compatibility

**Evaluation Thresholds**:
- Response latency thresholds
- Toolcall latency thresholds
- Sound quality thresholds (SNR, etc.)

For detailed configuration options, see `config.json` and `config_options.json` (for GUI field definitions).

## Troubleshooting

### Server Not Responding

- Ensure the WebSocket server is running before starting the evaluation
- Check that the server URL in `config.json` matches the server's address
- Verify the server is listening on the correct port (default: 8765)

### VAD Not Detecting Speech

- Check audio sample rate matches VAD requirements (16000Hz for webrtcvad)
- Verify audio levels are sufficient
- Adjust `threshold` and `min_speech_duration_ms` if needed

### STT Not Working

- For Google Web Speech API: Check internet connection
- For Vosk: Ensure model files are downloaded and path is correct in config
- Verify language settings match the audio language

### LLM Evaluation Errors

- Verify OpenAI API key is set in `.env` or `config.json`
- Check API quota and rate limits
- Ensure model name is correct (e.g., `gpt-4o-mini`)

