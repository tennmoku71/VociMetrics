# Interactive Voice Evaluator (IVE)

音声対話システムの評価フレームワーク

## 概要

Interactive Voice Evaluator (IVE) は、音声対話システムの品質を評価するためのツールです。以下の評価項目を自動で測定します：

- **Turn-taking**: 応答レイテンシ、割り込み対応
- **Sound**: SNR、ノイズプロファイリング、STT信頼度
- **Toolcall**: Toolcallの送信有無、回数、引数、レイテンシ
- **Dialogue**: テキスト比較（完全一致、編集距離、LLM評価）
- **Conversation Quality**: LLMによる対話全体の品質評価（相槌・フィラー、トーン一貫性、おもてなしスコア）

## セットアップ

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 2. 環境変数の設定

`.env`ファイルを作成し、OpenAI APIキーを設定してください：

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. 設定ファイルの確認

`config.json`を確認し、必要に応じて設定を変更してください。

## 使用方法

### GUIアプリケーション（推奨）

StreamlitベースのGUIアプリケーションを使用できます：

```bash
streamlit run gui.py
```

ブラウザが自動的に開き、以下の画面が表示されます：

1. **設定画面**: `config.json`の設定項目を表示
2. **実行画面**: シナリオファイルを選択して実行ボタンをクリック
3. **結果画面**: 評価結果を表示

### コマンドライン実行

```bash
# デフォルトシナリオ（dialogue.convo）を実行
python main.py

# 特定のシナリオファイルを指定
python main.py scenarios/tool_call.convo
```

## サーバーの起動

評価を実行する前に、WebSocketサーバーを起動する必要があります。

### シンプルな応答サーバー

```bash
python tests/simple_response.py
```

### Toolcall対応サーバー

```bash
python tests/tool_call.py
```

## シナリオファイル（.convo）

シナリオファイルは`scenarios/`ディレクトリに配置します。詳細は`scenarios/README.md`を参照してください。

### 例

```
#me こんにちは、お元気ですか？
#bot こんにちは、元気です。ありがとうございます。

#me 今日はいい天気ですね
#bot そうですね、とてもいい天気です。
```

## 出力ファイル

評価実行後、以下のファイルが生成されます：

- **Timeline JSON**: `reports/test_XXXXX_timeline.json` - イベントの時系列ログ
- **Log File**: `logs/test_XXXXX.log` - 詳細なログ
- **Recording WAV**: `reports/test_XXXXX_recording.wav` - 2チャンネル録音（左: ユーザー音声、右: ボット音声）

## ディレクトリ構造

```
interactive-voice-evaluator/
├── app/                    # GUIアプリケーション
│   └── gui.py             # StreamlitベースのGUI
├── evaluator/             # 評価エンジン
│   ├── evaluator.py      # メイン評価エンジン
│   ├── llm_evaluator.py  # LLMによる対話評価
│   ├── sound_evaluator.py # 音声品質評価
│   ├── stt_engine.py      # STTエンジン
│   ├── text_matcher.py   # テキスト比較エンジン
│   └── tts_engine.py      # TTSエンジン
├── parser/               # パーサー
│   └── convo_parser.py   # .convoファイルパーサー
├── tester/               # テスト実行
│   ├── orchestrator.py   # オーケストレーター
│   └── vad_detector.py   # VAD検出器
├── tests/                # テストサーバー
│   ├── simple_response.py # シンプルな応答サーバー
│   └── tool_call.py      # Toolcall対応サーバー
├── scenarios/            # シナリオファイル
├── reports/              # 評価結果（.gitignore）
├── logs/                 # ログファイル
├── config.json           # 設定ファイル
└── main.py              # メインエントリーポイント
```

## ライセンス

（ライセンス情報を追加してください）

