## 音声対話UX評価システム：Interactive Voice Evaluator (IVE) 仕様書

## 1. 概要

本システムは、音声対話システム（Voice Dialogue System）において、従来のテキストベースのテストでは困難だった「対話のリズム」「API連携のタイミング」「人間が感じる心地よさ」を、ルール・人間・LLMの3層で実行・評価するフレームワークである。

---

## 2. テスト実行レイヤー (Execution Layers)

テストの入力（発話）を生成・実行する主体を3種類定義する。

| **レイヤー** | **実行主体** | **目的・内容** |
| --- | --- | --- |
| **1. ルールテスト** | 自動スクリプト | 規定の音声ファイルを再生。割り込み（Barge-in）や沈黙のシミュレーションをミリ秒単位で制御。 |
| **2. 人間テスト** | テスター（人間） | 実際に対話し、偶発的な言い間違い、長い発話、感情的なトーンなどへの反応を検証。 |
| **3. LLMテスト** | Audio2Audio AI | 音声モデル（GPT-4o等）を使用。AI同士を対話させ、非定型で無限に近い会話パターンを自動生成。 |

---

## 3. 評価レイヤー (Evaluation Layers)

実行された結果（ログ・音源）を解析・判定する主体を3種類定義する。

| **レイヤー** | **評価主体** | **判定基準・方法** |
| --- | --- | --- |
| **1. ルール評価** | 解析エンジン | **時系列ログの定量的判定。** 応答速度、API引数の正確性、発話重複率などを数値で算出。 |
| **2. 人間評価** | レビューアー | **録音データの聴取。** 実際の「心地よさ」「親しみやすさ」「違和感」をMOS（平均意見評点）でスコアリング。 |
| **3. LLM評価** | Text-based LLM | **テキストログの定性解析。** コンテキストの理解、相槌の適切さ、FAQとしての正確性を評価。 |

---

## 4. データ仕様：統合時系列ログ (Unified Event Timeline)

すべての評価の根拠となるデータ形式。音声波形から得たイベントとAPIイベントを10ms単位で同期する。

JSON

`{
  "test_metadata": { "id": "test_001", "type": "LLM_Execution", "eval_status": "pending" },
  "events": [
    { "time": 0,    "type": "USER_SPEECH_START", "text": "予約をキャンセルしたいです" },
    { "time": 1800, "type": "USER_SPEECH_END" },
    { "time": 2000, "type": "API_CALL_START",    "function": "cancel_order", "args": {"id": "123"} },
    { "time": 2500, "type": "API_CALL_END",      "status": "success" },
    { "time": 2800, "type": "BOT_SPEECH_START",  "text": "かしこまりました。キャンセルを承りました。" },
    { "time": 4000, "type": "BOT_SPEECH_END" }
  ]
}`

---

## 5. コンポーネント構成

### 5.1 オーケストレーター (The Conductor)

- テストシナリオをパースし、各実行レイヤーを起動。
- APIモックサーバーとして機能し、ターゲットシステムからのFunction Callingを受信、記録する。

### 5.2 音声フロントエンド (Virtual Client)

- Playwright/WebRTCベース。ブラウザの仮想マイクを制御。
- **VAD (Voice Activity Detection)** を搭載し、システムの音声をリアルタイムで検知してタイムスタンプを打刻。

### 5.3 統合評価エンジン (The Judge)

- **Rule Engine**: ミリ秒単位の「間」を計算し、規定値（例：800ms以内）との乖離を算出。
- **LLM Evaluator**: Whisperで文字起こしした全ログをLLMに送り、UX品質をレポート。
- **Human Review UI**: 音声波形と時系列ログを並べて表示し、人間が評価を入力できる管理画面。

---

## 6. 主要評価指標 (Key Performance Indicators)

1. **Response Latency (RL)**: ユーザー発話終了からボット発話開始までの時間。
2. **Think Time (TT)**: APIレスポンス受信からボット発話開始までの時間。
3. **Barge-in Reaction (BR)**: 割り込み検知から発話停止までの速度。
4. **Function Success Rate (FSR)**: API実行の有無および引数の正解率。
5. **Comfort Score (CS)**: 人間およびLLMによる「対話の心地よさ」の総合評価点。

---

## 7. 運用フロー

1. **定義**: Botiumライクなシナリオファイルに、実行タイプ（ルール/人間/LLM）を記述。
2. **実行**: 指定されたレイヤーでテストを実行。APIコールと音声を同時記録。
3. **一次評価**: ルールエンジンが機械的に異常値（API失敗、過度な遅延）をフラグ立て。
4. **二次評価**: LLMが会話の質を評価。
5. **三次評価**: 必要に応じて人間が録音を聴き、最終的なUXスコアを確定。


1. 拡張仕様書：Interactive Voice Evaluator (IVE)
1.1 システムアーキテクチャ
IVEは「Orchestrator（指揮者）」を中心に、音声ストリームとAPI通信を同時に制御・記録します。

1.2 実行・評価の3層構造（再定義）
種別	実行（入力生成）	評価（品質判定）
ルール	aiortcによるWAV再生（ミリ秒制御）	タイムスタンプ解析（応答速度、API成功率）
人間	ブラウザ/マイクを通じたリアルタイム対話	人間によるMOS評価（音質、心地よさ）
AI (LLM)	Audio2Audio AIによる自動対話生成	LLM (Text)による定性解析（文脈、配慮）

2. 実装計画 (Implementation Plan)
Phase 1: 基盤開発（WebRTC & Logging）
まず、時間軸の基準となる「ログ基盤」と「音声送受信」を実装します。

1.1 Unified Loggerの実装:

Pythonのasyncio.get_event_loop().time()を使用し、音声パケットの送受信とAPIフックの時刻を単一の時計で記録する仕組みを構築。

1.2 aiortc アダプターの作成:

システム側とWebRTC（ICE/SDP）で接続し、音声トラック（MediaStreamTrack）を双方向で扱うクラスを実装。

aiortc内でVAD（発話検知）を走らせ、システムの声が聞こえた瞬間を自動打刻する。

Phase 2: シナリオ実行エンジンの構築
Botiumライクなシナリオを読み込み、複雑な挙動を再現します。

2.1 シナリオパサー:

.convoファイルを読み込み、「ユーザー発話開始 → API待機 → ボット応答確認」というタスクキューを生成。

2.2 割り込み (Barge-in) コントローラー:

システムが発話中であっても、非同期でWebRTCトラックに音声を流し込むロジックを実装。

2.3 Mock API Server (FastAPI):

システムからのFunction Callingを受け取るエンドポイント。受信した瞬間にUnified Loggerへイベントを飛ばす。

Phase 3: AI評価・人間評価レイヤーの実装
データを「価値あるレポート」に変換します。

3.1 LLM-as-a-Judge プロンプト構築:

「時系列ログ（テキスト＋タイミング）」をLLMに渡し、UX的な違和感を抽出するプロンプトを設計。

3.2 Streamlit Review App:

テスト結果のJSONを読み込み、波形グラフと対話ログ、API実行履歴を並べて表示。人間が「いいね」や「ダメ」を付けられるUI。

3. ターゲットシステム側に必要な対応（IF定義）
汎用性を保つため、システム側には以下の「口」を用意してもらいます。

WebRTC Endpoint:

テストモード時に、物理マイクの代わりにIVEからのWebRTC接続を受け入れる。

API Base URLの動的変更:

テスト時のみ、外部APIの宛先をIVEのMock Server（例: http://localhost:8888）に変更する。

4. 技術スタック詳細
カテゴリ	推奨ライブラリ	理由
Core	Python 3.10+ / asyncio	非同期イベント制御の要
WebRTC	aiortc	パケットレベルで音声を操作可能
Analysis	silero-vad	高精度かつ軽量な発話区間検知
Browser	Playwright (Python)	Web系システムの自動操作とStream取得
AI/LLM	openai / whisper	評価と文字起こしのデファクト
Report UI	Streamlit	評価用ダッシュボードの高速開発


推奨フォルダ構造
Plaintext

ive-tester/
├── app/                    # 評価用UI (Streamlit等)
│   └── main.py             # 人間評価・レポート閲覧用画面
├── evaluator/              # 解析・スコアリング
│   ├── __init__.py
│   ├── rule_evaluator.py   # レイテンシ等の定量的解析
│   ├── llm_evaluator.py    # LLM-as-a-Judge (定性解析)
│   └── scoring.py          # 各指標の統合スコアリング
├── tester/                 # テスト実行コア
│   ├── __init__.py
│   ├── orchestrator.py     # 全体の進行・時系列ログ管理
│   ├── webrtc_client.py    # aiortcによる音声送受信
│   ├── browser_client.py   # Web系システム用 (Playwright)
│   └── api_mock.py         # Function Calling受け口 (FastAPI)
├── parser/                 # シナリオ・設定ファイルのパース
│   ├── __init__.py
│   ├── convo_parser.py     # Botium形式(.convo)のパース
│   └── config_loader.py    # botium.json / config.jsonの読み込み
├── scenarios/              # .convo ファイル置き場
└── reports/               # 実行結果（JSON/録音）の出力先
├── tests/                  # IVE自体のユニットテスト
├── config.json             # 全体設定
├── requirements.txt
└── run_ive.py              # メインエントリーポイント（CLI）
各ディレクトリの詳細役割
1. tester/ (実行の要)
orchestrator.py: このツールの心臓部です。asyncio のイベントループを回し、「音声パケット送信」「API受信」「VAD判定」のすべての時刻を一つのリストに記録します。

webrtc_client.py: aiortc を使い、MediaStreamTrack を実装します。Python系システムには直接、Web系システムにはブラウザを介して接続します。

2. evaluator/ (判断の脳)
実行が終わった後に生成された timeline.json を読み込んで処理します。

llm_evaluator.py: Whisperでの文字起こしと、LLMへの評価依頼（プロンプト送信）を担当します。

3. parser/ (Botium互換レイヤー)
Botiumライクな文法を独自拡張（#API_CALL や #BARGE_IN）するためのロジックをここに閉じ込めます。

4. app/ (人間との接点)
テストが終わった後、streamlit run app/main.py で立ち上げ、波形グラフを見ながら人間がMOS評価（5段階評価）を入力できるようにします。

最初に着手すべきファイル
まずは以下の3つを作成し、**「最小構成のWebRTC疎通」**を目指すのが良いと思います。

requirements.txt: 必要なライブラリの書き出し。

tester/webrtc_client.py: aiortc で音声を送受信する最小クラス。

run_ive.py: シナリオなしで、とりあえず特定の音声を投げて返信を待つだけの実行スクリプト。