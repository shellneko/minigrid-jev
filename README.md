# minigrid-jev

[MiniGrid](https://minigrid.farama.org/) のグリッドワールドタスクを、TypeSafe の System One モデル **Jev** に解かせるベンチマークです。

Jev はテキスト生成モデルではなく「型付きの判断」を返すモデルです。本ベンチマークでは各ステップごとに、環境状態をJSON化した `state` と、取りうる行動を選択肢とした `Choice` 質問を `client.system_one()` に送り、返ってきた行動を実行します。

## 必要条件

- [uv](https://docs.astral.sh/uv/)
- 環境変数 `TYPESAFE_API_KEY` に TypeSafe の API キーが設定されていること

## セットアップ

```bash
uv sync
```

依存: `typesafe-sdk`, `minigrid` (gymnasium, pygame-ce などは自動で入ります)

## 使い方

```bash
# タスク一覧を表示
uv run python benchmark.py --list

# 単一タスクを実行
uv run python benchmark.py --tasks MiniGrid-DoorKey-5x5-v0 --episodes 1

# 難易度で絞って実行
uv run python benchmark.py --difficulty easy --episodes 3

# 全タスクを並列実行して結果をJSONに保存
uv run python benchmark.py --episodes 5 --jobs 8 --out results.json

# プレイを観察(pygameウィンドウ + 各ステップの判断を表示)
uv run python benchmark.py --tasks MiniGrid-Empty-5x5-v0 --episodes 1 --jobs 1 --verbose --render
```

### オプション

| オプション | 説明 | デフォルト |
|---|---|---|
| `--tasks` | 環境IDをカンマ区切りで指定 | 全タスク |
| `--difficulty` | `easy` / `medium` / `hard` で絞り込み | 指定なし |
| `--episodes` | タスクごとのエピソード数 | 3 |
| `--seed` | ベースシード(エピソード i は seed+i) | 0 |
| `--jobs` | 並列エピソード数 | 4 |
| `--max-steps` | 1エピソードのステップ上限 | 環境の上限 |
| `--model` | モデル名(`jev-latest` など) | `TYPESAFE_DEFAULT_MODEL` または SDK デフォルト |
| `--render` | pygame ウィンドウを表示(`--jobs 1` 推奨) | off |
| `--verbose` | 各ステップの行動・confidence・結果を表示 | off |
| `--out` | エピソード結果をJSONで保存するパス | なし |
| `--list` | タスク一覧を表示して終了 | - |

## タスク一覧

18タスクを3つの難易度に分類しています(`minigrid_jev/tasks.py` で定義)。

| 難易度 | 環境 | 内容 |
|---|---|---|
| easy | MiniGrid-Empty-5x5-v0 | 空の部屋でゴールへ |
| easy | MiniGrid-Empty-Random-5x5-v0 | ランダム初期位置の空の部屋 |
| easy | MiniGrid-DistShift1-v0 | lavaを避けてゴールへ |
| medium | MiniGrid-FourRooms-v0 | 4部屋をまたいでゴールへ |
| medium | MiniGrid-DoorKey-5x5-v0 / 6x6 | 鍵を拾いドアを開けてゴールへ |
| medium | MiniGrid-SimpleCrossingS9N1-v0 | 壁の隙間を抜けてゴールへ |
| medium | MiniGrid-RedBlueDoors-6x6-v0 | 赤→青の順にドアを開ける |
| medium | MiniGrid-GoToDoor-5x5-v0 | 指定色のドアに隣接して `done` を実行 |
| medium | MiniGrid-Fetch-5x5-N2-v0 | 指示された物体をpickup(誤pickupは即失敗) |
| hard | MiniGrid-DoorKey-8x8-v0 | より大きな部屋での鍵&ドア |
| hard | MiniGrid-Unlock-v0 | 別部屋の鍵でドアを解錠 |
| hard | MiniGrid-LavaGapS7-v0 | lavaを避けてゴールへ |
| hard | MiniGrid-Dynamic-Obstacles-6x6-v0 | 動く障害物を避ける |
| hard | MiniGrid-KeyCorridorS3R1-v0 | 鍵を探し部屋を巡ってターゲットをpickup |
| hard | MiniGrid-ObstructedMaze-1Dl-v0 | 鍵とブロックされたドアのある迷路 |
| hard | MiniGrid-MultiRoom-N2-S4-v0 | 連続する部屋を横断 |
| hard | MiniGrid-PutNear-6x6-N2-v0 | 指定物体をターゲットの隣にdrop |

## 仕組み

### state(毎ステップ送信)

```json
{
  "mission": "use the key to open the door and then get to the goal",
  "rules": "Walls and closed or locked doors cannot be walked through. ...",
  "agent": {"position": [1, 3], "facing": "south", "carrying": null},
  "cell_directly_ahead": "yellow key",
  "view": {"note": "...", "cells": [["unseen", ...], ...]},
  "steps_used": 0, "step_limit": 250,
  "positions_visited": [[1, 3]],
  "recent_actions": ["move_forward -> moved forward"]
}
```

- `view.cells` はエージェント中心の7×7視界(エージェントは下端中央・上向き)。各セルは `"locked yellow door"` のようなテキスト表現、視野外は `"unseen"`
- `rules` には全環境共通のルール(lavaは即死など)と環境固有ルール(`done` 必須、誤pickupは失敗など)を注入

### 質問(Choice)

「missionと状況から最適な次の一手を選べ」という `Choice` 質問を1つ発行。選択肢は状況に応じて動的に絞られます:

- 常時: `turn_left` / `turn_right` / `move_forward`
- 正面が拾える物体&素手: `pickup`
- 持ち物あり&正面が空床: `drop`
- 正面がドア/箱: `toggle`
- `done` が必要な環境のみ: `done`

### メトリクス

成功率、平均ステップ数、平均報酬、平均confidence、トークン使用量を環境別・難易度別に集計して表示します。`--out` でエピソードごとの詳細をJSON保存できます。

## ファイル構成

```
benchmark.py            # CLIエントリポイント
minigrid_jev/
  tasks.py              # タスク登録(難易度・環境固有ルール)
  encoding.py           # 観測 -> JSON state 変換
  agent.py              # JevAgent(Choice質問で行動決定)
  runner.py             # エピソードループ・集計・レポート
```

## 注意

- 各ステップ1回のAPI呼び出しが発生します。全タスク x 複数エピソードでは数千回の呼び出しになるため、まず `--tasks` や `--difficulty` で絞って試すことをおすすめします
- `--render` で `--jobs > 1` を指定するとエピソードごとにウィンドウが開きます
