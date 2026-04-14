# FCV1-Simulator

FCV1-Simulator は、pybind11 と Box2D で構築された Python モジュール (`simulator`) として利用できる、カーリングストーン物理シミュレータです。

この README は、GitHub Releases の成果物を利用して Python からシミュレーションを実行したいユーザー向けに記載しています。

## 概要

- 入力: 現在のストーン配置、投球状況、投球速度、回転方向、ルールモード
- エンジン: Box2D ベースの剛体シミュレーション + カーリング向け運動モデル
- 出力:
  - 最終ストーン位置 (NumPy 配列)
  - シミュレーション中にサンプリングした軌跡 (Python リスト)

## 対応環境

- OS:
  - Linux (`.so` 成果物)
  - Windows (`.pyd` 成果物)
- Python: 3.9 から 3.12 (CI/Release の対象)
- Python パッケージ:
  - `numpy < 2.0`

Python 依存をインストール:

```bash
pip install "numpy<2.0"
```

## クイックスタート (Release 成果物を使う)

1. Releases から OS と Python バージョンに合う成果物をダウンロードします。
2. プロジェクトに `build` ディレクトリを作成します。
3. 成果物を `build/` に置き、以下の名前に変更します。
   - Linux: `simulator.so`
   - Windows: `simulator.pyd`
4. Python から `StoneSimulator` を import して呼び出します。

例:

```python
import numpy as np
from build.simulator import StoneSimulator

sim = StoneSimulator()

# 12 stones (mixed doubles format): shape (12, 2)
stone_positions = np.array([
    [0.0, 34.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
    [0.0, 0.0],  [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
], dtype=np.float64)

result, trajectory = sim.simulator(
    stone_positions=stone_positions,
    shot=4,
    x_velocity=-0.09,
    y_velocity=2.32,
    angular_sign=1,
    team_id=1,
    shot_per_team=2,
    applied_rule=2,
)

print(result.shape)   # 12 stone 入力なら (2, 6, 2)、16 stone 入力なら (2, 8, 2)
print(len(trajectory))
```

## ソースからビルドする場合

Release 成果物を使わずローカルビルドする場合は、以下を実行します。

### 1) 依存関係の準備

```bash
pip install pybind11[global]
```

### 2) サブモジュール準備 (CI と同じ固定コミット)

```bash
git submodule update --init --recursive
cd extern/box2d
git checkout 9ebbbcd960ad424e03e5de6e66a40764c16f51bc
cd ../json
git checkout 11a835df85677002a8aadc5b4e945684c5b7f68b
cd ../..
```

### 3) Box2D ビルド

```bash
cd extern/box2d
mkdir -p build
cd build
cmake -DBOX2D_BUILD_DOCS=OFF -DBOX2D_BUILD_UNIT_TESTS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX="./" ..
cmake --build . --config Release
cmake --build . --target install --config Release
cd ../../..
```

### 4) nlohmann/json ビルド

```bash
cd extern/json
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX="./" ..
cmake --build .
cmake --build . --target install
cd ../../..
```

### 5) simulator モジュールをビルド

```bash
cd src
mkdir -p build
cd build
cmake ..
cmake --build . --config Release
```

生成物:

- Linux: `src/build/simulator.so`
- Windows: `src/build/simulator.pyd`

## Python API リファレンス

### クラス

- `StoneSimulator`

### メソッド

```python
simulator(
    stone_positions,
    shot,
    x_velocity,
    y_velocity,
    angular_sign,
    team_id,
    shot_per_team,
    applied_rule,
) -> (result, trajectory)
```

### 引数

1. `stone_positions` (`numpy.ndarray`)
   - 受け付ける形状:
     - `1D`: 長さ `32` (16 stones) または `24` (12 stones)
     - `2D`: 形状 `(16, 2)` または `(12, 2)`
   - チーム順序:
     - 16 stone モード: team0 `[0..7]`, team1 `[0..7]`
     - 12 stone モード: team0 `[0..5]`, team1 `[0..5]`

2. `shot` (`int`)
   - エンド内の総投球数。

3. `x_velocity` (`float`)
   - 投球ストーンの初期 x 速度。

4. `y_velocity` (`float`)
   - 投球ストーンの初期 y 速度。

5. `angular_sign` (`int`)
   - 回転方向:
     - `1`: 時計回り
     - `-1`: 反時計回り

6. `team_id` (`int`)
   - 投球チーム ID (`0` または `1`)。

7. `shot_per_team` (`int`)
   - チーム内の投球インデックス。

8. `applied_rule` (`int`)
   - ルールモード:
     - `0`: five rock rule
     - `1`: no tick rule
     - `2`: modified FGZ rule

### 戻り値

1. `result` (`numpy.ndarray`, 3次元)
   - 形状:
     - 16 stone 入力時: `(2, 8, 2)`
     - 12 stone 入力時: `(2, 6, 2)`
   - 最終軸は `(x, y)`。

2. `trajectory` (`list`)
   - 時系列ステップのリスト。
   - 各ステップは `(stone_id, x, y)` のタプルのリスト。
   - 100 フレームごとにサンプリングされます。
   - シミュレーション刻みは `0.001` 秒なので、約 `0.1` 秒間隔の軌跡です。

## ルールモード

- `applied_rule = 0` (five rock rule)
  - 早い投球 (`shot < 5`) で free-guard-zone 保護ロジックを適用します。

- `applied_rule = 1` (no tick rule)
  - 早い投球 (`shot < 5`) でセンターラインの no-tick 判定を適用します。

- `applied_rule = 2` (modified FGZ)
  - 最初の3投 (`shot < 3`) で既存の in-play ストーンを保護します。
  - このモードでは、置き石を考慮した内部投球インデックス処理が行われます。

## 入出力に関する注意

- 内部的には常に 16 スロットで計算し、12 stone 入力は内部マッピングされます。
- プレー外ストーンは `(0.0, 0.0)` で表現されます。
- プレーエリア制約は後処理でも適用されます。

## 実装されている主な定数

- ストーン半径: `0.145 m`
- ハウス半径: `1.829 m`
- ティーライン: `38.405 m`
- プレーエリア x 範囲: `[-2.375, 2.375]`
- 判定で使う y 範囲: およそ `[30.0, 40.234]`

## トラブルシュート

1. `ModuleNotFoundError: No module named 'build.simulator'`
   - 成果物が `build/` にあり、ファイル名が Linux では `simulator.so`、Windows では `simulator.pyd` になっているか確認してください。

2. NumPy 関連の import/runtime エラー
   - `numpy < 2.0` を使用してください。

3. ビルド時に Box2D や nlohmann_json が見つからない
   - `extern/box2d/build` と `extern/json/build` で依存ライブラリのビルドが完了しているか確認してください。

4. ローカルビルドと Release の挙動差がある
   - CI と同じ固定サブモジュールコミットを使ってください。

## 関連ファイル

- 使用例スクリプト: `src/test.py`
- 入力データ例: `src/data.json`
- 基本設定例: `src/config.json`
