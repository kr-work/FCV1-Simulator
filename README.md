# FCV1-Simulation
DC3のストーンシミュレーションをマルチスレッドで行うためのリポジトリです.

## Setup submodule
```
git submodule update --init --recursive
```

## Build Box2d
```
cd extern/box2d
mkdir build
cd build
cmake -DBOX2D_BUILD_DOCS=ON -DCMAKE_INSTALL_PREFIX="./" ..
cmake --build .
cmake --build . --target install
```

## Build simulator
CmakeLists.txtの4行目は,バージョン8以上必須
```
mkdir build
cd build
cmake ..
make
```

## Usage
### Set number of threads
srcディレクトリ内のconfig.jsonの中の
```
"thread_num": 8
```
の値で設定される.

### Notes
このプログラムを使用する際は
```
sim = Simulator()
```
でSimulatorクラスをインスタンス化してください.ここで,OpenMPによるスレッド作成も行うため,
試合開始前の準備時間中にインスタンス化してください.


### How to use Simulator
簡単な使用方法はtest.pyの通りである。
main関数の引数は
1. 先攻・後攻の順に格納したストーン座標
2. ストーンのショット数(0~15)
3. x方向の初速度ベクトル
4. y方向の初速度ベクトル
5. 回転方向(cw->1, ccw->-1)

戻り値は
1. main関数の引数にあるx・y方向の初速度ベクトル、回転方向のインデックス番号順に、シミュレーション後のストーン座標
2. ファイブロックルールが適用されたかどうかのbool値

## Dependencies
- numpy < 2.0
- OpenMP