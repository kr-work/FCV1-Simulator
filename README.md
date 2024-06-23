# FCV1-Simulation
DC3のストーンシミュレーションをマルチスレッドで行うためのリポジトリです.

## サブモジュールのセットアップ
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
Python側でこの実行ファイルを呼び出す際は,numpyもインストールしてください.
numpyを使用して,各ストーンの座標や投球初速度をシミュレーション(C++)側に与えています.
なお,シミュレーションに与えるストーンの座標は,先攻のストーン座標(x, y)を8組,高校のストーン座標(x, y)を8組
計16組を1つのarrayにしてください.
```
position: list = [-2.0, 34.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
np_position = np.array(position)
```

他のx・y方向の初速度ベクトル,回転方向についても同様に,それぞれ1つのarrayにしてください.
```
x_velocities: list = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
y_velocities: list = [2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4]
angular_velocities: list = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
np_x_velocities = np.array(x_velocities)
np_y_velocities = np.array(y_velocities)
np_angular_velocities = np.array(angular_velocities)
```
なお,cwの場合は1,ccwの場合は-1にしてください.