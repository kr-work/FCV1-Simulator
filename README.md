# FCV1-Simulation


## サブモジュールのセットアップ
```
git submodule update --init --recursive
```

## box2dのビルド
```
cd extern/box2d
mkdir build
cd build
cmake -DBOX2D_BUILD_DOCS=ON -DCMAKE_INSTALL_PREFIX="./" ..
cmake --build .
cmake --build . --target install
```

## 使い方
CmakeLists.txtの4行目は、バージョン8以上必須
```
mkdir build
cd build
cmake ..
make
```
