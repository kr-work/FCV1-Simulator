#include "box2d/box2d.h"
#include <cmath>

struct Velocity{
    b2Vec2 vel;
};

struct NeuralNetworkInput {
    std::vector<float> coordinates; // xy座標の組データが16個（32個のfloat値）
    int shotCount; // ショット数
    int endCount; // エンド数
    int scoreDifference; // 得点差
    bool isFinished; // 終了条件
};

std::vector<NeuralNetworkInput> inputs;

struct StonePosition {
    float x;
    float y;
};

struct Position {
    StonePosition stone0;
    StonePosition stone1;
    StonePosition stone2;
    StonePosition stone3;
    StonePosition stone4;
    StonePosition stone5;
    StonePosition stone6;
    StonePosition stone7;
};

struct TeamName {
    Position my_team;
    Position opponent_team;
};

struct Stones {
    TeamName stones;
};

struct ShotVelocity {
    float v_x;
    float v_y;
    int angle;
};

struct StateData {
    Stones stones;
    int shot;
    bool hummer;
    int score_diff;
    int end;
    ShotVelocity velocity;
};

namespace digitalcurling3 {
struct Vector2 {
    float x;  ///< x座標
    float y;  ///< y座標

    /// \brief (0, 0)で初期化します
    constexpr Vector2() : x(0.f), y(0.f) {}

    /// \brief 指定された座標で初期化します
    ///
    /// \param[in] x x座標
    /// \param[in] y y座標
    constexpr Vector2(float x, float y) : x(x), y(y) {}

    /// \brief ベクトルの加算を行います
    ///
    /// \param[in] v ベクトル
    /// \returns このベクトル自身
    constexpr Vector2 & operator += (Vector2 v)
    {
        x += v.x;
        y += v.y;
        return *this;
    }

    /// \brief ベクトルの減算を行います
    ///
    /// \param[in] v ベクトル
    /// \returns このベクトル自身
    constexpr Vector2 & operator -= (Vector2 v)
    {
        x -= v.x;
        y -= v.y;
        return *this;
    }

    /// \brief ベクトルにスカラー値を乗算します
    ///
    /// \param[in] f スカラー値
    /// \returns このベクトル自身
    constexpr Vector2 & operator *= (float f)
    {
        x *= f;
        y *= f;
        return *this;
    }

    /// \brief ベクトルをスカラー値で除算します
    ///
    /// \param[in] f スカラー値
    /// \returns このベクトル自身
    constexpr Vector2 & operator /= (float f)
    {
        x /= f;
        y /= f;
        return *this;
    }

    /// \brief ベクトルの長さを得ます
    ///
    /// \returns \code std::hypot(x, y) \endcode
    float Length() const
    {
        return std::hypot(x, y);
    }
};

/// \brief ベクトルを反転します
///
/// \param[in] v ベクトル
/// \returns \a -v
constexpr Vector2 operator - (Vector2 v)
{
    return { -v.x, -v.y };
}

/// \brief ベクトル同士の加算を行います
///
/// \param[in] v1 ベクトル1
/// \param[in] v2 ベクトル2
/// \returns \a v1+v2
constexpr Vector2 operator + (Vector2 v1, Vector2 v2)
{
    return { v1.x + v2.x, v1.y + v2.y };
}

/// \brief ベクトル同士の減算を行います
///
/// \param[in] v1 ベクトル1
/// \param[in] v2 ベクトル2
/// \returns \a v1-v2
constexpr Vector2 operator - (Vector2 v1, Vector2 v2)
{
    return { v1.x - v2.x, v1.y - v2.y };
}

/// \brief ベクトルとスカラー値の乗算を行います
///
/// \param[in] f スカラー値
/// \param[in] v ベクトル
/// \returns \a f*v
constexpr Vector2 operator * (float f, Vector2 v)
{
    return { f * v.x, f * v.y };
}

/// \brief ベクトルとスカラー値の乗算を行います
///
/// \param[in] v ベクトル
/// \param[in] f スカラー値
/// \returns \a v*f
constexpr Vector2 operator * (Vector2 v, float f)
{
    return f * v;
}

/// \brief ベクトルとスカラー値の除算を行います
///
/// \param[in] v ベクトル
/// \param[in] f スカラー値
/// \returns \a v*(1/f)
constexpr Vector2 operator / (Vector2 v, float f)
{
    return { v.x / f, v.y / f };
}

inline b2Vec2 ToB2Vec2(Vector2 v)
{
    return { v.x, v.y };
}

inline Vector2 ToDC2Vector2(b2Vec2 v)
{
    return { v.x, v.y };
}
}

namespace digitalcurling3 {
    /// \brief 位置，角度を格納します．
struct Transform {
    Vector2 position;  ///< 位置
    float angle;       ///< 角度

    /// \brief 位置(0, 0)，角度0で初期化します
    constexpr Transform() : position(), angle(0.f) {}

    /// \brief 指定された値で初期化します
    ///
    /// \param[in] position 位置
    /// \param[in] angle 角度
    constexpr Transform(Vector2 position, float angle) : position(position), angle(angle) {}
};
}

namespace digitalcurling3 {
    /// \brief 位置を格納します．
struct StoneData{
    Vector2 position;
};
}

namespace digitalcurling3 {
/// \brief ストーンどうしの衝突の情報
struct Collision {
    /// \brief 衝突に関するストーンの情報
    struct Stone {
        std::uint8_t id; ///< ストーンのID
        Transform transform; ///< ストーンの位置と角度

        /// \brief デフォルトコンストラクタ
        Stone() : id(0), transform() {}

        /// \brief 与えられたデータで初期化します
        ///
        /// \param[in] id ストーンのID
        /// \param[in] transform ストーンの位置と角度
        Stone(std::uint8_t id, Transform const& transform) : id(id), transform(transform) {}
    };
    Stone a; ///< 衝突したストーン
    Stone b; ///< 衝突したストーン
    float normal_impulse; ///< 法線方向の撃力
    float tangent_impulse; ///< 接線方向の撃力
    /// \brief 全パラメータを 0 で初期化します
    Collision()
        : a()
        , b()
        , normal_impulse(0.f)
        , tangent_impulse(0.f) {}
    /// \brief 与えられたパラメータで初期化します
    ///
    /// \param[in] a_id ストーンAのID
    /// \param[in] b_id ストーンBのID
    /// \param[in] a_transform ストーンAの位置
    /// \param[in] b_transform ストーンBの位置
    /// \param[in] normal_impulse 法線方向の撃力
    /// \param[in] tangent_impulse 接線方向の撃力
    Collision(std::uint8_t a_id, std::uint8_t b_id, Transform const& a_transform, Transform const& b_transform, float normal_impulse, float tangent_impulse)
        : a(a_id, a_transform)
        , b(b_id, b_transform)
        , normal_impulse(normal_impulse)
        , tangent_impulse(tangent_impulse) {}

    /// \brief ストーンどうしが接した座標を得る。
    ///
    /// \returns ストーンどうしが接した座標
    Vector2 GetContactPoint() const
    {
        return (a.transform.position + b.transform.position) * 0.5f;
    }
};
}

class StoneData {
public:
    b2Body* body;
    std::vector<digitalcurling3::Collision> collisions;
};


    
class SimulatorFCV1 {
public:
    class ContactListener : public b2ContactListener {
    public:
        ContactListener(SimulatorFCV1 * instance) : instance_(instance) {}
        virtual void PostSolve(b2Contact* contact, const b2ContactImpulse* impulse) override;
    private:
        SimulatorFCV1 * const instance_;
    };
private:
    ContactListener contact_listener_;
};

