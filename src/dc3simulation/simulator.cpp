#include <stdio.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <box2d/box2d.h>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <pybind11/stl.h>
#include "simulator.hpp"
#include <omp.h> // OpenMP
#include <set>

// シミュレーションについては,自分のストーン情報の次に相手のストーン情報という順番にstonesに格納する
// それと、ショット数と先攻後攻を示せば、次にどのストーン情報を用いたシミュレーションをするのかわかるはず
// 1601回のシミュレーションを行うために、作成したシミュレータ環境を複製して高速化したい
// 過去の設定を変えるのが面倒だから、main関数の引数のひとつで、1がきたらcw、-1がきたらccw

constexpr float kStoneRadius = 0.145f;
static constexpr ::uint8_t kStoneMax = 16;
constexpr float kPi = 3.14159265359f;
constexpr float cw = -kPi / 2.f;
constexpr float ccw = kPi / 2.f;
constexpr float y_upper_limit = 40.234f;
constexpr float x_upper_limit = 2.375f;
constexpr float x_lower_limit = -2.375f;
namespace py = pybind11;
std::vector<float> x_velocities;
std::vector<float> state_values;

// 返り値1つめ: 正規化されたベクトル
// 返り値2つめ: もとのベクトルの長さ
inline std::pair<b2Vec2, float> Normalize(b2Vec2 const &v)
{
    b2Vec2 normalized = v;
    float length = normalized.Normalize();
    return {normalized, length};
}

// 前進加速度 [m/s^2]
inline float LongitudinalAcceleration(float speed)
{
    constexpr float kGravity = 9.80665f;
    return -(0.00200985f / (speed + 0.06385782f) + 0.00626286f) * kGravity;
}

// ヨーレート 単位: [rad/s]
inline float YawRate(float speed, float angularVelocity)
{
    if (std::abs(angularVelocity) <= std::numeric_limits<float>::epsilon())
    {
        return 0.f;
    }
    return (angularVelocity > 0.f ? 1.0f : -1.0f) * 0.00820f * std::pow(speed, -0.8f);
}

inline float AngularAcceleration(float linearSpeed)
{
    float clampedSpeed = std::max(linearSpeed, 0.001f);
    return -0.025f / clampedSpeed;
}

class Simulator
{
    std::vector<digitalcurling3::StoneData> const &stones;
    bool hummer;
    int shot;
    Velocity const &velocity;
    float angular_velocity;

public:
    std::vector<int> is_awake;
    class ContactListener : public b2ContactListener
    {
    public:
        ContactListener(Simulator *instance) : instance_(instance) {}
        void PostSolve(b2Contact *contact, const b2ContactImpulse *impulse) override
        {
            auto a_body = contact->GetFixtureA()->GetBody();
            auto b_body = contact->GetFixtureB()->GetBody();

            digitalcurling3::Collision collision;
            collision.a.id = static_cast<int>(a_body->GetUserData().pointer);
            collision.b.id = static_cast<int>(b_body->GetUserData().pointer);

            instance_->is_awake.push_back(collision.a.id);
            instance_->is_awake.push_back(collision.b.id);

            b2WorldManifold world_manifold;
            contact->GetWorldManifold(&world_manifold);

            collision.normal_impulse = impulse->normalImpulses[0];
            collision.tangent_impulse = impulse->tangentImpulses[0];

            // instance_->storage_.collisions.emplace_back(std::move(collision));
        }

    private:
        Simulator *const instance_;
    };
    b2World world;
    b2BodyDef stone_body_def;
    ContactListener contact_listener_;
    std::array<b2Body *, static_cast<std::size_t>(kStoneMax)> stone_bodies;
    Simulator(std::vector<digitalcurling3::StoneData> const &stones, bool hummer, int shot, Velocity const &velocity, float angular_velocity) : stones(stones), hummer(hummer), shot(shot), velocity(velocity), angular_velocity(angular_velocity), world(b2Vec2(0, 0)), contact_listener_(this)
    {

        stone_body_def.type = b2_dynamicBody;
        stone_body_def.awake = false;
        stone_body_def.bullet = true;
        stone_body_def.enabled = false;

        b2CircleShape stone_shape;
        stone_shape.m_radius = kStoneRadius;

        b2FixtureDef stone_fixture_def;
        stone_fixture_def.shape = &stone_shape;
        stone_fixture_def.friction = 0.2f;                                        // 適当というかデフォルト値
        stone_fixture_def.restitution = 1.0;                                      // 完全弾性衝突(完全弾性衝突の根拠は無いし多分違う)
        stone_fixture_def.restitutionThreshold = 0.f;                             // 反発閾値。この値より大きい速度(m/s)で衝突すると反発が適用される。
        stone_fixture_def.density = 0.5f / (b2_pi * kStoneRadius * kStoneRadius); // kg/m^2

        for (size_t i = 0; i < kStoneMax; ++i)
        {
            stone_body_def.userData.pointer = static_cast<uintptr_t>(i); // ストーンのidが0~15まで割り振られる
            stone_bodies[i] = world.CreateBody(&stone_body_def);
            stone_bodies[i]->CreateFixture(&stone_fixture_def);
        }
        world.SetContactListener(&contact_listener_);
    }

    void Step()
    {
        // simulate
        while (!is_awake.empty())
        {
            for (int index : is_awake)
            {
                b2Vec2 const stone_velocity = stone_bodies[index]->GetLinearVelocity(); // copy
                auto const [normalized_stone_velocity, stone_speed] = Normalize(stone_velocity);
                float const angular_velocity = stone_bodies[index]->GetAngularVelocity();

                // 速度を計算
                // ストーンが停止してる場合は無視
                if (stone_speed > std::numeric_limits<float>::epsilon())
                {
                    // ストーンの速度を計算
                    float const new_stone_speed = stone_speed + LongitudinalAcceleration(stone_speed) * 0.001;
                    if (new_stone_speed <= 0.f)
                    {
                        stone_bodies[index]->SetLinearVelocity(b2Vec2_zero);
                        is_awake.erase(std::remove(is_awake.begin(), is_awake.end(), index), is_awake.end());
                    }
                    else
                    {
                        // if (stone_bodies[index]->GetPosition().y > y_upper_limit || stone_bodies[index]->GetPosition().x > x_upper_limit || stone_bodies[index]->GetPosition().x < x_lower_limit)
                        // {
                        //     stone_bodies[index]->SetLinearVelocity(b2Vec2_zero);
                        //     is_awake.erase(std::remove(is_awake.begin(), is_awake.end(), index), is_awake.end());
                        // }
                        float const yaw = YawRate(stone_speed, angular_velocity) * 0.001;
                        float const longitudinal_velocity = new_stone_speed * std::cos(yaw);
                        float const transverse_velocity = new_stone_speed * std::sin(yaw);
                        b2Vec2 const &e_longitudinal = normalized_stone_velocity;
                        b2Vec2 const e_transverse = e_longitudinal.Skew();
                        b2Vec2 const new_stone_velocity = longitudinal_velocity * e_longitudinal + transverse_velocity * e_transverse;
                        stone_bodies[index]->SetLinearVelocity(new_stone_velocity);
                    }
                }

                // 角速度を計算
                if (std::abs(angular_velocity) > std::numeric_limits<float>::epsilon())
                {
                    float const angular_accel = AngularAcceleration(stone_speed) * 0.001;
                    float new_angular_velocity = 0.f;
                    if (std::abs(angular_velocity) <= std::abs(angular_accel))
                    {
                        new_angular_velocity = 0.f;
                    }
                    else
                    {
                        new_angular_velocity = angular_velocity + angular_accel * angular_velocity / std::abs(angular_velocity);
                    }
                    stone_bodies[index]->SetAngularVelocity(new_angular_velocity);
                }
            }

            // storage.collisions.clear();

            world.Step(
                0.001,
                8,  // velocityIterations (公式マニュアルでの推奨値は 8)
                3); // positionIterations (公式マニュアルでの推奨値は 3)
        }
    }

    void SetStones()
    {
        // update bodies
        for (size_t i = 0; i < shot / 2 + 1; ++i)
        {
            auto &stone = stones[i];
            auto position = stone.position;
            if (position.x == 0.f && position.y == 0.f)
            {
                stone_bodies[i]->SetEnabled(false);
            }
            else
            {
                stone_bodies[i]->SetEnabled(true);
                stone_bodies[i]->SetAwake(true);
                stone_bodies[i]->SetTransform(b2Vec2(position.x, position.y), 0.f);
            }
        }

        for (size_t i = 8; i < shot / 2 + 9; ++i)
        {
            auto &stone = stones[i];
            auto position = stone.position;
            if (position.x == 0.f && position.y == 0.f)
            {
                stone_bodies[i]->SetEnabled(false);
            }
            else
            {
                stone_bodies[i]->SetEnabled(true);
                stone_bodies[i]->SetAwake(true);
                stone_bodies[i]->SetTransform(b2Vec2(position.x, position.y), 0.f);
            }
        }

        // 自チームの全ストーン情報の後に相手チームの全ストーン情報が格納されているため、先攻後攻とshot数から、現在投球するストーンのインデックス番号を計算して、そこに速度と角速度をセットする
        if (shot % 2 == 0)
        {
            int index = shot / 2 + hummer * 8;
            stone_bodies[index]->SetLinearVelocity(b2Vec2(velocity.vel.x, velocity.vel.y));
            stone_bodies[index]->SetAngularVelocity(angular_velocity);
            stone_bodies[index]->SetEnabled(true);
            stone_bodies[index]->SetAwake(true);
            stone_bodies[index]->SetTransform(b2Vec2(stones[index].position.x, stones[index].position.y), 0.f);
            is_awake.push_back(index);
        }
        else
        {
            int index = shot / 2 + !hummer * 8;
            stone_bodies[index]->SetLinearVelocity(b2Vec2(velocity.vel.x, velocity.vel.y));
            stone_bodies[index]->SetAngularVelocity(angular_velocity);
            stone_bodies[index]->SetEnabled(true);
            stone_bodies[index]->SetAwake(true);
            stone_bodies[index]->SetTransform(b2Vec2(stones[index].position.x, stones[index].position.y), 0.f);
            is_awake.push_back(index);
        }
    }

    std::vector<digitalcurling3::StoneData> GetStones()
    {
        std::vector<digitalcurling3::StoneData> stones;
        for (size_t i = 0; i < kStoneMax; ++i)
        {
            auto body = stone_bodies[i];
            if (body->GetPosition().y > y_upper_limit || body->GetPosition().x > x_upper_limit || body->GetPosition().x < x_lower_limit)
            {
                body->SetTransform(b2Vec2(0.f, 0.f), 0.f);
            }
            // std::cout << body->GetPosition().x << std::endl;
        }
        for (size_t i = 0; i < kStoneMax; ++i)
        {
            auto body = stone_bodies[i];
            stones.push_back({digitalcurling3::Vector2(body->GetPosition().x, body->GetPosition().y)});
            // std::cout << body->GetPosition().x << std::endl;
        }
        for (size_t i = 0; i < kStoneMax; ++i)
        {
            auto body = stone_bodies[i];
            std::cout << i << std::endl;
            std::cout << body->GetPosition().x << std::endl;
            std::cout << body->GetPosition().y << std::endl;
        }
        return stones;
    }

    // MC法で何度もシミュレーションを行うために、前回のシミュレーションで移動したストーンのみ初期値に戻す
    // void SetStones_awake(std::vector<digitalcurling3::StoneData> const& stones, int shot, Velocity const& velocity, float angular_velocity){
    //     // update bodies
    //     for (size_t i = 0; i < shot + 1; ++i) {
    //         auto & stone = stones[i];
    //         auto position = stone.position;
    //         if (position.x == 0.f && position.y == 0.f){
    //             stone_bodies[i]->SetEnabled(false);
    //         } else {
    //             stone_bodies[i]->SetEnabled(true);
    //             stone_bodies[i]->SetAwake(true);
    //             stone_bodies[i]->SetTransform(b2Vec2(position.x, position.y), 0.f);
    //         }
    //     }
    //     stone_bodies[shot]->SetLinearVelocity(b2Vec2(velocity.vel.x, velocity.vel.y));
    //     stone_bodies[shot]->SetAngularVelocity(angular_velocity);
    // }
};

// メンバ変数を持つクラスを定義
class MSSimulator
{
public:
    std::vector<digitalcurling3::StoneData> storage;
    Velocity velocity;
    int shot;
    bool hummer;
    int score;
    int angle;
    float angular_velocity;
    MSSimulator() : storage(), velocity(), shot(), hummer(), score(), angular_velocity() {}

    void main(std::string stones_str, std::string shot_str, std::string hummer_str, std::string score_diff_str, std::string velocity_str)
    {
        std::vector<digitalcurling3::StoneData> simulated_stones;
        nlohmann::json stones_json = nlohmann::json::parse(stones_str);
        nlohmann::json shot_json = nlohmann::json::parse(shot_str);
        nlohmann::json hummer_json = nlohmann::json::parse(hummer_str);
        nlohmann::json score_diff_json = nlohmann::json::parse(score_diff_str);
        nlohmann::json velocity_json = nlohmann::json::parse(velocity_str);
        for (const auto &stone : stones_json)
        {
            digitalcurling3::StoneData data;
            // std::cout << "Stone position x: " << stone["position"]["x"] << std::endl;
            data.position = digitalcurling3::Vector2(stone["position"]["x"], stone["position"]["y"]);
            // std::cout << "stone position" << std::endl;
            // std::cout << stone["position"]["x"] << std::endl;
            // std::cout << stone["position"]["y"] << std::endl;
            storage.push_back(data);
        }
        shot = shot_json["shot"];
        hummer = hummer_json["hummer"];
        score = score_diff_json["score"];
        velocity.vel = b2Vec2(velocity_json["x"], velocity_json["y"]);
        angle = velocity_json["angle"];
        angular_velocity = angle * cw;

        state_values.reserve(1602); // 1602回の
        x_velocities.reserve(801);
        for (float i = -0.4; i <= 0.4; i += 0.01)
        {
            x_velocities.push_back(i);
        }

        // int a = omp_get_max_threads();
        // std::cout << a << " concurrent threads are supported.\n";
        Simulator *simulator = new Simulator(storage, hummer, shot, velocity, angular_velocity);
        simulator->SetStones();
        simulator->Step();
        simulator->GetStones();
    }
};

// main関数

PYBIND11_MODULE(simulator, m)
{
    py::class_<MSSimulator>(m, "Simulator")
        .def(py::init<>())
        .def("main", &MSSimulator::main);
}