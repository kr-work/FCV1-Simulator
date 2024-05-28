#include <stdio.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <box2d/box2d.h>
#include <cmath>
#include <limits>
#include <pybind11/stl.h>
#include "fcv1_simulator.hpp"
#include <omp.h> // OpenMP
#include <set>
#include <thread>
#include <chrono>
#include <mutex>

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
constexpr float stone_x_upper_limit = x_upper_limit - 2 * kStoneRadius;
constexpr float stone_x_lower_limit = x_lower_limit + 2 * kStoneRadius;
constexpr float tee_line = 38.405f;
constexpr float house_radius = 1.829f;
constexpr float EPSILON = std::numeric_limits<float>::epsilon();
std::mutex cout_mutex;
// constexpr float stone_radius = 0.145f;
namespace py = pybind11;
std::vector<std::pair<float, float>> x_velocities;
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
    if (std::abs(angularVelocity) <= EPSILON)
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

inline py::tuple Vector2Totuple(const digitalcurling3::Vector2 &vec)
{
    return py::make_tuple(vec.x, vec.y);
}

class Simulator
{
    std::vector<digitalcurling3::StoneData> const &stones;
    bool my_team_flag;
    int shot;
    float angular_velocity;

public:
    std::vector<int> is_awake; // どのストーンが今動いているか
    std::vector<int> moved;    // どのストーンが動いたか
    std::vector<int> on_center_line;
    std::vector<int> in_free_guard_zone;
    bool free_guard_zone;
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

            AddUniqueID(instance_->is_awake, collision.a.id);
            AddUniqueID(instance_->is_awake, collision.b.id);

            AddUniqueID(instance_->moved, collision.a.id);
            AddUniqueID(instance_->moved, collision.b.id);

            b2WorldManifold world_manifold;
            contact->GetWorldManifold(&world_manifold);

            collision.normal_impulse = impulse->normalImpulses[0];
            collision.tangent_impulse = impulse->tangentImpulses[0];

            // instance_->storage_.collisions.emplace_back(std::move(collision));
        }

    private:
        Simulator *const instance_;

        void AddUniqueID(std::vector<int>& list, int id)
        {
            if (std::find(list.begin(), list.end(), id) == list.end())
            {
                list.push_back(id);
            }
        }
    };
    b2World world;
    b2BodyDef stone_body_def;
    ContactListener contact_listener_;
    std::array<b2Body *, static_cast<std::size_t>(kStoneMax)> stone_bodies;
    Simulator(std::vector<digitalcurling3::StoneData> const &stones) : stones(stones), world(b2Vec2(0, 0)), contact_listener_(this)
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
            stone_body_def.userData.pointer = static_cast<uintptr_t>(i);
            stone_bodies[i] = world.CreateBody(&stone_body_def);
            stone_bodies[i]->CreateFixture(&stone_fixture_def);
        }
        world.SetContactListener(&contact_listener_);
    }

    void IsFreeGuardZone()
    {
        for (size_t i = 0; i < kStoneMax; ++i)
        {
            auto body = stone_bodies[i];
            float dx = body->GetPosition().x;
            float dy = body->GetPosition().y - tee_line;
            float distance_squared = dx * dx + dy * dy;
            if (dy < 0 && distance_squared > house_radius * house_radius)
            {
                in_free_guard_zone.push_back(i);
            }
        }
    }

    void ChangeShotAndFlag(int shot, bool my_team_flag)
    {
        this->shot = shot;
        this->my_team_flag = my_team_flag;
    }

    bool IsInPlayarea()
    {
        for (int i : in_free_guard_zone)
        {
            auto body = stone_bodies[i];
            if (body->GetPosition().y > y_upper_limit || body->GetPosition().x > stone_x_upper_limit || body->GetPosition().x < stone_x_lower_limit)
            {
                for (int index : moved)
                {
                    auto stone = stones[index];
                    stone_bodies[index]->SetTransform(b2Vec2(stone.position.x, stone.position.y), 0.f);
                }
                return true;
            }
        }
        return false;
    }
    // ノーティックルール対応用関数
    // void OnCenterLine()
    // {
    //     for (size_t i = 0; i < kStoneMax; ++i)
    //     {
    //         auto stone_body = stone_bodies[i];
    //         if (stone_body.IsEnabled() && stone_radius - std::abs(stone_body->GetPosition().x) < 0.0)
    //         {
    //             on_center_line.push_back(i);
    //         }
    //     }
    // }

    // ノーティックルール対応用関数
    // void NoTickRule()
    // {
    //     for (int i : on_center_line)
    //     {
    //         auto stone_body = stone_bodies[i];
    //         if (stone_radius - std::abs(stone_body->GetPosition().x) > 0.0)
    //         {
    //             for (int index : moved)
    //             {
    //                 auto stone = stones[index];
    //                 stone_bodies[index]->SetTransform(b2Vec2(stone.position.x, stone.position.y), 0.f);
    //             }
    //             break;
    //         }
    //     }
    // }

    void Step(float seconds_per_frame)
    {
        // simulate
        while (!is_awake.empty())
        {
            for (auto &index : is_awake)
            {
                b2Vec2 const stone_velocity = stone_bodies[index]->GetLinearVelocity(); // copy
                auto const [normalized_stone_velocity, stone_speed] = Normalize(stone_velocity);
                float const angular_velocity = stone_bodies[index]->GetAngularVelocity();

                // 速度を計算
                // ストーンが停止してる場合は無視
                if (stone_speed > EPSILON)
                {
                    // ストーンの速度を計算
                    float const new_stone_speed = stone_speed + LongitudinalAcceleration(stone_speed) * seconds_per_frame;
                    if (new_stone_speed <= 0.f)
                    {
                        stone_bodies[index]->SetLinearVelocity(b2Vec2_zero);
                        is_awake.erase(std::remove(is_awake.begin(), is_awake.end(), index), is_awake.end());
                    }
                    else
                    {
                        float const yaw = YawRate(stone_speed, angular_velocity) * seconds_per_frame;
                        float const longitudinal_velocity = new_stone_speed * std::cos(yaw);
                        float const transverse_velocity = new_stone_speed * std::sin(yaw);
                        b2Vec2 const &e_longitudinal = normalized_stone_velocity;
                        b2Vec2 const e_transverse = e_longitudinal.Skew();
                        b2Vec2 const new_stone_velocity = longitudinal_velocity * e_longitudinal + transverse_velocity * e_transverse;
                        stone_bodies[index]->SetLinearVelocity(new_stone_velocity);
                    }
                }

                // 角速度を計算
                if (std::abs(angular_velocity) > EPSILON)
                {
                    float const angular_accel = AngularAcceleration(stone_speed) * seconds_per_frame;
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
                seconds_per_frame,
                8,  // velocityIterations (公式マニュアルでの推奨値は 8)
                3); // positionIterations (公式マニュアルでの推奨値は 3)
        }
    }

    void SetStones()
    {
        // update bodies
        int ally_position_size = shot / 2 + 1;
        int opponent_position_size = shot / 2 + 9;
        for (size_t i = 0; i < ally_position_size; ++i)
        {
            auto &stone = stones[i];
            digitalcurling3::Vector2 position = stone.position;
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

        for (size_t i = 8; i < opponent_position_size; ++i)
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

            // if (shot < 5)
            // {
            //     IsFreeGuardZone();
            // }
        }
    }

    void SetVelocity(float velocity_x, float velocity_y, float angular_velocity)
    {
        // 自チームの全ストーン情報の後に相手チームの全ストーン情報が格納されているため、先攻後攻とshot数から、現在投球するストーンのインデックス番号を計算して、そこに速度と角速度をセットする

        int index = shot / 2 + !my_team_flag * 8;
        stone_bodies[index]->SetLinearVelocity(b2Vec2(velocity_x, velocity_y));
        stone_bodies[index]->SetAngularVelocity(angular_velocity);
        stone_bodies[index]->SetEnabled(true);
        stone_bodies[index]->SetAwake(true);
        stone_bodies[index]->SetTransform(b2Vec2(stones[index].position.x, stones[index].position.y), 0.f);
        is_awake.push_back(index);
        moved.push_back(index);
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

    inline std::vector<digitalcurling3::StoneData> GetStones()
    {
        std::vector<digitalcurling3::StoneData> stones_data;
        for (auto body : stone_bodies)
        {
            auto position = body->GetPosition();
            stones_data.push_back({digitalcurling3::Vector2(position.x, position.y)});
        }
        return stones_data;
    }
};

// メンバ変数を持つクラスを定義
class MSSimulator
{
public:
    std::vector<digitalcurling3::StoneData> storage;
    std::vector<std::vector<digitalcurling3::StoneData>> simulated_stones;
    std::vector<std::vector<unsigned int>> local_free_guard_zone_flags;
    std::vector<std::vector<std::vector<digitalcurling3::StoneData>>> local_simulated_stones;
    std::vector<unsigned int> free_guard_zone_flags;
    float y_velocity;
    std::vector<Simulator *> simulators;
    int shot;
    bool my_team_flag;
    int score;
    int end;
    int num_threads;
    int thread_id;
    int index;
    MSSimulator() : storage(), shot(), my_team_flag(), score() {
        state_values.reserve(96); // 94回のシミュレーションを行った後、の盤面評価した際の値を格納する
        simulated_stones.reserve(96);
        x_velocities.reserve(96);
        free_guard_zone_flags.reserve(96);
        storage.reserve(16);

        for (float i = -0.24; i <= 0.24; i += 0.01)
        {
            x_velocities.push_back(std::make_pair(i, cw));
            x_velocities.push_back(std::make_pair(i, ccw));
        }
        omp_set_num_threads(8);
        num_threads = omp_get_max_threads();
        simulators.resize(num_threads);
        local_free_guard_zone_flags.resize(num_threads, std::vector<unsigned int>(12, 0));
        local_simulated_stones.resize(num_threads, std::vector<std::vector<digitalcurling3::StoneData>>(12));
        #pragma omp parallel num_threads(num_threads)
        {
            // Empty block. No operations are performed in the threads.
        }
    }


    // std::pair<py::list, bool> main(py::array_t<double> stone_positions, bool my_team_flag, int shot, int score, int end, double y_velocity, int angular_velocity)
    void main(py::array_t<double> stone_positions, bool my_team_flag, int shot, int score, int end, double y_velocity)
    {
        this->shot = shot;
        this->my_team_flag = my_team_flag;
        this->score = score;
        this->end = end;
        this->y_velocity = y_velocity;
        for (int i = 0; i < 16; i++)
        {
            storage.push_back(digitalcurling3::StoneData(digitalcurling3::Vector2(stone_positions.at(2*i), stone_positions.at(2*i+1))));
        }

        for (int i = 0; i < num_threads; i++) {
            simulators[i] = new Simulator(storage);
            simulators[i]->ChangeShotAndFlag(this->shot, this->my_team_flag);
        }

        // std::vector<std::vector<unsigned int>> local_free_guard_zone_flags(num_threads, std::vector<unsigned int>(12, 0));
        // std::vector<std::vector<std::vector<digitalcurling3::StoneData>>> local_simulated_stones(num_threads, std::vector<std::vector<digitalcurling3::StoneData>>(12));

        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int i = 0; i < 96; ++i) {
            int thread_id = omp_get_thread_num();

            simulators[thread_id]->SetStones();
            simulators[thread_id]->SetVelocity(x_velocities[i].first, this->y_velocity, x_velocities[i].second);
            simulators[thread_id]->Step(0.002);
            local_free_guard_zone_flags[thread_id][i%12] = simulators[thread_id]->IsInPlayarea();
            local_simulated_stones[thread_id][i%12] = simulators[thread_id]->GetStones();
        }
        int index = 0;
        simulated_stones.clear();
        for (int i = 0; i < num_threads; i++) {
            for (int j = 0; j < 12; j++) {
                simulated_stones.push_back(local_simulated_stones[i][j]);
            }
        }

        // for (int i = 0; i < 96; ++i) {
        //     for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
        //         if (local_free_guard_zone_flags[thread_id][i]) {
        //             free_guard_zone_flags[i] = local_free_guard_zone_flags[thread_id][i];
        //             simulated_stones[i] = local_simulated_stones[thread_id][i];
        //             break;
        //         }
        //     }
        // }
    }
};
//         #pragma omp parallel for num_threads(num_threads) schedule(static) private (simulated_stones, free_guard_zone_flags)
//         for (int i = 0; i < 96; ++i) {
//             int thread_id = omp_get_thread_num();

//             simulators[thread_id]->SetStones();
//             simulators[thread_id]->SetVelocity(x_velocities[i].first, this->y_velocity, x_velocities[i].second);
//             simulators[thread_id]->Step(0.001);
//             free_guard_zone_flags[i] = simulators[thread_id]->IsInPlayarea();
//             // std::lock_guard<std::mutex> guard(cout_mutex);
//             // simulated_stones[i] = simulators[thread_id]->GetStones();
//             // simulated_stones[i] = stones_and_flag.first;
//             // free_guard_zone_flags[i] = stones_and_flag.second;
//         }        
//     };
// };

// main関数

PYBIND11_MODULE(simulator, m)
{
    py::class_<MSSimulator>(m, "Simulator")
        .def(py::init<>())
        .def("main", &MSSimulator::main);
}