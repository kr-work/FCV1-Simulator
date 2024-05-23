#include <stdio.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <box2d/box2d.h>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <pybind11/stl.h>
#include "fcv1_simulator.hpp"
#include <omp.h> // OpenMP
#include <set>
#include <thread>
#include <chrono>

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
// constexpr float stone_radius = 0.145f;
namespace py = pybind11;
std::vector<float> x_velocities;
std::vector<float> state_values;
using json = nlohmann::json;

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

    std::pair<std::vector<digitalcurling3::StoneData>, bool> GetStones()
    {
        free_guard_zone = IsInPlayarea();
        std::vector<digitalcurling3::StoneData> stones_data;
        for (auto body : stone_bodies)
        {
            auto position = body->GetPosition();
            stones_data.push_back({digitalcurling3::Vector2(position.x, position.y)});
        }
        return {stones_data, free_guard_zone};
    }
};

// メンバ変数を持つクラスを定義
class MSSimulator
{
public:
    std::vector<digitalcurling3::StoneData> storage;
    std::vector<digitalcurling3::StoneData> simulated_stones;
    py::list simulated_stones_position;
    std::pair<std::vector<digitalcurling3::StoneData>, bool> stones_and_flag;
    bool free_guard_zone;
    float y_velocity;
    std::vector<Simulator *> simulators;
    int shot;
    bool my_team_flag;
    int score;
    int angle;
    float angular_velocity;
    int end;
    int num_threads;
    MSSimulator() : storage(), shot(), my_team_flag(), score(), angular_velocity() {
        state_values.reserve(94); // 94回のシミュレーションを行った後、の盤面評価した際の値を格納する
        x_velocities.reserve(47);
        storage.reserve(16);

        for (float i = -0.23; i <= 0.23; i += 0.01)
        {
            x_velocities.push_back(i);
        }
        num_threads = omp_get_max_threads();
        std::cout << num_threads << std::endl;
        #pragma omp parallel num_threads(num_threads)
        {
            // Empty block. No operations are performed in the threads.
        }
    }


    std::pair<py::list, bool> main(py::array_t<double> stone_positions, bool my_team_flag, int shot, int score, int end, double y_velocity, int angular_velocity)
    {
        std::chrono::system_clock::time_point t1, t2, t3, t4, t5, t6, t7, t8;
        t1 = std::chrono::system_clock::now();
        this->shot = shot;
        this->my_team_flag = my_team_flag;
        this->score = score;
        this->end = end;
        this->y_velocity = y_velocity;
        this->angular_velocity = angular_velocity*cw;
        for (int i = 0; i < 16; i++)
        {
            storage.push_back(digitalcurling3::StoneData(digitalcurling3::Vector2(stone_positions.at(2*i), stone_positions.at(2*i+1))));
        }
        t2 = std::chrono::system_clock::now();
        // for (const auto stone : storage)
        // {
        //     std::cout << stone.position.x << " " << stone.position.y << std::endl;
        // }
        // shot = state_json["shot"];
        // hummer = state_json["hummer"];
        // score = state_json["score_diff"];
        // end = state_json["end"];
        // velocity.vel = b2Vec2(state_json["velocity"]["vx"], state_json["velocity"]["vy"]);
        // angle = state_json["velocity"]["angle"];
        // angular_velocity = angle * cw;
        // auto my_team_stones = state_json["stones"]["my_team"];
        // auto opponent_team_stones = state_json["stones"]["opponent_team"];

        // for (size_t i = 0; i < 8; ++i)
        // {
        //     digitalcurling3::StoneData data;

        //     data.position = digitalcurling3::Vector2(my_team_stones["stone" + std::to_string(i) + "_position"]["x"], 
        //                                             my_team_stones["stone" + std::to_string(i) + "_position"]["y"]);
        //     storage.push_back(data);
        // }

        // for (size_t i = 0; i < 8; ++i)
        // {
        //     digitalcurling3::StoneData data;

        //     data.position = digitalcurling3::Vector2(opponent_team_stones["stone" + std::to_string(i) + "_position"]["x"], 
        //                                             opponent_team_stones["stone" + std::to_string(i) + "_position"]["y"]);
        //     storage.push_back(data);
        // }

        // int a = omp_get_max_threads();
        // std::cout << a << " concurrent threads are supported.\n";
        // for (int i = 0; i < num_threads; i++) {
        //     simulators[i] = new Simulator(storage);
        // }
        Simulator *simulator = new Simulator(storage);
        t3 = std::chrono::system_clock::now();
        simulator->ChangeShotAndFlag(this->shot, this->my_team_flag);
        t4 = std::chrono::system_clock::now();
        simulator->SetStones();
        t5 = std::chrono::system_clock::now();
        simulator->SetVelocity(-0.05, this->y_velocity, this->angular_velocity);
        t6 = std::chrono::system_clock::now();
        simulator->Step(0.001);
        t7 = std::chrono::system_clock::now();
        stones_and_flag = simulator->GetStones();
        simulated_stones = stones_and_flag.first;
        free_guard_zone = stones_and_flag.second;
        for (const digitalcurling3::StoneData &stone : simulated_stones)
        {
            simulated_stones_position.append(Vector2Totuple(stone.position));
            // std::cout << stone.position.x << " " << stone.position.y << std::endl;
        }
        t8 = std::chrono::system_clock::now();
        std::cout << "main: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << "ms" << std::endl;
        std::cout << "main: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count() << "ms" << std::endl;
        std::cout << "main: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count() << "ms" << std::endl;
        std::cout << "main: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t5 - t4).count() << "ms" << std::endl;
        std::cout << "main: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t6 - t5).count() << "ms" << std::endl;
        std::cout << "main: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t7 - t6).count() << "ms" << std::endl;
        std::cout << "main: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t8 - t7).count() << "ms" << std::endl;
        return {simulated_stones_position, free_guard_zone};
        
    };
};

// main関数

PYBIND11_MODULE(simulator, m)
{
    py::class_<MSSimulator>(m, "Simulator")
        .def(py::init<>())
        .def("main", &MSSimulator::main);
}