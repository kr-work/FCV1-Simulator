#include <stdio.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <nlohmann/json.hpp>
#include <pybind11/numpy.h>
#include <box2d/box2d.h>
#include <cmath>
#include <limits>
#include <pybind11/stl.h>
#include "fcv1_simulator.hpp"
#include <set>
#include <string>
#include <fstream>
#include <algorithm>


// constexpr float stone_radius = 0.145f;

json read_configfile(const std::string &filepath)
{
    std::ifstream ifs(filepath);
    json j;
    ifs >> j;
    return j;
}

// 返り値1つめ: 正規化されたベクトル
// 返り値2つめ: もとのベクトルの長さ
/// \brief To normalize the vector
/// \param[in] v The vector to be normalized
/// \returns A pair of the normalized vector and the length of the original vector
inline std::pair<b2Vec2, float> normalize(b2Vec2 const &v)
{
    b2Vec2 normalized = v;
    float length = normalized.Normalize();
    return {normalized, length};
}

/// \brief To calculate the longitudinal acceleration
/// \param[in] speed The speed of the stone
/// \returns The longitudinal acceleration
inline float longitudinal_acceleration(float speed)
{
    constexpr float kGravity = 9.80665f;
    return -(0.00200985f / (speed + 0.06385782f) + 0.00626286f) * kGravity;
}

/// \brief To calculate the yaw rate
/// \param[in] speed The speed of the stone
/// \param[in] angularVelocity The angular velocity of the stone
/// \returns The yaw rate
inline float yaw_tate(float speed, float angularVelocity)
{
    if (std::abs(angularVelocity) <= EPSILON)
    {
        return 0.f;
    }
    return (angularVelocity > 0.f ? 1.0f : -1.0f) * 0.00820f * std::pow(speed, -0.8f);
}

/// \brief To calculate the angular acceleration
/// \param[in] linearSpeed The speed of the stone
/// \returns The angular acceleration
inline float angular_acceleration(float linearSpeed)
{
    float clampedSpeed = std::max(linearSpeed, 0.001f);
    return -0.025f / clampedSpeed;
}

py::array_t<double> convert_stonedata(const digitalcurling3::StoneDataVector &simulated_stones)
{
    const size_t num_coordinates = 2; // x and y coordinates per stone
    std::vector<double> temp_result(stones_per_simulation * num_coordinates);

    for (size_t i = 0; i < stones_per_simulation; ++i)
    {
        size_t index = i * num_coordinates;
        temp_result[index] = simulated_stones.stones[i].position.x;
        temp_result[index + 1] = simulated_stones.stones[i].position.y;
    }

    py::array_t<double> result(py::array::ShapeContainer({stones_per_simulation * num_coordinates}), temp_result.data());
    return result;
}

void SimulatorFCV1::ContactListener::PostSolve(b2Contact *contact, const b2ContactImpulse *impulse)
{
    b2Body *a_body = contact->GetFixtureA()->GetBody();
    b2Body *b_body = contact->GetFixtureB()->GetBody();

    digitalcurling3::Collision collision;
    collision.a.id = static_cast<int>(a_body->GetUserData().pointer);
    collision.b.id = static_cast<int>(b_body->GetUserData().pointer);

    add_unique_id(instance_->is_awake, collision.a.id);
    add_unique_id(instance_->is_awake, collision.b.id);

    add_unique_id(instance_->moved, collision.a.id);
    add_unique_id(instance_->moved, collision.b.id);

    b2WorldManifold world_manifold;
    contact->GetWorldManifold(&world_manifold);

    collision.normal_impulse = impulse->normalImpulses[0];
    collision.tangent_impulse = impulse->tangentImpulses[0];
}

void SimulatorFCV1::ContactListener::add_unique_id(std::vector<int> &list, int id)
{
    if (std::find(list.begin(), list.end(), id) == list.end())
    {
        list.push_back(id);
    }
}

SimulatorFCV1::SimulatorFCV1(std::vector<digitalcurling3::StoneData> const &stones) : stones(stones), world(b2Vec2(0, 0)), contact_listener_(this)
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

void SimulatorFCV1::change_shot(int shot)
{
    this->shot = shot;
}

bool SimulatorFCV1::is_freeguardzone(b2Body *body)
{
    float dx = body->GetPosition().x;
    float dy = body->GetPosition().y - tee_line;
    float distance_squared = dx * dx + dy * dy;
    if (dy < 0 && distance_squared > house_radius * house_radius && body->GetPosition().y >= min_y)
    {
        return true;
    }
    return false;
}

void SimulatorFCV1::freeguardzone_checker()
{
    for (size_t i = 0; i < kStoneMax; ++i)
    {
        b2Body *body = stone_bodies[i];
        if (is_freeguardzone(body))
        {
            in_free_guard_zone.push_back(i);
        }
    }
}

// ファイブロックルール対応用関数
unsigned int SimulatorFCV1::is_in_playarea()
{
    for (int i : in_free_guard_zone)
    {
        b2Body *body = stone_bodies[i];
        if (body->GetPosition().y > y_upper_limit || body->GetPosition().x > stone_x_upper_limit || body->GetPosition().x < stone_x_lower_limit)
        {
            for (int index : moved)
            {
                digitalcurling3::StoneData stone = stones[index];
                stone_bodies[index]->SetTransform(b2Vec2(stone.position.x, stone.position.y), 0.f);
            }
            return true;
        }
    }
    return false;
}

// ノーティックルール対応用関数
bool SimulatorFCV1::on_center_line(b2Body *body)
{
    if (body->IsEnabled() && kStoneRadius - std::abs(body->GetPosition().x) < 0.0)
    {
        return true;
    }
    return false;
}

void SimulatorFCV1::no_tick_checker()
{
    for (size_t i = 0; i < kStoneMax; ++i)
    {
        b2Body *body = stone_bodies[i];
        float position_y = body -> GetPosition().y;
        if (on_center_line(body) && position_y > y_lower_limit && position_y < (tee_line - house_radius))
        {
            is_no_tick.push_back(i);
        }
    }
}

// ノーティックルール対応用関数
void SimulatorFCV1::no_tick_rule()
{
    for (int i : is_no_tick)
    {
        b2Body *body = stone_bodies[i];
        if (kStoneRadius - std::abs(body->GetPosition().x) > 0.0)
        {
            for (int index : moved)
            {
                auto stone = stones[index];
                stone_bodies[index]->SetTransform(b2Vec2(stone.position.x, stone.position.y), 0.f);
            }
            break;
        }
    }
}

std::vector<std::vector<StonePosition>> SimulatorFCV1::step(float seconds_per_frame)
{
    trajectory_list.clear();
    // simulate
    while (!is_awake.empty())
    {
        trajectory.clear();
        for (int &index : is_awake)
        {
            b2Vec2 const stone_velocity = stone_bodies[index]->GetLinearVelocity(); // copy
            auto const [normalized_stone_velocity, stone_speed] = normalize(stone_velocity);
            float const angular_velocity = stone_bodies[index]->GetAngularVelocity();

            // 速度を計算
            // ストーンが停止してる場合は無視
            if (stone_speed > EPSILON)
            {
                StonePosition pos = {index, stone_bodies[index]->GetPosition().x, stone_bodies[index]->GetPosition().y};
                trajectory.push_back(pos);
                // ストーンの速度を計算
                float const new_stone_speed = stone_speed + longitudinal_acceleration(stone_speed) * seconds_per_frame;
                if (new_stone_speed <= 0.f)
                {
                    stone_bodies[index]->SetLinearVelocity(b2Vec2_zero);
                    is_awake.erase(std::remove(is_awake.begin(), is_awake.end(), index), is_awake.end());
                }
                else
                {
                    float const yaw = yaw_tate(stone_speed, angular_velocity) * seconds_per_frame;
                    float const longitudinal_velocity = new_stone_speed * std::cos(yaw);
                    float const transverse_velocity = new_stone_speed * std::sin(yaw);
                    b2Vec2 const &e_longitudinal = normalized_stone_velocity;
                    b2Vec2 const e_transverse = e_longitudinal.Skew();
                    b2Vec2 const new_stone_velocity = longitudinal_velocity * e_longitudinal + transverse_velocity * e_transverse;
                    stone_bodies[index]->SetLinearVelocity(new_stone_velocity);
                }
            }else{
                is_awake.erase(std::remove(is_awake.begin(), is_awake.end(), index), is_awake.end());
                //if(is_awake.size() != 1){
                  //  std::cout << "size: " << is_awake.size() << ", normalized_vec_x: " << normalized_stone_velocity.x << ", normalized_vec_y: " << normalized_stone_velocity.y << ", speed: " << stone_speed << std::endl;   
                //}
            }

            // 角速度を計算
            if (std::abs(angular_velocity) > EPSILON)
            {
                float const angular_accel = angular_acceleration(stone_speed) * seconds_per_frame;
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
        trajectory_list.push_back(trajectory);

        // storage.collisions.clear();

        world.Step(
            seconds_per_frame,
            8,  // velocityIterations (公式マニュアルでの推奨値は 8)
            3); // positionIterations (公式マニュアルでの推奨値は 3)
    }
    return trajectory_list;
}

void SimulatorFCV1::set_stones()
{
    // update bodies
    int ally_position_size = shot / 2 + 1;
    int opponent_position_size = shot / 2 + 9;
    for (size_t i = 0; i < ally_position_size; ++i)
    {
        const digitalcurling3::StoneData &stone = stones[i];
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
        const digitalcurling3::StoneData &stone = stones[i];
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

        if (this->shot < 5)
        {
            freeguardzone_checker();
        }
    }
}

void SimulatorFCV1::set_velocity(float velocity_x, float velocity_y, float angular_velocity, unsigned int shot_per_team, unsigned int team_id)
{
    this->shot_per_team = shot_per_team;
    int index = this->shot_per_team + team_id * 8;
    stone_bodies[index]->SetLinearVelocity(b2Vec2(velocity_x, velocity_y));
    stone_bodies[index]->SetAngularVelocity(angular_velocity);
    stone_bodies[index]->SetEnabled(true);
    stone_bodies[index]->SetAwake(true);
    stone_bodies[index]->SetTransform(b2Vec2(0.0f, 0.0f), 0.f);
    is_awake.push_back(index);
    moved.push_back(index);
}

digitalcurling3::StoneDataVector SimulatorFCV1::get_stones()
{
    digitalcurling3::StoneDataVector stones_data;
    for (b2Body *body : stone_bodies)
    {
        b2Vec2 position = body->GetPosition();
        if (position.x > stone_x_upper_limit || position.x < stone_x_lower_limit || position.y > y_upper_limit || position.y < y_lower_limit)
        {
            body->SetTransform(b2Vec2(0.f, 0.f), 0.f);
        }
        b2Vec2 after_position = body->GetPosition();
        stones_data.stones.push_back({digitalcurling3::Vector2(after_position.x, after_position.y)});
    }
    return stones_data;
}

StoneSimulator::StoneSimulator() : storage(), shot(), trajectory()
{
    storage.reserve(16);
}

/// \brief Function to call from python
/// \param[in] stone_positions 16 stones' positions(The first 8 stones are the first attacker's stones, the last 8 stones are the second attacker's stones)
/// \param[in] shot The number of shots
/// \param[in] x_velocities The x component of the velocity of the stone to be thrown
/// \param[in] y_velocities The y component of the velocity of the stone to be thrown
/// \param[in] angular_sign 1 -> cw, -1 -> ccw
/// \param[in] team_id The team that throws the stone. Team0 or Team1
/// \param[in] shot_per_team The number of shots per team
/// \returns The positions of the stones after the simulations
std::tuple<py::array_t<double>, unsigned int, py::list> StoneSimulator::simulator(py::array_t<double> stone_positions, int total_shot, double x_velocity, double y_velocity, int angular_sign, unsigned int team_id, unsigned int shot_per_team)
{
    this->shot = total_shot;
    this->shot_per_team = shot_per_team;
    this->team_id = team_id;
    storage.clear();
    this->x_velocity = x_velocity;
    this->y_velocity = y_velocity;
    this->angular_velocity = angular_sign * cw;

    for (int i = 0; i < 16; i++)
    {
        storage.push_back(digitalcurling3::StoneData(digitalcurling3::Vector2(stone_positions.at(2 * i), stone_positions.at(2 * i + 1))));
    }

    simulatorFCV1 = new SimulatorFCV1(storage);
    simulatorFCV1->change_shot(this->shot);
    simulatorFCV1->set_stones();
    simulatorFCV1->set_velocity(this->x_velocity, this->y_velocity, this->angular_velocity, this->shot_per_team, this->team_id);

    trajectory = simulatorFCV1->step(0.001);
    free_guard_zone_flag = simulatorFCV1->is_in_playarea();
    simulated_stones = simulatorFCV1->get_stones();

    result = convert_stonedata(simulated_stones);

    count = 0;
    for (const std::vector<StonePosition> &step_stone_data : trajectory)
    {
        py::list step_list;
        for (const StonePosition &stone : step_stone_data)
        {
            if (count % 100 == 0)
            {
                step_list.append(py::make_tuple(stone.id, stone.x, stone.y));
            }
        }
        if (!step_list.empty())
        {
            trajectory_list.append(step_list);
        }
        count++;
    }

    return std::make_tuple(result, free_guard_zone_flag, trajectory_list);
}

// main関数

PYBIND11_MODULE(simulator, m)
{
    py::class_<StoneSimulator>(m, "StoneSimulator")
        .def(py::init<>())
        .def("simulator", &StoneSimulator::simulator);
}