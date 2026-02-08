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

py::array_t<double, 3> convert_stonedata(const digitalcurling3::StoneDataVector &simulated_stones, std::size_t stones_per_team_out)
{
    const int num_coordinates = 2; // x and y coordinates per stone

    const py::ssize_t teams_dim = static_cast<py::ssize_t>(num_teams);
    const py::ssize_t stones_dim = static_cast<py::ssize_t>(stones_per_team_out);
    const py::ssize_t coords_dim = static_cast<py::ssize_t>(num_coordinates);
    py::array_t<double, 3> stones_positions(py::array::ShapeContainer({teams_dim, stones_dim, coords_dim}));
    py::detail::unchecked_mutable_reference<double, 3> buf = stones_positions.mutable_unchecked<3>();

    // internal stone indexing is always 8 per team: [0..7]=team0, [8..15]=team1
    for (size_t team = 0; team < num_teams; ++team)
    {
        for (size_t stone = 0; stone < stones_per_team_out; ++stone)
        {
            const size_t internal_index = team * 8 + stone;
            buf(team, stone, 0) = simulated_stones.stones[internal_index].position.x;
            buf(team, stone, 1) = simulated_stones.stones[internal_index].position.y;
        }
    }

    return stones_positions;
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

void SimulatorFCV1::change_shot(int total_shot)
{
    this->total_shot = total_shot;
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
            in_free_guard_zone.push_back(static_cast<int>(i));
        }
    }
}

// ファイブロックルール対応用関数
void SimulatorFCV1::is_in_playarea()
{
    for (int i : in_free_guard_zone)
    {
        b2Body *body = stone_bodies[i];
        float position_x = body->GetPosition().x;
        float position_y = body->GetPosition().y;
        if (position_y > y_upper_limit || position_x > stone_x_upper_limit || position_x < stone_x_lower_limit || (position_x == 0.0f && position_y == 0.0f))
        {
            for (int i = 0; i < kStoneMax; ++i)
            {
                digitalcurling3::StoneData stone = stones[i];
                stone_bodies[i]->SetTransform(b2Vec2(stone.position.x, stone.position.y), 0.f);
                if (stone.position.x == 0.f && stone.position.y == 0.f)
                {
                    stone_bodies[i]->SetEnabled(false);
                    stone_bodies[i]->SetAwake(false);
                }
                else
                {
                    stone_bodies[i]->SetEnabled(true);
                    stone_bodies[i]->SetAwake(true);
                }
            }
        }
    }
}

// ノーティックルール対応用関数
bool SimulatorFCV1::on_center_line(b2Body *body)
{
    if (std::abs(body->GetPosition().x) <= kStoneRadius)
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
        if (position_y > y_lower_limit && position_y < (tee_line - house_radius) && on_center_line(body))
        {
            is_no_tick.push_back(static_cast<int>(i));
        }
    }
}

// ノーティックルール対応用関数
void SimulatorFCV1::no_tick_rule()
{
    for (int i : is_no_tick)
    {
        b2Body *body = stone_bodies[i];
        float position_x = body->GetPosition().x;
        float position_y = body->GetPosition().y;
        if (std::abs(position_x) > kStoneRadius || (position_x == 0.0f && position_y == 0.0f))
        {
            for (size_t j = 0; j < kStoneMax; ++j)
            {
                digitalcurling3::StoneData stone = stones[j];
                stone_bodies[j]->SetTransform(b2Vec2(stone.position.x, stone.position.y), 0.f);
            }
            break;
        }
    }
}

void SimulatorFCV1::modified_fgz_checker()
{
    protected_stones_modified_fgz.clear();
    for (size_t i = 0; i < kStoneMax; ++i)
    {
        b2Body *body = stone_bodies[i];
        float position_x = body->GetPosition().x;
        float position_y = body->GetPosition().y;
        // 投球前に「プレー中」の石を保護対象にする（ハウス内も含む）
        // ここでは「プレーから取り除かれたか（場外/無効化/原点扱い）」のみを違反として判定したいので、
        // 投球前にプレー中である石だけを記録する。
        if (position_x == 0.0f && position_y == 0.0f)
        {
            continue;
        }
        if (position_x > stone_x_upper_limit || position_x < stone_x_lower_limit || position_y > y_upper_limit || position_y < y_lower_limit)
        {
            continue;
        }
        protected_stones_modified_fgz.push_back(static_cast<int>(i));
    }
}

void SimulatorFCV1::modified_fgz_rule()
{
    // ルール:
    // そのエンドの最初の3投は、(ハウス内を含む) 既存の石をプレーから取り除いてはいけない。
    // 違反した場合は投球した石を取り除き、動いた石は投球前の位置に戻す。
    // この実装では、"取り除いた" を「プレーエリア外へ出た / 無効化された / (0,0)扱いになった」として扱う。
    bool violation = false;
    for (int id : protected_stones_modified_fgz)
    {
        b2Body *body = stone_bodies[id];
        b2Vec2 position = body->GetPosition();

        const bool removed_by_engine = !body->IsEnabled();
        const bool treated_as_removed = (position.x == 0.0f && position.y == 0.0f);
        const bool out_of_play = (position.x > stone_x_upper_limit || position.x < stone_x_lower_limit || position.y > y_upper_limit || position.y < y_lower_limit);

        if (removed_by_engine || treated_as_removed || out_of_play)
        {
            violation = true;
            break;
        }
    }

    if (!violation)
    {
        return;
    }

    // 投球前状態(stones)へ復元。
    // 投球石は投球前は (0,0) として渡される想定なので、結果的に取り除かれる。
    for (size_t i = 0; i < kStoneMax; ++i)
    {
        const digitalcurling3::StoneData &stone = stones[i];
        stone_bodies[i]->SetTransform(b2Vec2(stone.position.x, stone.position.y), 0.f);
        stone_bodies[i]->SetLinearVelocity(b2Vec2_zero);
        stone_bodies[i]->SetAngularVelocity(0.f);

        if (stone.position.x == 0.f && stone.position.y == 0.f)
        {
            stone_bodies[i]->SetEnabled(false);
            stone_bodies[i]->SetAwake(false);
        }
        else
        {
            stone_bodies[i]->SetEnabled(true);
            stone_bodies[i]->SetAwake(true);
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
                digitalcurling3::Vector2 stone_position = {stone_bodies[index]->GetPosition().x, stone_bodies[index]->GetPosition().y};
                StonePosition pos = {index, stone_position.x, stone_position.y};
                trajectory.push_back(pos);

                // ストーンがシート外の場合は計算から除外
                if (stone_position.x > stone_x_upper_limit || stone_x_lower_limit > stone_position.x)
                {
                    stone_bodies[index]->SetTransform(b2Vec2(0.f, 0.f), 0.f);
                    stone_bodies[index]->SetAwake(false);
                    stone_bodies[index]->SetEnabled(false);
                    is_awake.erase(std::remove(is_awake.begin(), is_awake.end(), index), is_awake.end());
                    continue;
                }
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
    for (size_t i = 0; i < kStoneMax; ++i)
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
}

void SimulatorFCV1::set_velocity(float velocity_x, float velocity_y, float angular_velocity, unsigned int shot_per_team, unsigned int team_id, unsigned int applied_rule)
{
    this->applied_rule = applied_rule;
    this->shot_per_team = shot_per_team;
    // 投球するストーンは (shot_per_team, team_id) で一意に決まる。
    // ミックスダブルス等での置き石対応(+1など)は、set_velocity 呼び出し前に shot_per_team を調整して渡す。
    int index = static_cast<int>(this->shot_per_team) + static_cast<int>(team_id) * 8;

    stone_bodies[index]->SetLinearVelocity(b2Vec2(velocity_x, velocity_y));
    stone_bodies[index]->SetAngularVelocity(angular_velocity);
    stone_bodies[index]->SetEnabled(true);
    stone_bodies[index]->SetAwake(true);
    stone_bodies[index]->SetTransform(b2Vec2(0.0f, 0.0f), 0.f);
    is_awake.push_back(index);
    moved.push_back(index);

    if (this->total_shot < 5)
    {
        if (applied_rule == 0)    // applied_rule=0: apply five rock rule
        {
            freeguardzone_checker();
        }
        else if (applied_rule == 1) // applied_rule=1: apply no tick rule
        {
            no_tick_checker();
        }
    }
    if (applied_rule == 2) // modified free guard zone rule
    {
        // modified FGZ: 最初の3投は全てのプレー中ストーンを保護（ハウス内含む）
        // 判定と復元は get_stones() 側で行う。
        if (this->total_shot < 3)
        {
            modified_fgz_checker();
        }
    }
}

digitalcurling3::StoneDataVector SimulatorFCV1::get_stones()
{
    if (this->total_shot < 5)
    {
        if (this->applied_rule == 0)
        {
            is_in_playarea();
        }
        else if (this->applied_rule == 1)
        {
            no_tick_rule();
        }
    }

    if (this->applied_rule == 2 && this->total_shot < 3)
    {
        modified_fgz_rule();
    }
    
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

StoneSimulator::StoneSimulator() : storage(), trajectory()
{
    storage.reserve(16);
}

/// \brief Function to call from python
/// \param[in] stone_positions
///   - Standard: 16 stones (8 per team). Order: team0[0..7], team1[0..7].
///   - Mixed doubles: 12 stones (6 per team). Order: team0[0..5], team1[0..5].
///   Accepts either a flat array of length (stones*2) or a 2D array of shape (stones, 2).
/// \param[in] total_shot The number of shots
/// \param[in] x_velocities The x component of the velocity of the stone to be thrown
/// \param[in] y_velocities The y component of the velocity of the stone to be thrown
/// \param[in] angular_sign 1 -> cw, -1 -> ccw
/// \param[in] team_id The team that throws the stone. Team0 or Team1
/// \param[in] shot_per_team The number of shots per team
/// \param[in] applied_rule The rule to be applied. 0 -> five rock rule, 1 -> no tick rule, 2 -> modified fgz
/// \returns The positions of the stones after the simulations
std::tuple<py::array_t<double, 3>, py::list> StoneSimulator::simulator(py::array_t<double> stone_positions, int total_shot, double x_velocity, double y_velocity, int angular_sign, unsigned int team_id, unsigned int shot_per_team, unsigned int applied_rule)
{
    this->total_shot = total_shot;
    this->shot_per_team = shot_per_team;
    this->team_id = team_id;
    storage.clear();
    this->x_velocity = x_velocity;
    this->y_velocity = y_velocity;
    this->angular_velocity = angular_sign * cw;

    // Parse input: allow 16 stones (standard) or 12 stones (mixed doubles).
    // Internally we always keep 16 stone slots (8 per team) for consistent IDs.
    size_t stones_in_input = 0;
    std::vector<std::pair<double, double>> input_xy;
    input_xy.reserve(16);

    const py::buffer_info buf_info = stone_positions.request();
    if (buf_info.ndim == 1)
    {
        const size_t n = static_cast<size_t>(buf_info.size);
        if (n != 32 && n != 24)
        {
            throw py::value_error("stone_positions must be length 32 (16 stones) or 24 (12 stones) for 1D input");
        }
        stones_in_input = n / 2;
        const py::detail::unchecked_reference<double, 1> r = stone_positions.unchecked<1>();
        for (size_t i = 0; i < stones_in_input; ++i)
        {
            input_xy.emplace_back(r(static_cast<py::ssize_t>(2 * i)), r(static_cast<py::ssize_t>(2 * i + 1)));
        }
    }
    else if (buf_info.ndim == 2)
    {
        if (buf_info.shape.size() < 2 || buf_info.shape[1] != 2)
        {
            throw py::value_error("stone_positions 2D input must have shape (N, 2)");
        }
        const size_t n = static_cast<size_t>(buf_info.shape[0]);
        if (n != 16 && n != 12)
        {
            throw py::value_error("stone_positions must have N=16 (standard) or N=12 (mixed doubles) for 2D input");
        }
        stones_in_input = n;
        const py::detail::unchecked_reference<double, 2> r = stone_positions.unchecked<2>();
        for (size_t i = 0; i < stones_in_input; ++i)
        {
            input_xy.emplace_back(r(static_cast<py::ssize_t>(i), 0), r(static_cast<py::ssize_t>(i), 1));
        }
    }
    else
    {
        throw py::value_error("stone_positions must be a 1D or 2D numpy array");
    }

    // Build internal 16-slot state.
    storage.clear();
    storage.resize(16, digitalcurling3::StoneData(digitalcurling3::Vector2(0.0, 0.0)));

    if (stones_in_input == 16)
    {
        for (size_t i = 0; i < 16; ++i)
        {
            storage[i] = digitalcurling3::StoneData(digitalcurling3::Vector2(static_cast<float>(input_xy[i].first), static_cast<float>(input_xy[i].second)));
        }
    }
    else if (stones_in_input == 12)
    {
        // Mixed doubles: 6 stones per team. Map to internal indices [0..5] and [8..13].
        for (size_t i = 0; i < 6; ++i)
        {
            storage[i] = digitalcurling3::StoneData(digitalcurling3::Vector2(static_cast<float>(input_xy[i].first), static_cast<float>(input_xy[i].second)));
            storage[8 + i] = digitalcurling3::StoneData(digitalcurling3::Vector2(static_cast<float>(input_xy[6 + i].first), static_cast<float>(input_xy[6 + i].second)));
        }
    }
    else
    {
        throw py::value_error("stone_positions must contain 12 or 16 stones");
    }

    simulatorFCV1 = new SimulatorFCV1(storage);
    simulatorFCV1->change_shot(this->total_shot);
    simulatorFCV1->set_stones();
    if (applied_rule == 2) // modified free guard zone rule
    {
        // ミックスダブルス想定: 各チームの index 0 を置き石にするため、投球石は +1 した投数で割り当てる。
        simulatorFCV1->set_velocity(this->x_velocity, this->y_velocity, this->angular_velocity, this->shot_per_team + 1, this->team_id, applied_rule);
    }
    else
    {
        // normal five rock rule or no tick rule
        simulatorFCV1->set_velocity(this->x_velocity, this->y_velocity, this->angular_velocity, this->shot_per_team, this->team_id, applied_rule);
    }
    

    trajectory = simulatorFCV1->step(0.001f);
    simulated_stones = simulatorFCV1->get_stones();

    const size_t stones_per_team_out = (stones_in_input == 12) ? 6 : 8;
    stones_positions = convert_stonedata(simulated_stones, stones_per_team_out);

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

    return std::make_tuple(stones_positions, trajectory_list);
}

// main関数

PYBIND11_MODULE(simulator, m)
{
    py::class_<StoneSimulator>(m, "StoneSimulator")
        .def(py::init<>())
        .def("simulator", &StoneSimulator::simulator);
}