struct Agent {
    pos: vec2f,
    color: vec3f,
    alive: f32,
    pop: f32,
    dir: f32, 
}

#storage agents array<Agent, 1024>

// --- 해시 및 지형 함수 (원상 복귀) ---
fn pcg2d(p: vec2u) -> vec2f {
    var v = p * 1664525u + 1013904223u;
    v.x += v.y * 1664525u; v.y += v.x * 1664525u;
    v = v ^ (v >> vec2u(16u));
    v.x += v.y * 1664525u; v.y += v.x * 1664525u;
    v = v ^ (v >> vec2u(16u));
    return vec2f(v) * (1.0 / f32(0xffffffffu));
}

fn pcg3d(p: vec3u) -> vec3f {
    var v = p * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v = v ^ (v >> vec3u(16u));
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return vec3f(v) * (1.0 / f32(0xffffffffu));
}

fn hash12(p: vec2f) -> f32 {
    var p3 = fract(vec3f(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn noise_tiled(p: vec2f, scale: f32) -> f32 {
    let i = floor(p); let f = fract(p);
    let i00 = (i + vec2f(0.0, 0.0)) % scale;
    let i10 = (i + vec2f(1.0, 0.0)) % scale;
    let i01 = (i + vec2f(0.0, 1.0)) % scale;
    let i11 = (i + vec2f(1.0, 1.0)) % scale;
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash12(i00), hash12(i10), u.x), mix(hash12(i01), hash12(i11), u.x), u.y);
}

fn fbm_tiled(p: vec2f, scale: f32) -> f32 {
    var v = 0.0; var a = 0.5; var pos = p; var s = scale;
    for (var i = 0; i < 6; i++) { v += a * noise_tiled(pos, s); pos *= 2.0; s *= 2.0; a *= 0.5; }
    return v;
}

fn get_height(uv: vec2f) -> f32 {
    let world_scale = 4.0;
    let h_raw = fbm_tiled(uv * world_scale, world_scale);
    return pow(h_raw, 1.5);
}

fn sample_food(pos: vec2f, size: vec2u) -> f32 {
    let coords = vec2i(fract(pos) * vec2f(size));
    return textureLoad(pass_in, coords, 0, 0).y;
}

fn sample_height(pos: vec2f, size: vec2u) -> f32 {
    let coords = vec2i(fract(pos) * vec2f(size));
    return textureLoad(pass_in, coords, 0, 0).x; // x 채널에 h가 저장되어 있음
}

// --- 메인 시뮬레이션 ---
@compute @workgroup_size(16, 16)
fn main_compute(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(screen);
    let pos_i = vec2i(id.xy);
    let uv = vec2f(id.xy) / vec2f(size);
    let sea_level = 0.5;
    
    // 1. 지형 및 식량 재생 (모든 픽셀 동일)
    let h = get_height(uv);
    var food = textureLoad(pass_in, pos_i, 0, 0).y;
    // 바다(h < 0.5)인 경우 식량은 항상 0입니다.
    if (h >= sea_level) { food = min(food + 0.0005, 1.0); } else { food = 0.0; }

    let agent_count = 1024u;
    for (var i = 0u; i < agent_count; i++) {
        if (agents[i].alive > 0.5) {
            let dist = distance(agents[i].pos * vec2f(size), vec2f(id.xy));
            let consume_radius = sqrt(agents[i].pop) * 1.6;
            if (dist < consume_radius) {
                let falloff = 1.0 - (dist / consume_radius);
                food = max(food - falloff * agents[i].pop * 0.00008, 0.0);
            }
        }
    }

    // 2. 에이전트 업데이트 (중앙 스레드 격리)
    if (id.x == 0 && id.y == 0) {
        for (var i = 0u; i < agent_count; i++) {
            var a = agents[i];
            
            if (time.frame == 0u) {
                if (i == 0u) {
                    a.pos = vec2f(0.5, 0.5); a.color = vec3f(1.0, 0.8, 0.2); a.alive = 1.0; a.pop = 30.0; a.dir = 0.0;
                } else { a.alive = 0.0; a.pop = 0.0; }
            } else if (a.alive > 0.5) {
                let r_seed = pcg2d(vec2u(bitcast<u32>(a.pos.x * 1537.0), i) + vec2u(time.frame));
                
                // 식량 센싱 (이미 바다인지 육지인지 정보를 포함함)
                let cur_food = sample_food(a.pos, size);
                let sensor_dist = 0.015;
                let sensor_angle = 0.65;
                let f_left  = sample_food(a.pos + vec2f(cos(a.dir - sensor_angle), sin(a.dir - sensor_angle)) * sensor_dist, size);
                let f_front = sample_food(a.pos + vec2f(cos(a.dir), sin(a.dir)) * sensor_dist, size);
                let f_right = sample_food(a.pos + vec2f(cos(a.dir + sensor_angle), sin(a.dir + sensor_angle)) * sensor_dist, size);
                
                let demand = a.pop * 0.001;

                // [최적화 핵심] 식량이 0인 곳(바다)을 회피
                // 단, 배가 고프지 않을 때만 회피 로직 작동
                if (f_front <= 0.0 && cur_food >= demand) {
                    a.dir += 0.15; // 바다 쪽이면 방향을 틈
                }

                // 조향 로직
                if (f_left > f_front && f_left > f_right) { a.dir -= 0.05; }
                else if (f_right > f_front && f_right > f_left) { a.dir += 0.05; }
                a.dir += (r_seed.x - 0.5) * 0.02;

                // 이동 속도 결정 (현재 위치 식량이 0이면 바다로 간주하여 느려짐)
                let speed = select(0.0001, 0.0005, cur_food > 0.0);
                a.pos = fract(a.pos + vec2f(cos(a.dir), sin(a.dir)) * speed);

                // 인구 변화
                if (cur_food > demand) { 
                    a.pop += (cur_food - demand) * 0.4; 
                } else { 
                    // 바다(cur_food == 0)이면 더 빨리 굶어죽음
                    let starv_rate = select(15.0, 5.0, cur_food > 0.0);
                    a.pop -= (demand - cur_food) * starv_rate + 0.05; 
                }

                // 분열/합병/변이 (기존과 동일)
                if (a.pop > 100.0) {
                    for (var j = 0u; j < 4u; j++) {
                        let c_idx = u32(pcg2d(vec2u(i, j + time.frame)).x * f32(agent_count)) % agent_count;
                        if (agents[c_idx].alive < 0.5 && c_idx != i) {
                            a.pop *= 0.45;
                            agents[c_idx] = Agent(a.pos, a.color, 1.0, a.pop, a.dir + 3.14);
                            break;
                        }
                    }
                }
                if (r_seed.x < 0.02) {
                    a.color = clamp(a.color + (pcg3d(vec3u(i, time.frame, 99u)) - 0.5) * (0.2 / (sqrt(a.pop) + 1.0)), vec3f(0.1), vec3f(1.0));
                }
                let other_idx = u32(r_seed.y * f32(agent_count)) % agent_count;
                if (other_idx != i && agents[other_idx].alive > 0.5) {
                    let other = agents[other_idx];
                    if (distance(a.pos, other.pos) < 0.02) {
                        let total_pop = a.pop + other.pop;
                        if (distance(a.color, other.color) < (0.8 / (total_pop * 0.01 + 1.0))) {
                            a.color = mix(a.color, other.color, other.pop / total_pop);
                            a.pop = total_pop;
                            agents[other_idx].alive = 0.0; 
                        }
                    }
                }
                if (a.pop < 1.0) { a.alive = 0.0; }
            }
            agents[i] = a;
        }
    }

    // --- 3. 렌더링 (지시대로 전수 조사 유지) ---
    var final_col: vec3f;
    if (h < sea_level) {
        final_col = mix(vec3f(0.02, 0.05, 0.1), vec3f(0.1, 0.2, 0.3), h / sea_level);
    } else {
        let land = mix(vec3f(0.2, 0.15, 0.1), vec3f(0.4, 0.4, 0.3), (h - sea_level) * 2.0);
        let forest = vec3f(0.02, 0.4, 0.08) * food;
        final_col = land + forest;
    }

    let screen_pos = uv * vec2f(size);
    for (var i = 0u; i < agent_count; i++) {
        if (agents[i].alive > 0.5) {
            let d = distance(agents[i].pos * vec2f(size), screen_pos);
            let r = sqrt(agents[i].pop) * 1.1 + 1.2;
            if (d < r) {
                final_col = agents[i].color;
                if (d > r * 0.8) { final_col *= 0.2; }
                break;
            }
        }
    }
    textureStore(pass_out, pos_i, 0, vec4f(h, food, 0.0, 1.0));
    textureStore(screen, pos_i, vec4f(final_col, 1.0));
}
