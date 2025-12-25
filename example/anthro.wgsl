struct Agent {
    pos: vec2f,
    color: vec3f,
    alive: f32,
    pop: f32,
    dir: f32, 
    cooldown: f32,
}

#storage agents array<Agent, 128>

// --- 지형 및 노이즈 함수 ---
fn pcg2d(p: vec2u) -> vec2f {
    var v = p * 1664525u + 1013904223u;
    v.x += v.y * 1664525u; v.y += v.x * 1664525u;
    v = v ^ (v >> vec2u(16u));
    v.x += v.y * 1664525u; v.y += v.x * 1664525u;
    return vec2f(v) * (1.0 / f32(0xffffffffu));
}

fn pcg3d(p: vec3u) -> vec3f {
    var v = p * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v = v ^ (v >> vec3u(16u));
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
    return pow(fbm_tiled(uv * world_scale, world_scale), 1.5);
}

fn sample_food(pos: vec2f, size: vec2u) -> f32 {
    let coords = vec2i(fract(pos) * vec2f(size));
    return textureLoad(pass_in, coords, 0, 0).y;
}

fn sample_height(pos: vec2f, size: vec2u) -> f32 {
    let coords = vec2i(fract(pos) * vec2f(size));
    return textureLoad(pass_in, coords, 0, 0).x;
}

@compute @workgroup_size(16, 16)
fn main_compute(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(screen);
    let pos_i = vec2i(id.xy);
    let uv = vec2f(id.xy) / vec2f(size);
    let sea_level = 0.5;
    
    let h = get_height(uv);
    var food = textureLoad(pass_in, pos_i, 0, 0).y;
    let max_food = select(0.05, 1.0, h >= sea_level);
    
    if (time.frame == 0u) { food = max_food; }
    else { food = min(food + 0.0005, max_food); }

    let agent_count = 128u;
    for (var i = 0u; i < agent_count; i++) {
        if (agents[i].alive > 0.5) {
            let dist = distance(agents[i].pos * vec2f(size), vec2f(id.xy));
            let consume_radius = sqrt(agents[i].pop) * 2.4; 
            if (dist < consume_radius) {
                let norm_dist = dist / consume_radius;
                let falloff = 0.5 * (1.0 + cos(norm_dist * 3.14159)); 
                let consumption_rate = falloff * agents[i].pop * 0.0004;
                food *= clamp(1.0 - consumption_rate, 0.0, 1.0);
            }
        }
    }

    if (id.x == 0 && id.y == 0) {
        for (var i = 0u; i < agent_count; i++) {
            var a = agents[i];
            if (time.frame == 0u) {
                if (i == 0u) { a.pos = vec2f(0.5, 0.5); a.color = vec3f(1.0, 0.8, 0.2); a.alive = 1.0; a.pop = 30.0; a.dir = 0.0; a.cooldown = 0.0; }
                else { a.alive = 0.0; }
            } else if (a.alive > 0.5) {
                let r_seed = pcg2d(vec2u(bitcast<u32>(a.pos.x * 1537.0), i) + vec2u(time.frame));
                let cur_food = sample_food(a.pos, size);
                let cur_h = sample_height(a.pos, size);
                
                // 1. 관성 감쇠(강화됨) 및 센서 벡터 합산
                // Decay를 0.7로 강화하여 식량 경사가 없으면 더 빨리 멈춤
                var move_vec = vec2f(cos(a.dir), sin(a.dir)) * 2.5 * 0.7; 
                
                let sensor_dist = 0.025;
                var sensor_sum = vec2f(0.0);
                for (var d = 0.0; d < 8.0; d += 1.0) {
                    let angle = d * (6.28318 / 8.0);
                    let s_pos = a.pos + vec2f(cos(angle), sin(angle)) * sensor_dist;
                    let s_food = sample_food(s_pos, size);
                    sensor_sum += vec2f(cos(angle), sin(angle)) * s_food;
                }
                move_vec += sensor_sum;

                // 2. 방향 및 속도 결정
                a.dir = atan2(move_vec.y, move_vec.x) + (r_seed.x - 0.5) * 0.05;
                
                let weight_factor = mix(1.2, 0.3, clamp(a.pop / 150.0, 0.0, 1.0));
                let base_speed = mix(0.00005, 0.0005, clamp(cur_h * 2.0, 0.1, 1.0));
                let speed = base_speed * clamp(length(move_vec) * 1.5, 0.05, 2.5) * weight_factor;

                a.pos = fract(a.pos + vec2f(cos(a.dir), sin(a.dir)) * speed);

                // 3. 인구 성장 및 섭취
                let demand = a.pop * 0.001;
                if (cur_food > demand) { a.pop += (cur_food - demand) * 0.5; }
                else { a.pop -= (demand - cur_food) * 10.0 + 0.05; }
                
                a.cooldown = max(a.cooldown - 1.0, 0.0);

                // 4. 돌연변이 로직 복구 (일정 시간마다 확률적 전이)
                if (r_seed.x < 0.1) {
                    // 인구가 많을수록 변이 폭이 좁아지도록 설계 (안정성)
                    let mutation_strength = 0.2 / (sqrt(a.pop) + 1.0);
                    let color_mut = pcg3d(vec3u(i, time.frame, 77u)) - 0.5;
                    a.color = clamp(a.color + color_mut * mutation_strength, vec3f(0.1), vec3f(1.0));
                }

                // 5. 확률적 분열 (70:30)
                let split_chance = clamp((a.pop - 100.0) * 0.01, 0.0, 0.5);
                if (r_seed.y < split_chance) {
                    for (var j = 0u; j < 4u; j++) {
                        let c_idx = u32(pcg2d(vec2u(i, j + time.frame)).x * 128.0) % 128u;
                        if (agents[c_idx].alive < 0.5 && c_idx != i) {
                            let child_pop = a.pop * 0.3;
                            a.pop *= 0.7;
                            a.cooldown = 100.0;
                            agents[c_idx] = Agent(a.pos, a.color, 1.0, child_pop, r_seed.x * 6.28, 100.0);
                            break;
                        }
                    }
                }
                
                // 6. 조건부 병합
                let other_idx = u32(r_seed.y * 128.0) % 128u;
                if (other_idx != i && agents[other_idx].alive > 0.5 && a.cooldown <= 0.0 && agents[other_idx].cooldown <= 0.0) {
                    if (distance(a.pos, agents[other_idx].pos) < 0.02) {
                        let total_pop = a.pop + agents[other_idx].pop;
                        if (distance(a.color, agents[other_idx].color) < (0.8 / (total_pop * 0.01 + 1.0))) {
                            a.color = mix(a.color, agents[other_idx].color, agents[other_idx].pop / total_pop);
                            a.pop = total_pop; agents[other_idx].alive = 0.0;
                        }
                    }
                }
                if (a.pop < 1.0) { a.alive = 0.0; }
            }
            agents[i] = a;
        }
    }

    // --- 렌더링 ---
    var final_col: vec3f;
    if (h < sea_level) {
        final_col = mix(vec3f(0.02, 0.05, 0.1), vec3f(0.1, 0.2, 0.3), h / sea_level) + vec3f(0, 0.15, 0.05) * food;
    } else {
        final_col = mix(vec3f(0.2, 0.15, 0.1), vec3f(0.4, 0.4, 0.3), (h - sea_level) * 2.0) + vec3f(0.02, 0.4, 0.08) * food;
    }

    let screen_pos = uv * vec2f(size);
    for (var i = 0u; i < 128u; i++) {
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
