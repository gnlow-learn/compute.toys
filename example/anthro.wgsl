struct Agent {
    pos: vec2f,
    color: vec3f,
    alive: f32,
    pop: f32,
    dir: f32, 
}

#storage agents array<Agent, 1024>

// --- 해시 함수들 ---
fn pcg2d(p: vec2u) -> vec2f {
    var v = p * 1664525u + 1013904223u;
    v.x += v.y * 1664525u; v.y += v.x * 1664525u;
    v = v ^ (v >> vec2u(16u));
    v.x += v.y * 1664525u; v.y += v.x * 1664525u;
    v = v ^ (v >> vec2u(16u));
    return vec2f(v) * (1.0 / f32(0xffffffffu));
}

// 3차원 랜덤 값을 위한 새로운 함수
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
    let h_base = pow(h_raw, 1.5);
    let sea_level = 0.5;
    if (h_base >= sea_level) {
        let n_h = (h_base - sea_level) / (1.0 - sea_level);
        return sea_level + smoothstep(-0.1, 0.9, n_h) * (1.0 - sea_level);
    }
    return h_base;
}

fn sample_food(pos: vec2f, size: vec2u) -> f32 {
    let coords = vec2i(fract(pos) * vec2f(size));
    return textureLoad(pass_in, coords, 0, 0).y;
}

// --- 메인 시뮬레이션 ---
@compute @workgroup_size(16, 16)
fn main_compute(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(screen);
    let pos_i = vec2i(id.xy);
    let uv = vec2f(id.xy) / vec2f(size);
    let sea_level = 0.5;
    
    let h = get_height(uv);
    var food = textureLoad(pass_in, pos_i, 0, 0).y;
    if (h >= sea_level) { food = min(food + 0.0005, 1.0); } else { food = 0.0; }

    let agent_count = 1024u;
    for (var i = 0u; i < agent_count; i++) {
        if (agents[i].alive > 0.5) {
            let dist = distance(agents[i].pos * vec2f(size), vec2f(id.xy));
            let consume_radius = sqrt(agents[i].pop) * 1.6;
            if (dist < consume_radius) {
                let falloff = 1.0 - (dist / consume_radius);
                let drain = falloff * agents[i].pop * 0.00008; 
                food = max(food - drain, 0.0);
            }
        }
    }

    let tid = id.x + id.y * size.x;
    if (tid < agent_count) {
        var a = agents[tid];
        
        if (time.frame == 0u) {
            if (tid == 0u) {
                a.pos = vec2f(0.5, 0.5); a.color = vec3f(1.0, 0.8, 0.2); a.alive = 1.0; a.pop = 30.0; a.dir = 0.0;
            } else { a.alive = 0.0; }
        } else if (a.alive > 0.5) {
            let r_pos = pcg2d(vec2u(bitcast<u32>(a.pos.x * 1537.0), bitcast<u32>(a.pos.y * 1537.0)) + vec2u(tid, time.frame));
            
            // 1. 돌연변이 (pcg3d 사용으로 에러 해결)
            if (r_pos.x < 0.01) { 
                let mut_val = pcg3d(vec3u(tid, time.frame, 123u)) - 0.5;
                a.color = clamp(a.color + mut_val * 0.05, vec3f(0.1), vec3f(1.0));
            }

            // 2. 유전적 동화 (Genetic Assimilation)
            let other_idx = u32(r_pos.y * f32(agent_count)) % agent_count;
            let other = agents[other_idx];
            if (other.alive > 0.5 && other_idx != tid) {
                let d = distance(a.pos, other.pos);
                if (d < 0.05) { 
                    a.color = mix(a.color, other.color, 0.02);
                }
            }

            // 이동 및 센싱
            let sensor_dist = 0.015;
            let sensor_angle = 0.65;
            let f_left  = sample_food(a.pos + vec2f(cos(a.dir - sensor_angle), sin(a.dir - sensor_angle)) * sensor_dist, size);
            let f_front = sample_food(a.pos + vec2f(cos(a.dir), sin(a.dir)) * sensor_dist, size);
            let f_right = sample_food(a.pos + vec2f(cos(a.dir + sensor_angle), sin(a.dir + sensor_angle)) * sensor_dist, size);
            
            if (f_left > f_front && f_left > f_right) { a.dir -= 0.05; }
            else if (f_right > f_front && f_right > f_left) { a.dir += 0.05; }
            a.dir += (r_pos.x - 0.5) * 0.02;

            var step_size = 0.00004 / max(r_pos.y * r_pos.y, 0.005); 
            step_size = min(step_size, 0.0008); 
            let next_pos = fract(a.pos + vec2f(cos(a.dir), sin(a.dir)) * step_size);
            if (get_height(next_pos) >= sea_level) { a.pos = next_pos; } 
            else { a.dir += 3.14159 * (r_pos.x + 0.5); }

            // 3. 인구 변화
            let current_food = sample_food(a.pos, size);
            let demand = a.pop * 0.001;
            if (current_food > demand) {
                a.pop += (current_food - demand) * 0.4; 
            } else {
                a.pop -= (demand - current_food) * 5.0 + 0.05; 
            }

            // 4. 분열 로직
            if (a.pop > 100.0) {
                var seed = u32(r_pos.x * 12345.0) + tid;
                for (var i = 0u; i < 8u; i++) {
                    let rand_val = pcg2d(vec2u(seed, i)).x;
                    let child_idx = u32(rand_val * f32(agent_count)) % agent_count;
                    if (agents[child_idx].alive < 0.5) {
                        let parent_pop = a.pop * 0.42;
                        a.pop = parent_pop;
                        agents[child_idx].pos = a.pos;
                        agents[child_idx].alive = 1.0;
                        agents[child_idx].pop = parent_pop;
                        agents[child_idx].dir = a.dir + (pcg2d(vec2u(seed)).x * 6.28);
                        agents[child_idx].color = a.color; 
                        break; 
                    }
                    seed += 1u;
                }
            }
            if (a.pop < 1.0) { a.alive = 0.0; }
        }
        agents[tid] = a;
    }

    // --- 렌더링 ---
    var final_col: vec3f;
    if (h < sea_level) {
        final_col = mix(vec3f(0.05, 0.08, 0.12), vec3f(0.1, 0.15, 0.25), h / sea_level);
    } else {
        let land = mix(vec3f(0.25, 0.18, 0.1), vec3f(0.45, 0.45, 0.35), (h - sea_level) * 2.0);
        let forest = vec3f(0.05, 0.45, 0.1) * food;
        final_col = land + forest;
    }

    let screen_pos = uv * vec2f(size);
    for (var i = 0u; i < agent_count; i++) {
        if (agents[i].alive > 0.5) {
            let d = distance(agents[i].pos * vec2f(size), screen_pos);
            let radius = sqrt(agents[i].pop) * 1.1 + 1.5;
            if (d < radius) {
                final_col = agents[i].color;
                if (d > radius * 0.75) { final_col *= 0.25; }
                break; 
            }
        }
    }

    textureStore(pass_out, pos_i, 0, vec4f(h, food, 0.0, 1.0));
    textureStore(screen, pos_i, vec4f(final_col, 1.0));
}
