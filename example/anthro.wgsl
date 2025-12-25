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

// 토러스 순환 샘플링 함수
fn sample_buffer(uv: vec2f, size: vec2u) -> vec4f {
    let coords = vec2i(fract(uv) * vec2f(size));
    return textureLoad(pass_in, coords, 0, 0);
}

@compute @workgroup_size(16, 16)
fn main_compute(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(screen);
    let pos_i = vec2i(id.xy);
    let uv = vec2f(id.xy) / vec2f(size);
    let sea_level = 0.5;
    
    // --- 1. 기본 지형 데이터 읽기 ---
    let h = get_height(uv);
    let prev_data = textureLoad(pass_in, pos_i, 0, 0);
    var food = prev_data.y;
    var gene_map = prev_data.zw; // z, w 채널을 지역 유전 정보(평균 색상의 일부)로 활용

    // --- 2. 초기화 (Frame 0) ---
    if (time.frame == 0u) {
        food = select(0.05, 1.0, h >= sea_level);
        gene_map = pcg2d(id.xy); // 무작위 유전자 값으로 초기화
    }

    // --- 3. 유전자 확산 및 블러 (토러스 순환) ---
    // 가우시안 블러 대용으로 인접 4픽셀과 믹스하여 서서히 번지게 함
    let offset = 1.0 / vec2f(size);
    let g_up = sample_buffer(uv + vec2f(0.0, offset.y), size).zw;
    let g_down = sample_buffer(uv - vec2f(0.0, offset.y), size).zw;
    let g_left = sample_buffer(uv - vec2f(offset.x, 0.0), size).zw;
    let g_right = sample_buffer(uv + vec2f(offset.x, 0.0), size).zw;
    gene_map = mix(gene_map, (g_up + g_down + g_left + g_right) * 0.25, 0.1);

    // --- 4. 에이전트 상호작용 및 유전자 각인 ---
    let agent_count = 128u;
    
    // 식량 소비 로직
    for (var i = 0u; i < agent_count; i++) {
        if (agents[i].alive > 0.5) {
            let dist = distance(agents[i].pos * vec2f(size), vec2f(id.xy));
            let consume_radius = sqrt(agents[i].pop) * 2.4; 
            if (dist < consume_radius) {
                let falloff = 0.5 * (1.0 + cos((dist / consume_radius) * 3.14159)); 
                food *= clamp(1.0 - (falloff * agents[i].pop * 0.0004), 0.0, 1.0);
            }
        }
    }

    // [중요] 지역 유전자 업데이트: 한 프레임당 한 에이전트씩 해당 위치 유전자에 각인
    let active_agent_idx = time.frame % agent_count;
    let target_a = agents[active_agent_idx];
    if (target_a.alive > 0.5) {
        let dist_to_target = distance(target_a.pos * vec2f(size), vec2f(id.xy));
        if (dist_to_target < 2.0) { // 에이전트 위치 주변 픽셀에 각인
            gene_map = mix(gene_map, target_a.color.xy, 0.1); 
        }
    }

    // --- 5. 에이전트 물리 및 생태 (기존 로직 유지) ---
    if (id.x == 0 && id.y == 0) {
        for (var i = 0u; i < agent_count; i++) {
            var a = agents[i];
            if (time.frame == 0u) {
                if (i == 0u) { a.pos = vec2f(0.5, 0.5); a.color = vec3f(1.0, 0.8, 0.2); a.alive = 1.0; a.pop = 30.0; a.dir = 0.0; a.cooldown = 0.0; }
                else { a.alive = 0.0; }
            } else if (a.alive > 0.5) {
                let r_seed = pcg2d(vec2u(bitcast<u32>(a.pos.x * 1537.0), i) + vec2u(time.frame));
                let cur_h = sample_buffer(a.pos, size).x;
                let pref_h = 0.6;
                let cur_food = sample_buffer(a.pos, size).y * (1.0 - clamp(abs(cur_h - pref_h) * 1.5, 0.0, 0.9));
                
                // 물리 및 이동
                var move_vec = vec2f(cos(a.dir), sin(a.dir)) * 1.75; 
                for (var d = 0.0; d < 8.0; d += 1.0) {
                    let angle = d * (6.28318 / 8.0);
                    let s_pos = a.pos + vec2f(cos(angle), sin(angle)) * 0.025;
                    let s_data = sample_buffer(s_pos, size);
                    move_vec += vec2f(cos(angle), sin(angle)) * s_data.y;
                    move_vec -= vec2f(cos(angle), sin(angle)) * pow(abs(s_data.x - pref_h), 2.0) * 3.0;
                }

                a.dir = atan2(move_vec.y, move_vec.x) + (r_seed.x - 0.5) * 0.05;
                let speed = mix(0.00005, 0.0005, clamp(cur_h * 2.0, 0.1, 1.0)) * clamp(length(move_vec) * 1.5, 0.05, 2.5) * mix(1.0, 0.2, clamp(a.pop / 200.0, 0.0, 1.0)) *
                            mix(1.2, 0.4, clamp(abs(cur_h - pref_h) * 2.0, 0.0, 1.0));
                a.pos = fract(a.pos + vec2f(cos(a.dir), sin(a.dir)) * speed);

                // 성장, 돌연변이, 분열, 상호작용 등... (이전 로직 동일)
                let demand = a.pop * 0.001;
                if (cur_food > demand) { a.pop += (cur_food - demand) * 0.5; } else { a.pop -= (demand - cur_food) * 10.0 + 0.05; }
                a.cooldown = max(a.cooldown - 1.0, 0.0);
                if (r_seed.x < 0.1) {
                    let p_rand = (pcg3d(vec3u(i, time.frame, 88u)) - 0.5);
                    a.color = clamp(a.color + p_rand * p_rand * p_rand * 8.0 * (0.3 / (sqrt(a.pop) + 1.0)), vec3f(0.1), vec3f(1.0));
                }
                if (r_seed.y < clamp((a.pop - 100.0) * 0.01, 0.0, 0.5)) {
                    for (var j = 0u; j < 4u; j++) {
                        let c_idx = u32(pcg2d(vec2u(i, j + time.frame)).x * 128.0) % 128u;
                        if (agents[c_idx].alive < 0.5 && c_idx != i) {
                            a.pop *= 0.7; a.cooldown = 100.0;
                            agents[c_idx] = Agent(a.pos, a.color, 1.0, a.pop * 0.428, r_seed.x * 6.28, 100.0);
                            break;
                        }
                    }
                }
                let other_idx = u32(r_seed.y * 128.0) % 128u;
                if (other_idx != i && agents[other_idx].alive > 0.5 && distance(a.pos, agents[other_idx].pos) < 0.02) {
                    a.color = mix(a.color, agents[other_idx].color, 0.1);
                    if (a.cooldown <= 0.0 && agents[other_idx].cooldown <= 0.0) {
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

    // --- 6. 렌더링 (지형 오버레이) ---
    var final_col: vec3f;
    let local_gene_color = vec3f(gene_map, 0.5); // z, w를 R, G로 사용하고 B는 고정

    if (h < sea_level) {
        final_col = mix(vec3f(0.01, 0.03, 0.1), vec3f(0.1, 0.3, 0.5), h / sea_level) + vec3f(0, 0.15, 0.05) * food;
    } else {
        let land_h = (h - sea_level) / (1.0 - sea_level);
        var land_base = mix(vec3f(0.1, 0.2, 0.1), vec3f(0.35, 0.3, 0.2), smoothstep(0.2, 0.7, land_h));
        land_base = mix(land_base, vec3f(0.9, 0.9, 1.0), smoothstep(0.75, 0.95, land_h));
        
        // [복구] 유전자 지도 오버레이: 육지 부분에만 유전자 지도 색상을 20% 믹스
        land_base = mix(land_base, local_gene_color, 1);
        
        let slope = get_height(uv + vec2f(0.002, 0.002)) - h;
        final_col = (land_base + vec3f(0.05, 0.4, 0.1) * food * (1.0 - smoothstep(0.6, 0.8, land_h))) * clamp(1.0 + slope * 30.0, 0.6, 1.4);
    }

    // 에이전트 그리기
    let screen_pos = uv * vec2f(size);
    for (var i = 0u; i < 128u; i++) {
        if (agents[i].alive > 0.5) {
            let d = distance(agents[i].pos * vec2f(size), screen_pos);
            let r = sqrt(agents[i].pop) * 0.55 + 0.6; 
            if (d < r) {
                final_col = agents[i].color;
                if (d > r * 0.8) { final_col *= 0.2; }
                break;
            }
        }
    }
    
    food = min(food + 0.0005, select(0.05, 1.0, h >= sea_level));
    textureStore(pass_out, pos_i, 0, vec4f(h, food, gene_map.x, gene_map.y));
    textureStore(screen, pos_i, vec4f(final_col, 1.0));
}
