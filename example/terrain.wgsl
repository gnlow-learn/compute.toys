// --- 지형 생성을 위한 유틸리티 함수 ---

fn hash(p: vec2f) -> f32 {
    let p3 = fract(vec3f(p.xyx) * 0.1031);
    let p3_2 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3_2.x + p3_2.y) * p3_2.z);
}

fn noise_tiled(p: vec2f, scale: f32) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let i00 = (i + vec2f(0.0, 0.0)) % scale;
    let i10 = (i + vec2f(1.0, 0.0)) % scale;
    let i01 = (i + vec2f(0.0, 1.0)) % scale;
    let i11 = (i + vec2f(1.0, 1.0)) % scale;
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i00), hash(i10), u.x),
               mix(hash(i01), hash(i11), u.x), u.y);
}

fn fbm_tiled(p: vec2f, scale: f32) -> f32 {
    var v = 0.0;
    var a = 0.5;
    var pos = p;
    var s = scale;
    for (var i = 0; i < 6; i++) {
        v += a * noise_tiled(pos, s);
        pos *= 2.0;
        s *= 2.0;
        a *= 0.5;
    }
    return v;
}

@compute @workgroup_size(16, 16)
fn main_compute(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(screen);
    if (id.x >= size.x || id.y >= size.y) { return; }

    let pos = vec2i(id.xy);
    let uv = vec2f(id.xy) / vec2f(size);
    
    let world_scale = 4.0; 
    let world_pos = uv * world_scale;
    
    // 1. 기본 지형 생성 (바다 모양 결정)
    let h_base_raw = fbm_tiled(world_pos, world_scale);
    let h_base = pow(h_base_raw, 1.5); 

    var h = h_base;
    let sea_level = 0.5;

    // 2. 육지 높이 강화 (산맥을 더 웅장하게)
    if (h_base >= sea_level) {
        let n_h = (h_base - sea_level) / (1.0 - sea_level);
        // smoothstep으로 중간 고도를 확 끌어올려 산맥 면적 확보
        let steep_h = smoothstep(-0.1, 0.9, n_h); 
        h = sea_level + steep_h * (1.0 - sea_level);
    }

    // 3. 변수 선언 (스코프 문제 해결을 위해 밖에서 선언)
    var color: vec3f;
    var terrain_type: f32 = 0.0; 

    // 4. 색상 판정 및 지형 타입 할당
    if (h < sea_level) {
        // 바다
        color = mix(vec3f(0.05, 0.1, 0.3), vec3f(0.1, 0.2, 0.5), h / sea_level);
        terrain_type = 0.0;
    } else {
        // 육지: 0.5~0.6 평지 / 0.6~0.85 산맥 / 0.85~1.0 만년설
        if (h < 0.6) {
            color = vec3f(0.15, 0.45, 0.15); // 평지
            terrain_type = 1.0;
        } else if (h < 0.85) {
            let t = (h - 0.6) / 0.25;
            color = mix(vec3f(0.15, 0.45, 0.15), vec3f(0.4, 0.3, 0.2), t); // 산맥
            terrain_type = 2.0;
        } else {
            let t = (h - 0.85) / 0.15;
            color = mix(vec3f(0.4, 0.3, 0.2), vec3f(0.9, 0.9, 1.0), t); // 만년설
            terrain_type = 3.0;
        }
    }

    // 최종 저장 (h는 아날로그 높이값, terrain_type은 정수형 카테고리)
    textureStore(pass_out, pos, 0, vec4f(h, terrain_type, 0.0, 1.0));
    textureStore(screen, pos, vec4f(color, 1.0));
}
