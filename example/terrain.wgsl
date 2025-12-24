// --- 지형 생성을 위한 유틸리티 함수 ---

// 1. PCG 기반 해시 (직선 아티팩트 방지 및 타일링 안정성)
fn hash(p: vec2f) -> f32 {
    let p3 = fract(vec3f(p.xyx) * 0.1031);
    let p3_2 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3_2.x + p3_2.y) * p3_2.z);
}

// 2. 타일링 가능한 노이즈 (Value Noise with Wrapping)
fn noise_tiled(p: vec2f, scale: f32) -> f32 {
    let i = floor(p);
    let f = fract(p);
    
    // 경계면에서 인덱스를 0으로 되돌려(Wrap-around) 끊김 방지
    let i00 = (i + vec2f(0.0, 0.0)) % scale;
    let i10 = (i + vec2f(1.0, 0.0)) % scale;
    let i01 = (i + vec2f(0.0, 1.0)) % scale;
    let i11 = (i + vec2f(1.0, 1.0)) % scale;

    // Smoothstep interpolation
    let u = f * f * (3.0 - 2.0 * f);

    return mix(mix(hash(i00), hash(i10), u.x),
               mix(hash(i01), hash(i11), u.x), u.y);
}

// 3. Tiled FBM (Fractional Brownian Motion)
fn fbm_tiled(p: vec2f, scale: f32) -> f32 {
    var v = 0.0;
    var a = 0.5;
    var pos = p;
    var s = scale;
    for (var i = 0; i < 6; i++) {
        v += a * noise_tiled(pos, s);
        pos *= 2.0;
        s *= 2.0; // 옥타브가 올라갈수록 타일링 주기도 함께 증가
        a *= 0.5;
    }
    return v;
}

// --- 메인 컴퓨트 커널 ---

@compute @workgroup_size(16, 16)
fn main_compute(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(screen);
    if (id.x >= size.x || id.y >= size.y) { return; }

    let pos = vec2i(id.xy);
    let uv = vec2f(id.xy) / vec2f(size);
    
    // 월드 스케일: 정수여야 함 (4.0 = 지도가 가로세로 4칸 주기로 반복)
    let world_scale = 4.0;
    let world_pos = uv * world_scale;
    
    // 지형 높이 계산 및 감쇠 처리
    let h_raw = fbm_tiled(world_pos, world_scale);
    let h = pow(h_raw, 1.6);

    // --- 지형 판정 및 색상 설정 (해변 제거 버전) ---
    var color: vec3f;
    var terrain_type: f32 = 0.0; 

    if (h < 0.5) {
        // 바다 (깊이에 따라 미세한 색상 차이만 부여)
        color = mix(vec3f(0.05, 0.1, 0.3), vec3f(0.1, 0.2, 0.5), h * 2.0);
        terrain_type = 0.0;
    } else if (h < 0.75) {
        // 평지 (이동 및 거주 적합지)
        color = vec3f(0.15, 0.45, 0.15);
        terrain_type = 1.0;
    } else if (h < 0.88) {
        // 산맥 (장벽 역할)
        color = vec3f(0.4, 0.3, 0.2);
        terrain_type = 2.0;
    } else {
        // 고산지대
        color = vec3f(0.9, 0.9, 1.0);
        terrain_type = 3.0;
    }

    // 마우스 클릭 시 시각적 피드백
    if (mouse.click > 0 && distance(vec2f(pos), vec2f(mouse.pos)) < 15.0) {
        color = vec3f(1.0, 0.0, 0.0);
    }

    // 데이터 보존 및 화면 출력
    textureStore(pass_out, pos, 0, vec4f(h, terrain_type, 0.0, 1.0));
    textureStore(screen, pos, vec4f(color, 1.0));
}
