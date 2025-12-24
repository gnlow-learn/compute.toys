const THRESHOLD: f32 = 0.1; // 사용자 요청 수치

@compute @workgroup_size(16, 16)
fn main_compute(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<i32>(textureDimensions(screen));
    if (id.x >= u32(size.x) || id.y >= u32(size.y)) { return; }

    let pos = vec2<i32>(id.xy);
    let current_data = textureLoad(pass_in, pos, 0, 0);
    
    var feature = current_data.rg;
    var next_feature = feature;

    // 1. 초기화 (화면을 100% 색상으로 채움)
    if (time.elapsed < 0.2 || length(feature) == 0.0) {
        next_feature = vec2<f32>(hash(vec2<f32>(pos) * 1.2), hash(vec2<f32>(pos) * 3.4));
    } else {
        // 2. 주변 만족도 체크 (Torus 적용)
        var total_dist = 0.0;
        for (var y: i32 = -1; y <= 1; y++) {
            for (var x: i32 = -1; x <= 1; x++) {
                if (x == 0 && y == 0) { continue; }
                let sample_pos = (pos + vec2<i32>(x, y) + size) % size;
                let neighbor = textureLoad(pass_in, sample_pos, 0, 0);
                total_dist += distance(feature, neighbor.rg);
            }
        }
        let avg_dist = total_dist / 8.0;

        // 3. 스왑(Swap) 로직
        // 내가 불만족스럽다면 주변의 무작위 입자와 자리를 바꿉니다.
        if (avg_dist > THRESHOLD) {
            let h = hash(vec2<f32>(pos) + time.elapsed);
            // 8방향 중 무작위 선택
            let angle = floor(h * 8.0) * 0.785398;
            let offset = vec2<i32>(round(vec2<f32>(cos(angle), sin(angle))));
            let target_pos = (pos + offset + size) % size;
            
            let target_data = textureLoad(pass_in, target_pos, 0, 0);
            
            // 확률적으로 타겟의 특성을 내 것으로 가져옴 (스왑의 절반)
            // GPU 병렬성 때문에 완전한 1:1 교환은 아니지만, 
            // 통계적으로 모든 픽셀이 이 짓을 하면 색상 비중은 보존됩니다.
            if (hash(vec2<f32>(pos) * time.elapsed) > 0.5) {
                next_feature = target_data.rg;
            }
        }
    }

    // 데이터 저장 (B 채널은 사용 안 함, 꽉 찬 상태)
    textureStore(pass_out, pos, 0, vec4<f32>(next_feature.x, next_feature.y, 0.0, 1.0));
    
    // 화면 출력
    textureStore(screen, pos, vec4<f32>(next_feature.x, next_feature.y, 0.5, 1.0));
}

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}
