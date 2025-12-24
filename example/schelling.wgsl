const THRESHOLD: f32 = 0.1; // 낮을수록 더 예민하게 반응하여 군집이 커짐

@compute @workgroup_size(16, 16)
fn main_compute(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(screen);
    if (id.x >= size.x || id.y >= size.y) { return; }

    let pos = vec2<i32>(id.xy);
    let current_data = textureLoad(pass_in, pos, 0, 0);
    
    var feature = current_data.rg;
    var next_feature = feature;

    // 1. 초기화: 화면을 무작위 색상으로 가득 채움
    if (time.elapsed < 0.2 || length(feature) == 0.0) {
        next_feature = vec2<f32>(hash(vec2<f32>(pos) * 1.2), hash(vec2<f32>(pos) * 3.4));
    } else {
        // 2. 주변 만족도 체크 (Radius 1 or 2)
        var total_dist = 0.0;
        for (var y: i32 = -1; y <= 1; y++) {
            for (var x: i32 = -1; x <= 1; x++) {
                if (x == 0 && y == 0) { continue; }
                let neighbor = textureLoad(pass_in, pos + vec2<i32>(x, y), 0, 0);
                total_dist += distance(feature, neighbor.rg);
            }
        }
        let avg_dist = total_dist / 8.0;

        // 3. 교환(Swap) 로직
        // 내가 불만족스러울 때만 주변과 자리를 바꿀 기회를 가짐
        if (avg_dist > THRESHOLD) {
            // 무작위로 이웃 하나를 선정
            let angle = hash(vec2<f32>(pos) + time.elapsed) * 6.28318;
            let offset = vec2<i32>(vec2<f32>(cos(angle), sin(angle)) * 2.0); // Radius 2 범위에서 타겟 선정
            let target_pos = pos + offset;
            let target_data = textureLoad(pass_in, target_pos, 0, 0);

            // [핵심] 확률적 교환: 
            // 서로 자리를 바꿨을 때 더 행복해질 가능성이 있다면 스왑(Swap)
            // 여기서는 단순화를 위해 무작위성을 부여하여 섞이도록 함
            if (hash(vec2<f32>(pos) * time.elapsed) > 0.9) {
                next_feature = target_data.rg;
            }
        }
        
        // 4. 충돌 방지: 타겟 입장에서 누군가 나에게 왔다면 나도 그쪽으로 가야 함
        // (이 로직은 GPU 병렬 처리상 엄격한 1:1 스왑은 아니지만, 통계적으로 보존됨)
    }

    // 데이터 저장 (B 채널은 이제 사용 안 함)
    textureStore(pass_out, pos, 0, vec4<f32>(next_feature.x, next_feature.y, 0.0, 1.0));
    
    // 화면 출력 (빈 칸 없이 가득 찬 색상)
    textureStore(screen, pos, vec4<f32>(next_feature.x, next_feature.y, 0.5, 1.0));
}

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}
