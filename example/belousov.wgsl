@compute @workgroup_size(16, 16)
fn main_compute(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(screen);
    if (id.x >= size.x || id.y >= size.y) { return; }
    let pos = vec2<i32>(id.xy);

    // 1. 현재 나의 종 식별 (가장 강한 채널 찾기)
    let current_raw = textureLoad(pass_in, pos, 0, 0).rgb;
    var my_type = 0; // 0: 바위(R), 1: 보(G), 2: 가위(B)
    if (current_raw.g > current_raw.r && current_raw.g > current_raw.b) { my_type = 1; }
    if (current_raw.b > current_raw.r && current_raw.b > current_raw.g) { my_type = 2; }

    // 2. 주변 이웃 중 나를 잡아먹는 천적의 수 카운트
    var predator_count = 0;
    for (var y: i32 = -1; y <= 1; y++) {
        for (var x: i32 = -1; x <= 1; x++) {
            if (x == 0 && y == 0) { continue; }
            let neighbor = textureLoad(pass_in, pos + vec2<i32>(x, y), 0, 0).rgb;
            
            // 나를 이기는 종이 있는지 확인
            // 바위(0) < 보(1) < 가위(2) < 바위(0)
            let n_type = select(select(0, 1, neighbor.g > neighbor.r && neighbor.g > neighbor.b), 2, neighbor.b > neighbor.r && neighbor.b > neighbor.g);
            if ((my_type == 0 && n_type == 1) || 
                (my_type == 1 && n_type == 2) || 
                (my_type == 2 && n_type == 0)) {
                predator_count++;
            }
        }
    }

    // 3. 진화 규칙: 천적이 주변에 3명 이상이면 나도 그 종으로 변함
    var next_state = current_raw;
    if (predator_count >= 3) {
        if (my_type == 0) { next_state = vec3<f32>(0.0, 1.0, 0.0); } // 바위 -> 보
        else if (my_type == 1) { next_state = vec3<f32>(0.0, 0.0, 1.0); } // 보 -> 가위
        else if (my_type == 2) { next_state = vec3<f32>(1.0, 0.0, 0.0); } // 가위 -> 바위
    }

    // 4. 초기화 및 클릭
    if (time.elapsed < 0.2) {
        let h = hash(vec2<f32>(pos));
        if (h < 0.33) { next_state = vec3<f32>(1.0, 0.0, 0.0); }
        else if (h < 0.66) { next_state = vec3<f32>(0.0, 1.0, 0.0); }
        else { next_state = vec3<f32>(0.0, 0.0, 1.0); }
    }

    if (mouse.click > 0 && distance(vec2<f32>(pos), vec2<f32>(mouse.pos)) < 10.0) {
        next_state = vec3<f32>(1.0, 1.0, 1.0); // 마우스는 중립 상태(흰색) 주입
    }

    textureStore(pass_out, pos, 0, vec4<f32>(next_state, 1.0));
    textureStore(screen, pos, vec4<f32>(next_state, 1.0));
}

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}
