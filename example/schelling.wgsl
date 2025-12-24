const THRESHOLD: f32 = 0.1; // 매우 예민한 설정
const EMPTY: f32 = 0.0;
const OCCUPIED: f32 = 1.0;

@compute @workgroup_size(16, 16)
fn main_compute(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<i32>(textureDimensions(screen));
    if (id.x >= u32(size.x) || id.y >= u32(size.y)) { return; }

    let pos = vec2<i32>(id.xy);
    let current_data = textureLoad(pass_in, pos, 0, 0);
    
    var feature = current_data.rg;
    var is_active = current_data.b;

    var next_feature = feature;
    var next_active = is_active;

    // 1. 초기화 (1%의 빈 공간만 남겨서 유동성 확보)
    if (time.elapsed < 0.2 || (is_active == 0.0 && length(feature) == 0.0)) {
        let h = hash(vec2<f32>(pos));
        if (h < 0.01) { 
            next_active = EMPTY;
            next_feature = vec2<f32>(0.0);
        } else {
            next_active = OCCUPIED;
            next_feature = vec2<f32>(hash(vec2<f32>(pos) * 1.2), hash(vec2<f32>(pos) * 3.4));
        }
    } else {
        // 2. 토러스 좌표 계산 함수 (클로저 대신 직접 계산)
        // pos + offset을 size로 나누어 떨어지게 처리
        
        if (is_active > 0.5) {
            var total_dist = 0.0;
            var count = 0.0;
            
            // Radius 2 탐색
            for (var y: i32 = -2; y <= 2; y++) {
                for (var x: i32 = -2; x <= 2; x++) {
                    if (x == 0 && y == 0) { continue; }
                    
                    // 토러스 좌표 적용: (pos + offset + size) % size
                    let sample_pos = (pos + vec2<i32>(x, y) + size) % size;
                    let neighbor = textureLoad(pass_in, sample_pos, 0, 0);
                    
                    if (neighbor.b > 0.5) {
                        total_dist += distance(feature, neighbor.rg);
                        count += 1.0;
                    }
                }
            }
            let avg_dist = select(total_dist / count, 0.0, count == 0.0);

            // 3. 불만족 시 이동 (빈칸 찾기)
            if (avg_dist > THRESHOLD) {
                var can_move = false;
                for (var y: i32 = -1; y <= 1; y++) {
                    for (var x: i32 = -1; x <= 1; x++) {
                        let sample_pos = (pos + vec2<i32>(x, y) + size) % size;
                        if (textureLoad(pass_in, sample_pos, 0, 0).b < 0.5) { 
                            can_move = true; 
                            break;
                        }
                    }
                    if (can_move) { break; }
                }
                
                if (can_move && hash(vec2<f32>(pos) + time.elapsed) > 0.8) {
                    next_active = EMPTY;
                }
            }
        } else {
            // 빈 공간: 토러스 이웃 중 불만족한 개체 탐색
            for (var y: i32 = -1; y <= 1; y++) {
                for (var x: i32 = -1; x <= 1; x++) {
                    let n_pos = (pos + vec2<i32>(x, y) + size) % size;
                    let n_data = textureLoad(pass_in, n_pos, 0, 0);
                    
                    if (n_data.b > 0.5) {
                        if (hash(vec2<f32>(n_pos) + time.elapsed) > 0.8) {
                             next_active = OCCUPIED;
                             next_feature = n_data.rg;
                             break;
                        }
                    }
                }
                if (next_active > 0.5) { break; }
            }
        }
    }

    // 데이터 저장
    textureStore(pass_out, pos, 0, vec4<f32>(next_feature.x, next_feature.y, next_active, 1.0));
    
    // 출력
    var display_color = vec3<f32>(1.0);
    if (next_active > 0.5) {
        display_color = vec3<f32>(next_feature.x, next_feature.y, 0.6);
    }
    textureStore(screen, pos, vec4<f32>(display_color, 1.0));
}

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}
