const STEEPNESS: f32 = 1.0; 
const CENTER: f32 = 0.0;    
const BORDER_THRESHOLD: f32 = 0.12; 
const BORDER_SMOOTHNESS: f32 = 0.08;
const TEMPORAL_SMOOTHING: f32 = 0.95; // 높을수록 선이 더 끈적하고 부드럽게 변함

@compute @workgroup_size(16, 16)
fn main_compute(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<i32>(textureDimensions(screen));
    if (id.x >= u32(size.x) || id.y >= u32(size.y)) { return; }

    let pos = vec2<i32>(id.xy);
    
    // pass_in에서 이전 프레임 데이터를 읽어옵니다.
    // r, g: 색상 데이터 / b: 이전 프레임의 국경선 강도(시간축 메모리)
    let prev_data = textureLoad(pass_in, pos, 0, 0);
    
    var feature = prev_data.rg;
    var next_feature = feature;

    // 1. 업데이트 로직 (유지)
    if (time.elapsed < 0.2 || (feature.x == 0.0 && feature.y == 0.0)) {
        next_feature = vec2<f32>(hash(vec2<f32>(pos) * 1.2), hash(vec2<f32>(pos) * 3.4));
    } else {
        var total_dist = 0.0;
        for (var y: i32 = -1; y <= 1; y++) {
            for (var x: i32 = -1; x <= 1; x++) {
                if (x == 0 && y == 0) { continue; }
                let sample_pos = (pos + vec2<i32>(x, y) + size) % size;
                total_dist += distance(feature, textureLoad(pass_in, sample_pos, 0, 0).rg);
            }
        }
        let avg_dist = total_dist / 8.0;
        let move_probability = 1.0 / (1.0 + exp(-STEEPNESS * (avg_dist - CENTER)));
        
        if (hash(vec2<f32>(pos) + time.elapsed) < move_probability) {
            let rand_h = hash(vec2<f32>(pos) * (time.elapsed + 1.0));
            let angle = floor(rand_h * 8.0) * 0.785398;
            let offset = vec2<i32>(i32(round(cos(angle))), i32(round(sin(angle))));
            next_feature = textureLoad(pass_in, (pos + offset + size) % size, 0, 0).rg;
        }
    }

    // 2. 공간축 가우시안 (Spatial Smoothing)
    var spatial_diff = 0.0;
    for (var fy: i32 = -1; fy <= 1; fy++) {
        for (var fx: i32 = -1; fx <= 1; fx++) {
            let n_pos = (pos + vec2<i32>(fx, fy) + size) % size;
            spatial_diff += distance(next_feature, textureLoad(pass_in, n_pos, 0, 0).rg);
        }
    }
    let current_border = smoothstep(BORDER_THRESHOLD, BORDER_THRESHOLD + BORDER_SMOOTHNESS, spatial_diff / 9.0);

    // 3. 시간축 가우시안 (Temporal Smoothing)
    // 이전 프레임의 b 채널 값을 가져와 현재 국경선 강도와 섞습니다.
    let prev_border = prev_data.b;
    let smooth_border = mix(current_border, prev_border, TEMPORAL_SMOOTHING);

    // 4. 데이터 보존 및 출력
    // 다음 프레임을 위해 b 채널에 매끄러워진 국경선 정보를 저장합니다.
    textureStore(pass_out, pos, 0, vec4<f32>(next_feature.x, next_feature.y, smooth_border, 1.0));

    // 최종 화면 렌더링
    let base_rgb = vec3<f32>(next_feature.x, next_feature.y, 0.6);
    let final_rgb = mix(base_rgb, vec3<f32>(0.0), smooth_border);
    textureStore(screen, pos, vec4<f32>(final_rgb, 1.0));
}

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}
