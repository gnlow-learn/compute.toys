const STEEPNESS: f32 = 1.0; 
const CENTER: f32 = 0.0;    

// --- 사용자 튜닝 파라미터 (철저히 유지) ---
const BORDER_THRESHOLD: f32 = 0.03; 
const SHARPNESS: f32 = 10.0;        
const TEMPORAL_DIFFUSION: f32 = 0.5; 

// --- 시간 민감도 통합 제어 ---
const TIME_SENSITIVITY: f32 = 0.5; 
const GROWTH_RATE: f32 = 0.04 * TIME_SENSITIVITY;
const INHIBITION_RATE: f32 = 0.01 * TIME_SENSITIVITY;

// --- [시각화 전용] 선 압축 강도 ---
// 1.0: 원본 두께 / 5.0 이상: 매우 예리한 실선
// (기록되는 데이터에는 영향을 주지 않음)
const LINE_COMPRESSION: f32 = 4.5; 

@compute @workgroup_size(16, 16)
fn main_compute(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<i32>(textureDimensions(screen));
    if (id.x >= u32(size.x) || id.y >= u32(size.y)) { return; }

    let pos = vec2<i32>(id.xy);
    let prev_data = textureLoad(pass_in, pos, 0, 0);
    
    var feature = prev_data.rg;
    var next_feature = feature;

    // 1. 시뮬레이션 업데이트 (사용자 로직)
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

    // 2. 공간 신호 감지 및 시간축 에너지 확산
    var spatial_signal = 0.0;
    var neighbor_mem_avg = 0.0;
    let offsets = array<vec2<i32>, 4>(vec2<i32>(1,0), vec2<i32>(-1,0), vec2<i32>(0,1), vec2<i32>(0,-1));
    for (var i = 0; i < 4; i++) {
        let n_pos = (pos + offsets[i] + size) % size;
        let n_data = textureLoad(pass_in, n_pos, 0, 0);
        spatial_signal += distance(next_feature, n_data.rg);
        neighbor_mem_avg += n_data.b;
    }
    spatial_signal /= 4.0;
    neighbor_mem_avg /= 4.0;

    // 3. 시간축 Reaction-Diffusion (데이터 기록용)
    var border_mem = mix(max(0.0, prev_data.b), max(0.0, neighbor_mem_avg), TEMPORAL_DIFFUSION);
    let target_val = 1.0 / (1.0 + exp(-SHARPNESS * (spatial_signal - BORDER_THRESHOLD)));
    
    if (target_val > 0.5) {
        border_mem += (target_val - border_mem) * GROWTH_RATE;
    } else {
        border_mem -= border_mem * INHIBITION_RATE;
    }
    border_mem = clamp(border_mem, 0.0, 1.0);

    // [중요] 다음 프레임 시뮬레이션을 위해 원본 border_mem 저장
    textureStore(pass_out, pos, 0, vec4<f32>(next_feature.x, next_feature.y, border_mem, 1.0));

    // 4. 비선형 압축 시각화 (화면 출력용)
    // 에너지를 거듭제곱하여 분포의 끝자락을 수축시킵니다.
    let compressed_energy = pow(max(0.0, border_mem), LINE_COMPRESSION);

    let base_rgb = vec3<f32>(next_feature.x, next_feature.y, 0.6);
    
    // 에너지가 아주 조금이라도 있으면 그리기 위해 문턱값을 대폭 낮춤
    let line_mask = smoothstep(0.01, 0.1, compressed_energy);
    let final_rgb = mix(base_rgb, vec3<f32>(0.0), line_mask);
    
    textureStore(screen, pos, vec4<f32>(final_rgb, 1.0));
}

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}
