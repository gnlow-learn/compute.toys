const STEEPNESS: f32 = 1.0; 
const CENTER: f32 = 0.0;    

// 시간축 Reaction-Diffusion 파라미터 튜닝
const BORDER_THRESHOLD: f32 = 0.03; 
const GROWTH_RATE: f32 = 0.04;       // 선이 형성되는 속도
const INHIBITION_RATE: f32 = 0.01;   // 배경 청소 속도 (살짝 낮춰서 선의 잔상을 유지)
const TEMPORAL_DIFFUSION: f32 = 0.5; // 시간축 확산 강도 (0.0~1.0, 높을수록 선이 끈적하게 연결됨)
const SHARPNESS: f32 = 10.0;         

@compute @workgroup_size(16, 16)
fn main_compute(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<i32>(textureDimensions(screen));
    if (id.x >= u32(size.x) || id.y >= u32(size.y)) { return; }

    let pos = vec2<i32>(id.xy);
    let prev_data = textureLoad(pass_in, pos, 0, 0);
    
    var feature = prev_data.rg;
    var next_feature = feature;

    // 1. 시뮬레이션 업데이트 (유지)
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

    // 2. [공간 감지] + [시간축 확산 준비]
    var spatial_signal = 0.0;
    var neighbor_mem_avg = 0.0;
    let offsets = array<vec2<i32>, 4>(vec2<i32>(1,0), vec2<i32>(-1,0), vec2<i32>(0,1), vec2<i32>(0,-1));
    
    for (var i = 0; i < 4; i++) {
        let n_pos = (pos + offsets[i] + size) % size;
        let n_data = textureLoad(pass_in, n_pos, 0, 0);
        spatial_signal += distance(next_feature, n_data.rg); // 공간 신호 감지
        neighbor_mem_avg += n_data.b;                        // 주변 시간축 에너지 수집
    }
    spatial_signal /= 4.0;
    neighbor_mem_avg /= 4.0;

    // 3. [시간축 Reaction-Diffusion]
    // 현재 픽셀의 메모리와 주변 픽셀의 메모리를 섞어 '확산(Diffusion)'을 일으킵니다.
    var border_mem = mix(prev_data.b, neighbor_mem_avg, TEMPORAL_DIFFUSION);
    
    // 비선형 증폭 타겟
    let target_val = 1.0 / (1.0 + exp(-SHARPNESS * (spatial_signal - BORDER_THRESHOLD)));
    
    // 반응(Reaction) 로직
    if (target_val > 0.5) {
        border_mem += (target_val - border_mem) * GROWTH_RATE;
    } else {
        border_mem -= border_mem * INHIBITION_RATE;
    }
    border_mem = clamp(border_mem, 0.0, 1.0);

    // 4. 저장 및 출력
    textureStore(pass_out, pos, 0, vec4<f32>(next_feature.x, next_feature.y, border_mem, 1.0));

    let base_rgb = vec3<f32>(next_feature.x, next_feature.y, 0.6);
    let line_mask = smoothstep(0.4, 0.5, border_mem);
    let final_rgb = mix(base_rgb, vec3<f32>(0.0, 0.0, 0.0), line_mask);
    
    textureStore(screen, pos, vec4<f32>(final_rgb, 1.0));
}

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}
