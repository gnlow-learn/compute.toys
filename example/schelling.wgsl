const STEEPNESS: f32 = 1.0; // 값이 높을수록 Hard Threshold에 가까워짐
const CENTER: f32 = 0.0;    // 이 거리(차이)를 기준으로 이동 욕구가 급증함

@compute @workgroup_size(16, 16)
fn main_compute(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<i32>(textureDimensions(screen));
    if (id.x >= u32(size.x) || id.y >= u32(size.y)) { return; }

    let pos = vec2<i32>(id.xy);
    let current_data = textureLoad(pass_in, pos, 0, 0);
    
    var feature = current_data.rg;
    var next_feature = feature;

    if (time.elapsed < 0.2 || (feature.x == 0.0 && feature.y == 0.0)) {
        next_feature = vec2<f32>(hash(vec2<f32>(pos) * 1.2), hash(vec2<f32>(pos) * 3.4));
    } else {
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

        // --- Soft Threshold: Sigmoid Function ---
        // avg_dist가 CENTER보다 크면 확률이 0.5 이상으로 올라갑니다.
        let move_probability = 1.0 / (1.0 + exp(-STEEPNESS * (avg_dist - CENTER)));
        
        let h = hash(vec2<f32>(pos) + time.elapsed);
        
        if (h < move_probability) {
            let rand_h = hash(vec2<f32>(pos) * (time.elapsed + 1.0));
            let angle = floor(rand_h * 8.0) * 0.785398;
            let offset = vec2<i32>(round(vec2<f32>(cos(angle), sin(angle))));
            let target_pos = (pos + offset + size) % size;
            
            let target_data = textureLoad(pass_in, target_pos, 0, 0);
            next_feature = target_data.rg;
        }
    }

    textureStore(pass_out, pos, 0, vec4<f32>(next_feature.x, next_feature.y, 0.0, 1.0));
    textureStore(screen, pos, vec4<f32>(next_feature.x, next_feature.y, 0.6, 1.0));
}

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}
