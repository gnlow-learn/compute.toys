const THRESHOLD: f32 = 0.45; 
const EMPTY: f32 = 0.0;
const AGENT_A: f32 = 1.0; 
const AGENT_B: f32 = 2.0;

@compute @workgroup_size(16, 16)
fn main_compute(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(screen);
    if (id.x >= size.x || id.y >= size.y) { return; }

    let pos = vec2<i32>(id.xy);
    let current_data = textureLoad(pass_in, pos, 0, 0);
    
    var agent_type = current_data.r; 
    var next_agent_type = agent_type;

    // 1. 초기화
    if (time.elapsed < 0.1) {
        let h = hash(vec2<f32>(pos));
        if (h < 0.25) { 
            next_agent_type = EMPTY; 
        } else if (h < 0.62) { 
            next_agent_type = AGENT_A; 
        } else { 
            next_agent_type = AGENT_B; 
        }
    } else {
        // 2. 에이전트 로직 (Radius 2)
        if (agent_type > 0.5) {
            var same_count = 0.0;
            var total_neighbors = 0.0;

            for (var y: i32 = -2; y <= 2; y++) {
                for (var x: i32 = -2; x <= 2; x++) {
                    if (x == 0 && y == 0) { continue; }
                    
                    let neighbor = textureLoad(pass_in, pos + vec2<i32>(x, y), 0, 0).r;
                    if (neighbor > 0.5) {
                        total_neighbors += 1.0;
                        if (abs(neighbor - agent_type) < 0.1) {
                            same_count += 1.0;
                        }
                    }
                }
            }

            let satisfaction = select(same_count / total_neighbors, 1.0, total_neighbors == 0.0);

            if (satisfaction < THRESHOLD) {
                if (hash(vec2<f32>(pos) + time.elapsed) > 0.85) { // 이주 확률 소폭 상향
                    next_agent_type = EMPTY;
                }
            }
        } else {
            // 3. 빈 공간에 이사 오는 로직
            let h = hash(vec2<f32>(pos) + time.elapsed);
            if (h > 0.98) {
                next_agent_type = select(AGENT_A, AGENT_B, hash(vec2<f32>(pos) * 1.5) > 0.5);
            }
        }
    }

    if (mouse.click > 0 && distance(vec2<f32>(pos), vec2<f32>(mouse.pos)) < 15.0) {
        next_agent_type = EMPTY;
    }

    textureStore(pass_out, pos, 0, vec4<f32>(next_agent_type, 0.0, 0.0, 1.0));
    
    // 4. 출력 색상 설정
    var color = vec3<f32>(1.0, 1.0, 1.0); // 기본값: 흰색 (EMPTY)
    
    if (next_agent_type == AGENT_A) { 
        color = vec4<f32>(1.0, 0.2, 0.2, 1.0).rgb; // 진한 빨강
    } else if (next_agent_type == AGENT_B) { 
        color = vec4<f32>(0.2, 0.4, 1.0, 1.0).rgb; // 진한 파랑
    }
    
    textureStore(screen, pos, vec4<f32>(color, 1.0));
}

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}
