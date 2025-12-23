// ---------------------------------------------------------
// 1. 전역 상수 및 유틸리티 (에러 수정: let -> const)
// ---------------------------------------------------------

const MAX_FIRMS: i32 = 64;

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

fn hsv2rgb(c: vec3<f32>) -> vec3<f32> {
    let k = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let p = abs(fract(c.xxx + k.xyz) * 6.0 - k.www);
    return c.z * mix(k.xxx, clamp(p - k.xxx, vec3(0.0), vec3(1.0)), c.y);
}

fn get_distinct_color(id: i32) -> vec3<f32> {
    let golden_ratio_conjugate = 0.618033988749895;
    var h = (f32(id) * golden_ratio_conjugate) % 1.0;
    return hsv2rgb(vec3(h, 0.75, 0.9));
}

// 데이터 접근 함수
fn get_firm_main(id: i32) -> vec4<f32> {
    return textureLoad(pass_in, vec2<i32>(id, 0), 0, 0);
}

fn get_firm_ext(id: i32) -> vec4<f32> {
    return textureLoad(pass_in, vec2<i32>(id + MAX_FIRMS, 0), 0, 0);
}

fn levy_step(seed: vec2<f32>, frame: f32) -> vec2<f32> {
    let r1 = max(hash(seed + frame), 0.0001);
    let r2 = hash(seed + frame + 0.5);
    let alpha = 1.35; 
    let length = pow(r1, -1.0 / alpha) * 0.004;
    let angle = r2 * 6.283185;
    return vec2<f32>(cos(angle), sin(angle)) * length;
}

// ---------------------------------------------------------
// 2. 경제 로직 (순서 중요: update 위에 배치)
// ---------------------------------------------------------

fn calc_total_profit(test_pos: vec2<f32>, test_price: f32, firm_id: i32) -> f32 {
    var revenue = 0.0;
    let budget = 0.95;
    let sensitivity = 11.0;
    let transport = 2.0;

    for (var y = 0.1; y < 1.0; y += 0.2) {
        for (var x = 0.1; x < 1.0; x += 0.2) {
            let p = vec2<f32>(x, y);
            let my_cost = test_price + distance(p, test_pos) * transport; 
            var min_other_cost = 1e10;
            for (var i = 0; i < MAX_FIRMS; i++) {
                if (i == firm_id) { continue; }
                let other = get_firm_main(i);
                if (other.w <= 0.0) { continue; }
                let other_cost = other.z + distance(p, other.xy) * transport;
                min_other_cost = min(min_other_cost, other_cost);
            }
            if (my_cost < min_other_cost) {
                let prob = 1.0 / (1.0 + exp(sensitivity * (my_cost - budget)));
                revenue += prob;
            }
        }
    }
    return test_price * revenue;
}

// ---------------------------------------------------------
// 3. 메인 업데이트 (데이터 확장 적용)
// ---------------------------------------------------------

@compute @workgroup_size(64, 1)
fn main_update(@builtin(global_invocation_id) id: vec3<u32>) {
    let firm_id = i32(id.x);
    if (firm_id >= MAX_FIRMS) { return; }

    var main_data = get_firm_main(firm_id);
    var ext_data = get_firm_ext(firm_id);
    let current_frame = f32(time.frame);

    if (time.frame < 1) {
        if (firm_id < 8) {
            main_data = vec4<f32>(hash(vec2(f32(firm_id), 0.1)), hash(vec2(f32(firm_id), 0.2)), 0.5, 1.2);
            ext_data = vec4<f32>(current_frame, 0.0, 0.0, 0.0);
        }
    } else {
        if (main_data.w <= 0.0) { // 폐업 상태
    // 창업 확률 판정
    let spawn_prob = hash(vec2(f32(firm_id), time.elapsed));
    if (spawn_prob < 0.003) {
        // [개선] 더 복잡한 시드를 사용하여 화면 전체에 골고루 뿌려지게 함
        let seed_x = vec2(time.elapsed * 1.1, f32(firm_id) * 0.55);
        let seed_y = vec2(time.elapsed * 2.2, f32(firm_id) * 0.88);
        
        let new_x = hash(seed_x);
        let new_y = hash(seed_y);
        
        // 초기 자본 1.0, 초기 가격 0.5, 창업 프레임 기록
        main_data = vec4<f32>(new_x, new_y, 0.5, 1.0);
        ext_data = vec4<f32>(current_frame, 0.0, 0.0, 0.0);
    }
} else {
            let rev = calc_total_profit(main_data.xy, main_data.z, firm_id);
            var fixed_cost = 0.15;
            
            let screen_size = vec2<f32>(textureDimensions(screen));
            let norm_mouse = vec2<f32>(mouse.pos) / screen_size;
            if (mouse.click > 0 && distance(norm_mouse, main_data.xy) < 0.1) { fixed_cost += 0.8; }

            main_data.w += (rev * 0.25) - fixed_cost;

            if (main_data.w <= 0.0) {
                main_data = vec4<f32>(0.0);
                ext_data = vec4<f32>(0.0);
            } else {
                let step = levy_step(main_data.xy, current_frame);
                let test_pos = clamp(main_data.xy + step, vec2(0.02), vec2(0.98));
                let test_price = clamp(main_data.z + (hash(main_data.xy + current_frame) - 0.5) * 0.1, 0.2, 1.5);
                
                if (calc_total_profit(test_pos, test_price, firm_id) > rev) {
                    main_data = vec4<f32>(test_pos, test_price, main_data.w);
                } else {
                    main_data = vec4<f32>(main_data.xy, main_data.z, main_data.w);
                }
            }
        }
    }
    textureStore(pass_out, vec2<i32>(firm_id, 0), 0, main_data);
    textureStore(pass_out, vec2<i32>(firm_id + MAX_FIRMS, 0), 0, ext_data);
}
// ---------------------------------------------------------
// 시각화: 자본=크기, 생존시간=기업색상 연동 시계
// ---------------------------------------------------------

@compute @workgroup_size(16, 16)
fn main_compute(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(screen);
    if (id.y == 0u) { textureStore(screen, id.xy, vec4(0.0)); return; }
    
    let f_pos = vec2<f32>(id.xy) / vec2<f32>(size);
    let screen_size = vec2<f32>(size);
    var min_cost = 1e10;
    var owner = -1;

    // 1. 소비자 영역 계산
    for (var i = 0; i < MAX_FIRMS; i++) {
        let firm = get_firm_main(i);
        if (firm.w <= 0.0) { continue; }
        let cost = firm.z + distance(f_pos, firm.xy) * 2.2;
        if (cost < min_cost) { min_cost = cost; owner = i; }
    }

    var col = vec3(0.01);
    if (owner != -1) {
        let base_col = get_distinct_color(owner);
        col = base_col * (1.0 / (1.0 + exp(12.0 * (min_cost - 0.95)))) * 0.45;
    }

    // 2. 기업 상점 및 생존 시계 렌더링
    for (var i = 0; i < MAX_FIRMS; i++) {
        let main = get_firm_main(i);
        let ext = get_firm_ext(i);
        if (main.w <= 0.0) { continue; }
        
        let center = main.xy * screen_size;
        let dist = distance(f_pos * screen_size, center);
        
        // [변경] 로그 스케일 적용
        // log(main.w + 1.0)을 쓰는 이유: 자본이 0일 때 log(1)=0이 되어 음수를 방지함
        // 15.0은 기본 스케일 계수, 5.0은 최소 반지름입니다.
        let radius = log(main.w + 1.0) * 15.0 + 5.0; 
        
        if (dist < radius) {
            let firm_col = get_distinct_color(i);
            let age = f32(time.frame) - ext.x;
            let age_ratio = (age % 3000.0) / 3000.0;
            
            let dir = (f_pos * screen_size) - center;
            let angle = (atan2(dir.x, -dir.y) / 6.28318) + 0.5;
            
            if (angle < age_ratio) {
                col = mix(firm_col, vec3(1.0), 0.7);
            } else {
                col = firm_col * 0.25;
            }
            
            let border_width = 1.0 + (main.z * 1.5);
            if (dist > radius - border_width) {
                col = vec3(1.0); 
            }
        }
    }
    
    let norm_mouse = vec2<f32>(mouse.pos) / screen_size;
    if (mouse.click > 0 && distance(f_pos, norm_mouse) < 0.1) { col += vec3(0.2, 0.0, 0.0); }
    textureStore(screen, id.xy, vec4(col, 1.0));
}
