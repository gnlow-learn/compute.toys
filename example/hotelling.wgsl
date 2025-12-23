// ---------------------------------------------------------
// 1. 전역 상수 및 유틸리티
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
    let f_id_val = f32(id);
    let h = (f_id_val * 0.618033988749895) % 1.0;
    let s = mix(0.4, 0.9, hash(vec2(f_id_val, 0.123)));
    let v = mix(0.6, 1.0, hash(vec2(f_id_val, 0.456)));
    return hsv2rgb(vec3(h, s, v));
}

fn get_firm_main(id: i32) -> vec4<f32> { return textureLoad(pass_in, vec2<i32>(id, 0), 0, 0); }
fn get_firm_ext(id: i32) -> vec4<f32> { return textureLoad(pass_in, vec2<i32>(id + MAX_FIRMS, 0), 0, 0); }

fn get_land_value(pos: vec2<f32>) -> f32 {
    let center_dist = distance(pos, vec2<f32>(0.5, 0.5));
    return exp(-1.5 * center_dist);
}

// ---------------------------------------------------------
// 2. 경제 로직
// ---------------------------------------------------------
fn calc_total_profit(test_pos: vec2<f32>, test_price: f32, firm_id: i32) -> f32 {
    var revenue = 0.0;
    let transport = 2.2;
    for (var y = 0.1; y < 1.0; y += 0.2) {
        for (var x = 0.1; x < 1.0; x += 0.2) {
            let p = vec2<f32>(x, y);
            let my_cost = test_price + distance(p, test_pos) * transport; 
            var min_other_cost = 1e10;
            for (var i = 0; i < MAX_FIRMS; i++) {
                if (i == firm_id) { continue; }
                let other = get_firm_main(i);
                if (other.w <= 0.0) { continue; }
                min_other_cost = min(min_other_cost, other.z + distance(p, other.xy) * transport);
            }
            if (my_cost < min_other_cost) {
                revenue += 1.0 / (1.0 + exp(11.0 * (my_cost - 0.95)));
            }
        }
    }
    return test_price * revenue;
}

// ---------------------------------------------------------
// 3. 메인 업데이트 (수익성 기반 지가 시스템)
// ---------------------------------------------------------
@compute @workgroup_size(64, 1)
fn main_update(@builtin(global_invocation_id) id: vec3<u32>) {
    let firm_id = i32(id.x);
    if (firm_id >= MAX_FIRMS) { return; }

    var main_data = get_firm_main(firm_id);
    var ext_data = get_firm_ext(firm_id);
    let current_frame = f32(time.frame);
    let f_id = f32(firm_id); // 에러 해결: f_id 선언

    if (time.frame < 1) {
        if (firm_id < 12) {
            let start_pos = vec2(hash(vec2(f_id, 0.1)), hash(vec2(f_id, 0.2)));
            main_data = vec4<f32>(start_pos, 0.5, 1.0);
            ext_data = vec4<f32>(current_frame, 0.0, 0.0, 0.0);
        }
    } else {
        if (main_data.w <= 0.0) {
            if (hash(vec2(f_id, time.elapsed)) < 0.002) {
                let new_x = hash(vec2(time.elapsed * 1.1, f_id * 0.55));
                let new_y = hash(vec2(time.elapsed * 2.2, f_id * 0.88));
                main_data = vec4<f32>(new_x, new_y, 0.5, 1.0);
                ext_data = vec4<f32>(current_frame, 0.0, 0.0, 0.0);
            }
        } else {
            let land_v = get_land_value(main_data.xy);
            let rev = calc_total_profit(main_data.xy, main_data.z, firm_id);
            let boosted_rev = rev * (1.0 + sqrt(land_v) * 0.4); 
            let fixed_cost = 0.04 + (land_v * land_v * 0.25); 
            
            let current_net_profit = (boosted_rev * 0.08) - fixed_cost;
            main_data.w += current_net_profit;

            if (main_data.w <= 0.0) {
                main_data = vec4<f32>(0.0);
                ext_data = vec4<f32>(0.0);
            } else {
                let r1 = max(hash(main_data.xy + current_frame), 0.0001);
                let r2 = hash(main_data.xy + current_frame + 0.5);
                let step = vec2(cos(r2 * 6.28), sin(r2 * 6.28)) * pow(r1, -1.0/1.35) * 0.0015;
                let test_pos = clamp(main_data.xy + step, vec2(0.02), vec2(0.98));
                
                let test_lv = get_land_value(test_pos);
                let test_rev = calc_total_profit(test_pos, main_data.z, firm_id);
                let test_net_profit = (test_rev * (1.0 + sqrt(test_lv) * 0.4) * 0.08) - (0.04 + (test_lv * test_lv * 0.25));

                let random_move = hash(vec2(f_id, time.elapsed)) < 0.005;
                if (test_net_profit > current_net_profit || random_move) {
                    main_data.x = test_pos.x;
                    main_data.y = test_pos.y;
                    main_data.z = clamp(main_data.z + (hash(main_data.xy + current_frame) - 0.5) * 0.02, 0.2, 1.5);
                }
            }
        }
    }
    textureStore(pass_out, vec2<i32>(firm_id, 0), 0, main_data);
    textureStore(pass_out, vec2<i32>(firm_id + MAX_FIRMS, 0), 0, ext_data);
}

// ---------------------------------------------------------
// 4. 시각화
// ---------------------------------------------------------
@compute @workgroup_size(16, 16)
fn main_compute(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(screen);
    if (id.y == 0u) { textureStore(screen, id.xy, vec4(0.0)); return; }
    let f_pos = vec2<f32>(id.xy) / vec2<f32>(size);
    let screen_size = vec2<f32>(size);
    
    let lv = get_land_value(f_pos);
    var final_col = vec3(0.01 + lv * 0.02, 0.01 + lv * 0.01, 0.02 + lv * 0.02);

    var min_cost = 100.0;
    var costs: array<f32, 64>;
    var is_active: array<bool, 64>;

    for (var i = 0; i < MAX_FIRMS; i++) {
        let f = get_firm_main(i);
        is_active[i] = f.w > 0.0;
        if (is_active[i]) {
            let c = f.z + distance(f_pos, f.xy) * 2.2;
            costs[i] = c;
            min_cost = min(min_cost, c);
        }
    }

    var total_w = 0.0;
    var acc_col = vec3(0.0);
    for (var i = 0; i < MAX_FIRMS; i++) {
        if (!is_active[i]) { continue; }
        let w = exp(-12.0 * (costs[i] - min_cost));
        let p = 1.0 / (1.0 + exp(12.0 * (costs[i] - 1.0)));
        acc_col += get_distinct_color(i) * w * p;
        total_w += w;
    }
    if (total_w > 0.0) { final_col = mix(final_col, (acc_col / total_w), 0.5); }

    for (var i = 0; i < MAX_FIRMS; i++) {
        let m = get_firm_main(i);
        let e = get_firm_ext(i);
        if (m.w <= 0.0) { continue; }
        let dist = distance(f_pos * screen_size, m.xy * screen_size);
        let radius = log(m.w + 1.0) * 15.0 + 5.0;
        
        if (dist < radius) {
            let f_col = get_distinct_color(i);
            let age_ratio = ((f32(time.frame) - e.x) % 6000.0) / 6000.0;
            let angle = (atan2(f_pos.x - m.x, -(f_pos.y - m.y)) / 6.283) + 0.5;
            
            var circle_col = vec3(0.0);
            if (angle < age_ratio) { circle_col = mix(f_col, vec3(1.0), 0.7); }
            else { circle_col = f_col * 0.2; }
            if (dist > radius - (1.0 + m.z * 1.5)) { circle_col = vec3(1.0); }
            final_col = circle_col;
        }
    }

    let nm = vec2<f32>(mouse.pos) / screen_size;
    if (mouse.click > 0 && distance(f_pos, nm) < 0.1) { final_col += vec3(0.2, 0.0, 0.0); }
    textureStore(screen, id.xy, vec4(clamp(final_col, vec3(0.0), vec3(1.0)), 1.0));
}
