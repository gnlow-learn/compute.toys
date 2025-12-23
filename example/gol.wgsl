@compute @workgroup_size(16, 16)
fn main_compute(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(screen);
    if (id.x >= size.x || id.y >= size.y) { return; }

    let pos = vec2<i32>(id.xy);
    var alive_neighbors = 0;

    for (var y: i32 = -1; y <= 1; y++) {
        for (var x: i32 = -1; x <= 1; x++) {
            if (x == 0 && y == 0) { continue; }
            
            let neighbor = textureLoad(pass_in, pos + vec2<i32>(x, y), 0, 0).r;
            if (neighbor > 0.5) {
                alive_neighbors++;
            }
        }
    }

    let current_alive = textureLoad(pass_in, pos, 0, 0).r > 0.5;
    var next_state = 0.0;

    if (current_alive) {
        if (alive_neighbors == 2 || alive_neighbors == 3) {
            next_state = 1.0; 
        }
    } else {
        if (alive_neighbors == 3) {
            next_state = 1.0;
        }
    }

    if (mouse.click > 0 && distance(vec2<f32>(pos), vec2<f32>(mouse.pos)) < 15.0) {
        next_state = 1.0;
    }
    
    if (time.elapsed < 0.1 && hash(vec2<f32>(pos)) > 0.9) {
        next_state = 1.0;
    }

    textureStore(pass_out, pos, 0, vec4<f32>(next_state, 0.0, 0.0, 1.0));
    
    textureStore(screen, pos, vec4<f32>(vec3<f32>(next_state), 1.0));
}

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}
