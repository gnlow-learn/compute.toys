struct Agent {
    pos: vec2<f32>,
    angle: f32,
    life: f32,
}

@group(0) @binding(0) var<storage, read_write> agents: array<Agent>;

fn hash1(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453);
}

fn torus_pos(p: vec2<f32>, size: vec2<f32>) -> vec2<i32> {
    return vec2<i32>((p + size) % size);
}

fn sense(agent: Agent, angle_offset: f32, dist: f32, size: vec2<f32>) -> f32 {
    let sensor_angle = agent.angle + angle_offset;
    let offset = vec2<f32>(cos(sensor_angle), sin(sensor_angle)) * dist;
    let s_pos = torus_pos(agent.pos + offset, size);
    return textureLoad(pass_in, s_pos, 0, 0).r;
}

@compute @workgroup_size(16, 16)
fn main_compute(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(screen);
    let f_size = vec2<f32>(size);
    let write_pos = vec2<i32>(id.xy);
    
    if (id.x < u32(size.x) && id.y < u32(size.y)) {
        var sum = 0.0;
        for(var i: i32 = -1; i <= 1; i++) {
            for(var j: i32 = -1; j <= 1; j++) {
                sum += textureLoad(pass_in, torus_pos(vec2<f32>(write_pos) + vec2<f32>(f32(i), f32(j)), f_size), 0, 0).r;
            }
        }
        var final_val = (sum / 9.0) * 0.88; 

        if (mouse.click > 0 && distance(vec2<f32>(write_pos), vec2<f32>(mouse.pos)) < 15.0) {
            final_val = 25.0; 
        }
        
        textureStore(pass_out, write_pos, 0, vec4<f32>(vec3<f32>(final_val), 1.0));

        let p = clamp(final_val, 0.0, 1.0);
        let color = vec3<f32>(
            pow(p, 0.5),
            pow(p, 1.2) * 0.9,
            pow(p, 4.0) * 0.3
        ) + vec3<f32>(0.02, 0.0, 0.05);
        
        textureStore(screen, write_pos, vec4<f32>(color, 1.0));
    }

    let agent_idx = id.y * 16u + id.x; 
    let num_agents = arrayLength(&agents);

    if (agent_idx < num_agents) {
        var agent = agents[agent_idx];

        if (time.elapsed < 0.1 || agent.life <= 0.0) {
            let r = hash1(f32(agent_idx) + time.elapsed) * 40.0;
            let th = hash1(f32(agent_idx) * 1.7) * 6.28;
            let spawn_origin = select(f_size * 0.5, vec2<f32>(mouse.pos), mouse.click > 0);
            agent.pos = spawn_origin + vec2<f32>(r * cos(th), r * sin(th));
            agent.angle = hash1(f32(agent_idx) * 3.1 + time.elapsed) * 6.28;
            agent.life = 1.0;
        }

        let sensor_dist = 22.0;
        let sensor_angle = 0.45;
        let f1 = sense(agent, 0.0, sensor_dist, f_size);
        let f2 = sense(agent, sensor_angle, sensor_dist, f_size);
        let f3 = sense(agent, -sensor_angle, sensor_dist, f_size);

        if (f1 + f2 + f3 < 0.1) {
            agent.life -= 0.012;
        } else {
            agent.life += 0.006;
        }
        agent.life = clamp(agent.life, 0.0, 1.0);

        let turn_speed = 0.25;
        if (f1 > f2 && f1 > f3) {
        } else if (f2 > f3) {
            agent.angle += turn_speed;
        } else if (f3 > f2) {
            agent.angle -= turn_speed;
        }

        agent.pos = (agent.pos + vec2<f32>(cos(agent.angle), sin(agent.angle)) * (1.2 + agent.life) + f_size) % f_size;

        if (agent.life > 0.1) {
            let ipos = vec2<i32>(agent.pos);
            textureStore(pass_out, ipos, 0, vec4<f32>(1.5));
        }
        
        agents[agent_idx] = agent;
    }
}
