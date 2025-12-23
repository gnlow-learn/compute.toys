struct Agent {
    pos: vec2<f32>,
    angle: f32,
    unused: f32, 
}

@group(0) @binding(0) var<storage, read_write> agents: array<Agent>;

fn hash1(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453);
}

fn sense(agent: Agent, angle_offset: f32, dist: f32) -> f32 {
    let sensor_angle = agent.angle + angle_offset;
    let offset = vec2<f32>(cos(sensor_angle), sin(sensor_angle)) * dist;
    let sensor_pos = vec2<i32>(agent.pos + offset);
    return textureLoad(pass_in, sensor_pos, 0, 0).r;
}

@compute @workgroup_size(16, 16)
fn main_compute(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(screen);
    let write_pos = vec2<i32>(id.xy);
    
    if (id.x < u32(size.x) && id.y < u32(size.y)) {
        var sum = 0.0;
        for(var i: i32 = -1; i <= 1; i++) {
            for(var j: i32 = -1; j <= 1; j++) {
                sum += textureLoad(pass_in, write_pos + vec2<i32>(i, j), 0, 0).r;
            }
        }
        let avg = sum / 9.0;
        let final_val = avg * 0.92; 
        
        textureStore(pass_out, write_pos, 0, vec4<f32>(vec3<f32>(final_val), 1.0));
        textureStore(screen, write_pos, vec4<f32>(vec3<f32>(final_val), 1.0));
    }

    let agent_idx = id.y * 16u + id.x; 
    let num_agents = arrayLength(&agents);

    if (agent_idx < num_agents) {
        var agent = agents[agent_idx];

        if (time.elapsed < 0.1) {
            agent.pos = vec2<f32>(size) * 0.5 + vec2<f32>(hash1(f32(agent_idx)), hash1(f32(agent_idx)+1.0)) * 50.0;
            agent.angle = hash1(f32(agent_idx) * 3.1) * 6.28;
        }

        let sensor_angle = 0.785; 
        let sensor_dist = 15.0;
        let f1 = sense(agent, 0.0, sensor_dist);
        let f2 = sense(agent, sensor_angle, sensor_dist);
        let f3 = sense(agent, -sensor_angle, sensor_dist);

        let turn_speed = 0.3;
        if (f1 > f2 && f1 > f3) {
        } else if (f1 < f2 && f1 < f3) {
            agent.angle += (hash1(f32(agent_idx) + time.elapsed) - 0.5) * 2.0 * turn_speed;
        } else if (f2 > f3) {
            agent.angle += sensor_angle * turn_speed;
        } else if (f3 > f2) {
            agent.angle -= sensor_angle * turn_speed;
        }

        agent.pos += vec2<f32>(cos(agent.angle), sin(agent.angle)) * 1.5;

        if (agent.pos.x < 0.0 || agent.pos.x >= f32(size.x) || agent.pos.y < 0.0 || agent.pos.y >= f32(size.y)) {
            agent.pos = vec2<f32>(size) * 0.5; 
        }

        let ipos = vec2<i32>(agent.pos);
        textureStore(pass_out, ipos, 0, vec4<f32>(1.0));
        
        agents[agent_idx] = agent;
    }
}
