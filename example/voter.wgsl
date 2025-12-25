#dispatch_once init
@compute @workgroup_size(16, 16)
fn init(@builtin(global_invocation_id) id: uint3) {
    let coord = int2(id.xy);
    
    let r = fract(sin(dot(float2(coord), float2(12.9898, 78.233))) * 43758.5453);
    let g = fract(sin(dot(float2(coord), float2(34.1234, 56.789))) * 12345.6789);
    let b = fract(sin(dot(float2(coord), float2(89.0123, 12.345))) * 67890.1234);
    
    passStore(0, coord, float4(r, g, b, 1.0));
}

@compute @workgroup_size(16, 16)
fn main_loop(@builtin(global_invocation_id) id: uint3) {
    let coord = int2(id.xy);
    let screen_size = int2(SCREEN_WIDTH, SCREEN_HEIGHT);

    // PCG Hash 기반 정수 난수 생성 (편향 방지)
    var state = uint(id.x + id.y * uint(SCREEN_WIDTH) + uint(time.frame) * 76543u);
    state = state * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    let rand_uint = (word >> 22u) ^ word;

    // 0~8 사이의 값을 뽑아 3x3 그리드 좌표(-1, 0, 1) 생성
    let move_idx = rand_uint % 9u;
    let offset_x = int(move_idx % 3u) - 1;
    let offset_y = int(move_idx / 3u) - 1;
    
    let neighbor_coord = (coord + int2(offset_x, offset_y) + screen_size) % screen_size;
    let next_gene = passLoad(0, neighbor_coord, 0);

    passStore(0, coord, next_gene);
    textureStore(screen, coord, next_gene);
}
