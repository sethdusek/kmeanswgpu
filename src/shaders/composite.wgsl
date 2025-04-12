@group(0) @binding(0)
var assignment: texture_storage_2d<r32uint, read>;
@group(0) @binding(1)
var<storage, read> centroids: array<array<u64, 4>>;
@group(0) @binding(2)
var output_image: texture_storage_2d<bgra8unorm, write>;

struct MouseState {
    mouse_down: u32,
    x: f32,
    y: f32
}

@group(0) @binding(3)
var<uniform> mouse_state: MouseState;


@compute
@workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let load: u32 = textureLoad(assignment, global_id.xy).x;
    let centroid = centroids[load];
    var color = vec3<f32>(f32(centroid[0]) / 255.0,
                f32(centroid[1]) / 255.0,
                f32(centroid[2]) / 255.0,
                );
    let mouse_position_cluster = textureLoad(assignment, vec2<u32>(u32(mouse_state.x), u32(mouse_state.y))).x;
    if mouse_state.mouse_down != 0 && load != mouse_position_cluster {
        color*=0.7;
    }
    else if mouse_state.mouse_down != 0 {
        color = min(color*1.1, vec3(1.0));
    }

    textureStore(output_image, global_id.xy, vec4(color, 1.0));
}
