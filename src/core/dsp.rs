pub fn to_planar_stereo(interleaved: &[f32], channels: u16) -> Vec<[f32; 2]> {
    if channels == 1 {
        interleaved.iter().map(|&x| [x, x]).collect()
    } else {
        let mut out = Vec::with_capacity(interleaved.len() / 2);
        let mut i = 0;
        while i + 1 < interleaved.len() {
            out.push([interleaved[i], interleaved[i + 1]]);
            i += 2;
        }
        out
    }
}

pub fn hann2(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let w = (std::f32::consts::PI * i as f32 / (n as f32 - 1.0)).sin();
            w * w
        })
        .collect()
}
