use super::*;

#[test]
fn pixels_match_scalar_rounding() {
    let src: Vec<u8> = (0..32 * 18).map(|i| ((i * 97 + i / 11) & 255) as u8).collect();
    for w in [8, 16] { for h in [8, 16] { for phase in 0..4 { for round in 0..2 { for avg in [false, true] {
        let mut dst = vec![113u8; 32 * 16];
        let mut expected = dst.clone();
        for y in 0..h { for x in 0..w {
            let at = y * 32 + x;
            let a = u32::from(src[at]);
            let v = match phase {
                0 => a,
                1 => (a + u32::from(src[at + 1]) + round) >> 1,
                2 => (a + u32::from(src[at + 32]) + round) >> 1,
                _ => (a + u32::from(src[at + 1]) + u32::from(src[at + 32]) + u32::from(src[at + 33]) + round + 1) >> 2,
            };
            expected[at] = if avg { ((113 + v + 1) >> 1) as u8 } else { v as u8 };
        }}
        pixels(&mut dst, 32, &src, 32, w, h, phase, round as u16, avg);
        assert_eq!(dst, expected);
    }}}}}
}

#[test]
fn column_transform_matches_wrapping_scalar() {
    let matrix = [
        [16383,22725,21407,19266,16383,12873,8867,4520],
        [16383,19266,8867,-4520,-16383,-22725,-21407,-12873],
        [16383,12873,-8867,-22725,-16383,4520,21407,19266],
        [16383,4520,-21407,-12873,16383,19266,-8867,-22725],
        [16383,-4520,-21407,12873,16383,-19266,-8867,22725],
        [16383,-12873,-8867,22725,-16383,-4520,21407,-19266],
        [16383,-19266,8867,4520,-16383,22725,-21407,12873],
        [16383,-22725,21407,-19266,16383,-12873,8867,-4520],
    ];
    let mut seed = 1u32;
    for _ in 0..128 {
        let block = core::array::from_fn(|_| { seed = seed.wrapping_mul(1664525).wrapping_add(1013904223); (seed >> 16) as i16 });
        let actual = columns(&block);
        for y in 0..8 { for x in 0..8 {
            let mut value = 16383i32 * 32;
            for k in 0..8 { value = value.wrapping_add((matrix[y][k] as i32).wrapping_mul(i32::from(block[k * 8 + x]))); }
            assert_eq!(actual[y][x], value >> 20);
        }}
    }
}
