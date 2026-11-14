//! FFmpeg's integer inverse DCT ("simple IDCT"), 8-bit / int16-coefficient
//! instantiation, ported from `libavcodec/simple_idct.c` +
//! `simple_idct_template.c` (LGPL-2.1-or-later), commit 2da55bf.
//! Bit-exact with `ff_simple_idct_put_int16_8bit`, the C reference of the
//! `ff_simple_idct_put_neon` default FFmpeg uses for MPEG-4 at 8 bpp.

const W1: i64 = 22725;
const W2: i64 = 21407;
const W3: i64 = 19266;
const W4: i64 = 16383;
const W5: i64 = 12873;
const W6: i64 = 8867;
const W7: i64 = 4520;
const ROW_SHIFT: u32 = 11;
const COL_SHIFT: u32 = 20;
const DC_SHIFT: i32 = 3;

/// `idctRowCondDC` — one 8-coefficient row, in place, DC-only shortcut
/// included (`HAVE_FAST_64BIT` branch of the C).
pub fn idct_row_cond_dc(row: &mut [i16; 8]) {
    if row[1] == 0
        && row[2] == 0
        && row[3] == 0
        && row[4] == 0
        && row[5] == 0
        && row[6] == 0
        && row[7] == 0
    {
        let temp = (i64::from(row[0]) * (1 << DC_SHIFT)) & 0xffff;
        let v = temp as i16;
        *row = [v; 8];
        return;
    }

    // SUINT arithmetic: int16 sign-extended into unsigned-int multiply.
    let r = |i: usize| -> u32 { row[i] as i32 as u32 };

    let mut a0: u32 = (W4 as u32)
        .wrapping_mul(r(0))
        .wrapping_add(1 << (ROW_SHIFT - 1));
    let mut a1 = a0;
    let mut a2 = a0;
    let mut a3 = a0;

    a0 = a0.wrapping_add((W2 as u32).wrapping_mul(r(2)));
    a1 = a1.wrapping_add((W6 as u32).wrapping_mul(r(2)));
    a2 = a2.wrapping_sub((W6 as u32).wrapping_mul(r(2)));
    a3 = a3.wrapping_sub((W2 as u32).wrapping_mul(r(2)));

    let mut b0: u32 = (W1 as u32).wrapping_mul(r(1));
    b0 = b0.wrapping_add((W3 as u32).wrapping_mul(r(3)));
    let mut b1: u32 = (W3 as u32).wrapping_mul(r(1));
    b1 = b1.wrapping_sub((W7 as u32).wrapping_mul(r(3)));
    let mut b2: u32 = (W5 as u32).wrapping_mul(r(1));
    b2 = b2.wrapping_sub((W1 as u32).wrapping_mul(r(3)));
    let mut b3: u32 = (W7 as u32).wrapping_mul(r(1));
    b3 = b3.wrapping_sub((W5 as u32).wrapping_mul(r(3)));

    if row[4] != 0 || row[5] != 0 || row[6] != 0 || row[7] != 0 {
        a0 = a0
            .wrapping_add((W4 as u32).wrapping_mul(r(4)))
            .wrapping_add((W6 as u32).wrapping_mul(r(6)));
        a1 = a1
            .wrapping_sub((W4 as u32).wrapping_mul(r(4)))
            .wrapping_sub((W2 as u32).wrapping_mul(r(6)));
        a2 = a2
            .wrapping_sub((W4 as u32).wrapping_mul(r(4)))
            .wrapping_add((W2 as u32).wrapping_mul(r(6)));
        a3 = a3
            .wrapping_add((W4 as u32).wrapping_mul(r(4)))
            .wrapping_sub((W6 as u32).wrapping_mul(r(6)));

        b0 = b0.wrapping_add((W5 as u32).wrapping_mul(r(5)));
        b0 = b0.wrapping_add((W7 as u32).wrapping_mul(r(7)));

        b1 = b1.wrapping_sub((W1 as u32).wrapping_mul(r(5)));
        b1 = b1.wrapping_sub((W5 as u32).wrapping_mul(r(7)));

        b2 = b2.wrapping_add((W7 as u32).wrapping_mul(r(5)));
        b2 = b2.wrapping_add((W3 as u32).wrapping_mul(r(7)));

        b3 = b3.wrapping_add((W3 as u32).wrapping_mul(r(5)));
        b3 = b3.wrapping_sub((W1 as u32).wrapping_mul(r(7)));
    }

    let out = |a: u32, b: u32| -> i16 { ((a.wrapping_add(b) >> ROW_SHIFT) as i32) as i16 };
    let out_neg = |a: u32, b: u32| -> i16 { ((a.wrapping_sub(b) >> ROW_SHIFT) as i32) as i16 };
    row[0] = out(a0, b0);
    row[7] = out_neg(a0, b0);
    row[1] = out(a1, b1);
    row[6] = out_neg(a1, b1);
    row[2] = out(a2, b2);
    row[5] = out_neg(a2, b2);
    row[3] = out(a3, b3);
    row[4] = out_neg(a3, b3);
}

/// `IDCT_COLS` — the sparse column pass, reading `col[8*n]`.
#[cfg(not(any(
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
fn idct_cols(col: &[i16; 64], c: usize) -> (u32, u32, u32, u32, u32, u32, u32, u32) {
    // `col[r * 8]` holds the column's coefficient for row `r` (the
    // caller assembles it in the same strided layout the C's `block +
    // c` pointer walk reads through `col[8 * n]`), so `rd(n)` mirrors
    // the C's `col[8 * n]` reads exactly.
    let rd = |n: usize| -> u32 { col[8 * n + c] as i32 as u32 };

    let mut a0: u32 =
        (W4 as u32).wrapping_mul(rd(0).wrapping_add(((1 << (COL_SHIFT - 1)) / W4 as u32) as u32));
    let mut a1 = a0;
    let mut a2 = a0;
    let mut a3 = a0;

    a0 = a0.wrapping_add((W2 as u32).wrapping_mul(rd(2)));
    a1 = a1.wrapping_add((W6 as u32).wrapping_mul(rd(2)));
    a2 = a2.wrapping_sub((W6 as u32).wrapping_mul(rd(2)));
    a3 = a3.wrapping_sub((W2 as u32).wrapping_mul(rd(2)));

    let mut b0: u32 = (W1 as u32).wrapping_mul(rd(1));
    let mut b1: u32 = (W3 as u32).wrapping_mul(rd(1));
    let mut b2: u32 = (W5 as u32).wrapping_mul(rd(1));
    let mut b3: u32 = (W7 as u32).wrapping_mul(rd(1));

    b0 = b0.wrapping_add((W3 as u32).wrapping_mul(rd(3)));
    b1 = b1.wrapping_sub((W7 as u32).wrapping_mul(rd(3)));
    b2 = b2.wrapping_sub((W1 as u32).wrapping_mul(rd(3)));
    b3 = b3.wrapping_sub((W5 as u32).wrapping_mul(rd(3)));

    if col[8 * 4 + c] != 0 {
        a0 = a0.wrapping_add((W4 as u32).wrapping_mul(rd(4)));
        a1 = a1.wrapping_sub((W4 as u32).wrapping_mul(rd(4)));
        a2 = a2.wrapping_sub((W4 as u32).wrapping_mul(rd(4)));
        a3 = a3.wrapping_add((W4 as u32).wrapping_mul(rd(4)));
    }

    if col[8 * 5 + c] != 0 {
        b0 = b0.wrapping_add((W5 as u32).wrapping_mul(rd(5)));
        b1 = b1.wrapping_sub((W1 as u32).wrapping_mul(rd(5)));
        b2 = b2.wrapping_add((W7 as u32).wrapping_mul(rd(5)));
        b3 = b3.wrapping_add((W3 as u32).wrapping_mul(rd(5)));
    }

    if col[8 * 6 + c] != 0 {
        a0 = a0.wrapping_add((W6 as u32).wrapping_mul(rd(6)));
        a1 = a1.wrapping_sub((W2 as u32).wrapping_mul(rd(6)));
        a2 = a2.wrapping_add((W2 as u32).wrapping_mul(rd(6)));
        a3 = a3.wrapping_sub((W6 as u32).wrapping_mul(rd(6)));
    }

    if col[8 * 7 + c] != 0 {
        b0 = b0.wrapping_add((W7 as u32).wrapping_mul(rd(7)));
        b1 = b1.wrapping_sub((W5 as u32).wrapping_mul(rd(7)));
        b2 = b2.wrapping_add((W3 as u32).wrapping_mul(rd(7)));
        b3 = b3.wrapping_sub((W1 as u32).wrapping_mul(rd(7)));
    }

    (a0, a1, a2, a3, b0, b1, b2, b3)
}

fn transform(block: &mut [i16; 64]) -> [[i32; 8]; 8] {
    if block[1..].iter().all(|&v| v == 0) {
        let dc = block[0].wrapping_mul(8);
        block[..8].fill(dc);
        let v = 16383i32.wrapping_mul(i32::from(dc) + 32) >> COL_SHIFT;
        return [[v; 8]; 8];
    }
    for row in block.chunks_exact_mut(8) {
        idct_row_cond_dc(row.try_into().unwrap());
    }
    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    {
        crate::simd::columns(block)
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        let mut out = [[0i32; 8]; 8];
        for c in 0..8 {
            let (a0, a1, a2, a3, b0, b1, b2, b3) = idct_cols(block, c);
            let v = [
                a0.wrapping_add(b0),
                a1.wrapping_add(b1),
                a2.wrapping_add(b2),
                a3.wrapping_add(b3),
                a3.wrapping_sub(b3),
                a2.wrapping_sub(b2),
                a1.wrapping_sub(b1),
                a0.wrapping_sub(b0),
            ];
            for r in 0..8 {
                out[r][c] = (v[r] as i32) >> COL_SHIFT;
            }
        }
        out
    }
}

/// `ff_simple_idct_add_int16_8bit` semantics for a residual block: the
/// same row/column passes as [`simple_idct_put_8bit`] but **without** the
/// `av_clip_pixel` — MPEG-4 inter residuals are signed and are clipped
/// only after the §7.3 prediction add (FFmpeg's `add_dct` /
/// `put_signed_pixels_clamped` behaviour). `out[col][row]`, values in
/// `-2048..=2047` (the C's signed 8-bit saturating store maps to i16).
pub fn simple_idct_add_8bit(block: &mut [i16; 64]) -> [[i16; 8]; 8] {
    let rows = transform(block);
    let mut out = [[0i16; 8]; 8];
    for c in 0..8 {
        for r in 0..8 {
            out[c][r] = rows[r][c] as i16;
        }
    }
    out
}

/// `ff_simple_idct_put_int16_8bit`: one 8×8 int16 coefficient block into
/// clipped 0..255 pixels. `out[col][row]`.
pub fn simple_idct_put_8bit(block: &mut [i16; 64]) -> [[u8; 8]; 8] {
    let rows = transform(block);
    let mut out = [[0u8; 8]; 8];
    for r in 0..8 {
        let values = rows[r].map(|v| v as i16);
        let mut pixels = [0u8; 8];
        #[cfg(any(
            target_arch = "aarch64",
            all(target_arch = "wasm32", target_feature = "simd128")
        ))]
        crate::simd::residual(&mut pixels, &values, false);
        #[cfg(not(any(
            target_arch = "aarch64",
            all(target_arch = "wasm32", target_feature = "simd128")
        )))]
        for c in 0..8 {
            pixels[c] = values[c].clamp(0, 255) as u8;
        }
        for c in 0..8 {
            out[c][r] = pixels[c];
        }
    }
    out
}
