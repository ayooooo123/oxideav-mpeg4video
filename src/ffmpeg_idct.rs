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
fn idct_cols(col: &[i16; 64]) -> (u32, u32, u32, u32, u32, u32, u32, u32) {
    // `col[r * 8]` holds the column's coefficient for row `r` (the
    // caller assembles it in the same strided layout the C's `block +
    // c` pointer walk reads through `col[8 * n]`), so `rd(n)` mirrors
    // the C's `col[8 * n]` reads exactly.
    let rd = |n: usize| -> u32 { col[8 * n] as i32 as u32 };

    let mut a0: u32 = (W4 as u32)
        .wrapping_mul(rd(0).wrapping_add(((1 << (COL_SHIFT - 1)) / W4 as u32) as u32));
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

    if col[8 * 4] != 0 {
        a0 = a0.wrapping_add((W4 as u32).wrapping_mul(rd(4)));
        a1 = a1.wrapping_sub((W4 as u32).wrapping_mul(rd(4)));
        a2 = a2.wrapping_sub((W4 as u32).wrapping_mul(rd(4)));
        a3 = a3.wrapping_add((W4 as u32).wrapping_mul(rd(4)));
    }

    if col[8 * 5] != 0 {
        b0 = b0.wrapping_add((W5 as u32).wrapping_mul(rd(5)));
        b1 = b1.wrapping_sub((W1 as u32).wrapping_mul(rd(5)));
        b2 = b2.wrapping_add((W7 as u32).wrapping_mul(rd(5)));
        b3 = b3.wrapping_add((W3 as u32).wrapping_mul(rd(5)));
    }

    if col[8 * 6] != 0 {
        a0 = a0.wrapping_add((W6 as u32).wrapping_mul(rd(6)));
        a1 = a1.wrapping_sub((W2 as u32).wrapping_mul(rd(6)));
        a2 = a2.wrapping_add((W2 as u32).wrapping_mul(rd(6)));
        a3 = a3.wrapping_sub((W6 as u32).wrapping_mul(rd(6)));
    }

    if col[8 * 7] != 0 {
        b0 = b0.wrapping_add((W7 as u32).wrapping_mul(rd(7)));
        b1 = b1.wrapping_sub((W5 as u32).wrapping_mul(rd(7)));
        b2 = b2.wrapping_add((W3 as u32).wrapping_mul(rd(7)));
        b3 = b3.wrapping_sub((W1 as u32).wrapping_mul(rd(7)));
    }

    (a0, a1, a2, a3, b0, b1, b2, b3)
}

/// `ff_simple_idct_add_int16_8bit` semantics for a residual block: the
/// same row/column passes as [`simple_idct_put_8bit`] but **without** the
/// `av_clip_pixel` — MPEG-4 inter residuals are signed and are clipped
/// only after the §7.3 prediction add (FFmpeg's `add_dct` /
/// `put_signed_pixels_clamped` behaviour). `out[col][row]`, values in
/// `-2048..=2047` (the C's signed 8-bit saturating store maps to i16).
pub fn simple_idct_add_8bit(block: &mut [i16; 64]) -> [[i16; 8]; 8] {
    for r in 0..8 {
        let mut row = [0i16; 8];
        row.copy_from_slice(&block[r * 8..r * 8 + 8]);
        idct_row_cond_dc(&mut row);
        block[r * 8..r * 8 + 8].copy_from_slice(&row);
    }
    let mut out = [[0i16; 8]; 8];
    for c in 0..8 {
        // Mirror the C's strided view: `col[8 * r]` holds the column's
        // coefficient for row `r` (the C reads `col[8 * n]` off `block +
        // c`, whose `col[8*n]` is `block[8*n*8 + c]` — coefficient row
        // `n`, column `c`).
        let mut col = [0i16; 64];
        for r in 0..8 {
            col[r * 8] = block[r * 8 + c];
        }
        let (a0, a1, a2, a3, b0, b1, b2, b3) = idct_cols(&col);
        // The C casts the wrapped unsigned sum/difference to `int` first,
        // then does an arithmetic right shift; the signed store saturates
        // to int16 instead of clipping to 0..255.
        let px = |a: u32, b: u32| -> i16 {
            ((a.wrapping_add(b) as i32) >> COL_SHIFT).clamp(i16::MIN as i32, i16::MAX as i32) as i16
        };
        let px_neg = |a: u32, b: u32| -> i16 {
            ((a.wrapping_sub(b) as i32) >> COL_SHIFT).clamp(i16::MIN as i32, i16::MAX as i32) as i16
        };
        out[c] = [
            px(a0, b0),
            px(a1, b1),
            px(a2, b2),
            px(a3, b3),
            px_neg(a3, b3),
            px_neg(a2, b2),
            px_neg(a1, b1),
            px_neg(a0, b0),
        ];
    }
    out
}

/// `ff_simple_idct_put_int16_8bit`: one 8×8 int16 coefficient block into
/// clipped 0..255 pixels. `out[col][row]`.
pub fn simple_idct_put_8bit(block: &mut [i16; 64]) -> [[u8; 8]; 8] {
    for r in 0..8 {
        let mut row = [0i16; 8];
        row.copy_from_slice(&block[r * 8..r * 8 + 8]);
        idct_row_cond_dc(&mut row);
        block[r * 8..r * 8 + 8].copy_from_slice(&row);
    }
    let mut out = [[0u8; 8]; 8];
    for c in 0..8 {
        // Mirror the C's strided view: `col[8 * r]` holds the column's
        // coefficient for row `r` (the C reads `col[8 * n]` off `block +
        // c`, whose `col[8*n]` is `block[8*n*8 + c]` — coefficient row
        // `n`, column `c`).
        let mut col = [0i16; 64];
        for r in 0..8 {
            col[r * 8] = block[r * 8 + c];
        }
        let (a0, a1, a2, a3, b0, b1, b2, b3) = idct_cols(&col);
        // The C casts the wrapped unsigned sum/difference to `int` first,
        // then does an arithmetic right shift.
        let px = |a: u32, b: u32| -> u8 { ((a.wrapping_add(b) as i32) >> COL_SHIFT).clamp(0, 255) as u8 };
        let px_neg =
            |a: u32, b: u32| -> u8 { ((a.wrapping_sub(b) as i32) >> COL_SHIFT).clamp(0, 255) as u8 };
        out[c] = [
            px(a0, b0),
            px(a1, b1),
            px(a2, b2),
            px(a3, b3),
            px_neg(a3, b3),
            px_neg(a2, b2),
            px_neg(a1, b1),
            px_neg(a0, b0),
        ];
    }
    out
}
