//! MPEG-4 Simple Studio Profile decoding: 10-bit intra VOPs in 4:2:2,
//! 4:4:4 or RGB whose macroblocks are DCT- or DPCM-coded.
//!
//! Ported from FFmpeg (commit 2da55bf): libavcodec/mpeg4videodec.c
//! (decode_studio_vol_header, decode_studiovisualobject,
//! decode_studio_vop_header, ff_mpeg4_decode_studio_slice_header,
//! mpeg4_decode_studio_mb, mpeg4_decode_studio_block,
//! mpeg4_decode_dpcm_macroblock, read_quant_matrix_ext,
//! extension_and_user_data, next_start_code_studio,
//! ff_mpeg4_decode_studio), h263dec.c (ff_h263_decode_frame and
//! decode_slice, their studio paths), ituh263dec.c (ff_h263_resync, studio
//! path), mpeg4data.h (the studio VLC tables) and
//! simple_idct_template.c (BIT_DEPTH 10, IN_IDCT_DEPTH 32: the
//! `ff_simple_idct_put_int32_10bit` FFmpeg's studio profile uses).
//! License: LGPL-2.1-or-later

use crate::decoder::StreamDecodeError;
use oxideav_core::PixelFormat;

const SLICE_START_CODE: u32 = 0x1B7;
const EXT_START_CODE: u32 = 0x1B8;
const USER_DATA_START_CODE: u32 = 0x1B2;
const GOP_START_CODE: u32 = 0x1B3;
const VOS_START_CODE: u32 = 0x1B0;
const VISUAL_OBJECT_START_CODE: u32 = 0x1B5;
const VOP_START_CODE: u32 = 0x1B6;

/// AV_PROFILE_MPEG4_SIMPLE_STUDIO and the studio video object types.
const SIMPLE_STUDIO_PROFILE: u32 = 14;
const SIMPLE_STUDIO_VO_TYPE: u32 = 14;
const CORE_STUDIO_VO_TYPE: u32 = 15;

/// QUANT_MATRIX_EXT_ID
const QUANT_MATRIX_EXT_ID: u32 = 3;

/// Chroma formats: CHROMA_422, CHROMA_444.
const CHROMA_422: u32 = 2;

/// Blocks per macroblock by chroma format (mpeg4_block_count).
const BLOCK_COUNT: [usize; 4] = [0, 6, 8, 12];

/// The studio profile's sample depth (decode_studio_vol_header accepts
/// only 10).
const BITS: u32 = 10;

const ERR: fn(&'static str) -> StreamDecodeError = StreamDecodeError::Studio;

/// Whether `data` starts a studio-profile stream: a visual object
/// sequence of the Simple Studio Profile (levels 1 to 8) or a VOL of a
/// studio object type, before any VOP (ff_mpeg4_parse_picture_header).
pub fn starts_studio(data: &[u8]) -> bool {
    let mut i = 0;
    while i + 5 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            let code = 0x100 | u32::from(data[i + 3]);
            match code {
                VOS_START_CODE => {
                    let (profile, level) = (u32::from(data[i + 4] >> 4), u32::from(data[i + 4] & 15));
                    return profile == SIMPLE_STUDIO_PROFILE && (1..9).contains(&level);
                }
                0x120..=0x12F => {
                    // random_accessible_vol, then video_object_type_indication
                    let vo_type = (u32::from(data[i + 4]) << 1 | u32::from(data.get(i + 5).copied().unwrap_or(0)) >> 7) & 0xFF;
                    return vo_type == SIMPLE_STUDIO_VO_TYPE || vo_type == CORE_STUDIO_VO_TYPE;
                }
                VOP_START_CODE => return false,
                _ => {}
            }
            i += 3;
        } else {
            i += 1;
        }
    }
    false
}

/// GetBitContext over a packet: MSB first, zeros past the end.
struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Bits<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn left(&self) -> isize {
        (self.data.len() * 8) as isize - self.pos as isize
    }

    fn show(&self, n: usize) -> u32 {
        let mut v = 0u32;
        for k in 0..n {
            let at = self.pos + k;
            let bit = self.data.get(at / 8).map_or(0, |b| (b >> (7 - at % 8)) & 1);
            v = (v << 1) | u32::from(bit);
        }
        v
    }

    fn get(&mut self, n: usize) -> u32 {
        let v = self.show(n);
        self.pos += n;
        v
    }

    fn skip(&mut self, n: usize) {
        self.pos += n;
    }

    fn align(&mut self) {
        self.pos = self.pos.div_ceil(8) * 8;
    }

    /// get_xbits: a sign-folded value of `n` bits (a leading 0 makes it
    /// negative).
    fn get_xbits(&mut self, n: usize) -> i32 {
        let v = self.get(n) as i32;
        if v >> (n - 1) != 0 { v } else { v - ((1 << n) - 1) }
    }

    /// get_unary(gb, 1, len): zero bits before a one, at most `len`.
    fn unary(&mut self, len: u32) -> u32 {
        let mut n = 0;
        while n < len && self.get(1) == 0 {
            n += 1;
        }
        n
    }

    /// next_start_code_studio
    fn next_start_code(&mut self) {
        self.align();
        while self.left() >= 24 && self.show(24) != 1 {
            self.skip(8);
        }
    }
}

/// A VLC built as ff_vlc_init_from_lengths builds one: codes assigned in
/// table order, a negative length taking code space without a symbol.
struct Vlc {
    /// (length, code, symbol), the code in the low `length` bits.
    codes: Vec<(u32, u32, u8)>,
    max_len: u32,
}

impl Vlc {
    fn from_lengths(table: &[(u8, i8)]) -> Self {
        let mut codes = Vec::with_capacity(table.len());
        let mut code: u64 = 0;
        let mut max_len = 0;
        for &(symbol, len) in table {
            let bits = u32::from(len.unsigned_abs());
            if len > 0 {
                codes.push((bits, (code >> (32 - bits)) as u32, symbol));
                max_len = max_len.max(bits);
            }
            if len != 0 {
                code += 1u64 << (32 - bits);
            }
        }
        Self { codes, max_len }
    }

    /// get_vlc2: the symbol of the code that starts at the reader, or
    /// `None` for an invalid one.
    fn read(&self, gb: &mut Bits<'_>) -> Option<u32> {
        for len in 1..=self.max_len {
            let v = gb.show(len as usize);
            if let Some(&(_, _, symbol)) = self.codes.iter().find(|&&(l, c, _)| l == len && c == v) {
                gb.skip(len as usize);
                return Some(u32::from(symbol));
            }
        }
        None
    }
}

/// ff_mpeg4_studio_dc_luma: (symbol, length)
const DC_LUMA: [(u8, i8); 19] = [
    (2, 4), (10, 4), (3, 4), (1, 5), (0, 6), (11, 7), (12, 8), (13, 9), (14, 10), (15, 11), (16, 12), (17, 13),
    (18, 13), (7, 3), (6, 3), (8, 3), (5, 3), (9, 3), (4, 3),
];

/// ff_mpeg4_studio_dc_chroma
const DC_CHROMA: [(u8, i8); 19] = [
    (0, 4), (8, 4), (1, 4), (9, 5), (10, 6), (11, 7), (12, 8), (13, 9), (14, 10), (15, 11), (16, 12), (17, 13),
    (18, 13), (5, 3), (4, 3), (6, 3), (3, 3), (7, 3), (2, 3),
];

/// ff_mpeg4_studio_intra
const INTRA: [&[(u8, i8)]; 12] = [
    &[
        (0, -6), (21, 13), (6, 13), (5, 12), (4, 11), (20, 10), (3, 9), (12, 8), (11, 7), (10, 7), (2, 7), (19, 6),
        (18, 6), (9, 6), (8, 5), (17, 4), (7, 4), (1, 4), (0, 4), (16, 3), (15, 3), (14, 3), (13, 2),
    ],
    &[(0, -6), (21, 8), (20, 8), (19, 7), (18, 5), (17, 4), (16, 3), (15, 2), (14, 1)],
    &[
        (0, -6), (0, -15), (20, 15), (19, 14), (6, 14), (5, 14), (21, 13), (18, 13), (17, 11), (12, 10), (4, 9), (16, 8),
        (3, 7), (15, 6), (11, 6), (2, 5), (1, 5), (10, 4), (9, 4), (14, 3), (8, 3), (7, 3), (0, 3), (13, 2),
    ],
    &[
        (0, -6), (20, 13), (12, 13), (6, 13), (5, 13), (21, 12), (19, 12), (18, 10), (4, 9), (11, 8), (17, 7), (16, 6),
        (3, 6), (15, 5), (10, 5), (2, 5), (0, 5), (9, 4), (8, 4), (1, 4), (7, 3), (14, 2), (13, 2),
    ],
    &[
        (0, -6), (0, -15), (12, 15), (6, 14), (21, 13), (20, 13), (5, 13), (19, 11), (11, 10), (4, 9), (18, 8), (10, 7),
        (3, 7), (0, 7), (17, 6), (16, 6), (9, 6), (2, 5), (8, 4), (1, 4), (15, 3), (7, 3), (14, 2), (13, 2),
    ],
    &[
        (0, -6), (0, -15), (20, 15), (12, 14), (11, 13), (6, 13), (5, 13), (21, 12), (4, 12), (19, 11), (10, 11), (3, 10),
        (0, 10), (9, 8), (18, 7), (8, 7), (2, 7), (17, 6), (7, 5), (1, 5), (16, 3), (15, 2), (14, 2), (13, 2),
    ],
    &[
        (0, -6), (0, -15), (12, 15), (11, 14), (6, 14), (5, 14), (21, 12), (20, 12), (10, 12), (4, 11), (0, 11), (9, 10),
        (3, 10), (19, 8), (8, 8), (2, 8), (18, 6), (7, 6), (1, 4), (17, 3), (14, 3), (13, 3), (16, 2), (15, 2),
    ],
    &[
        (0, -6), (12, 12), (6, 12), (21, 11), (11, 11), (5, 11), (20, 10), (10, 10), (9, 9), (0, 9), (8, 8), (2, 8),
        (19, 7), (7, 7), (4, 7), (3, 7), (18, 5), (1, 5), (14, 4), (13, 4), (17, 2), (16, 2), (15, 2),
    ],
    &[
        (0, -6), (12, 13), (6, 13), (21, 12), (11, 12), (5, 12), (20, 11), (3, 11), (10, 10), (9, 10), (2, 10), (0, 10),
        (8, 9), (7, 8), (4, 8), (19, 6), (1, 6), (13, 4), (18, 3), (15, 3), (14, 3), (17, 2), (16, 2),
    ],
    &[
        (0, -6), (12, 12), (11, 12), (6, 12), (0, 12), (21, 10), (10, 10), (5, 10), (20, 8), (9, 8), (2, 8), (8, 7),
        (7, 7), (4, 6), (3, 6), (1, 6), (13, 5), (19, 4), (14, 4), (16, 3), (15, 3), (18, 2), (17, 2),
    ],
    &[
        (0, -6), (12, 13), (6, 13), (5, 13), (0, 13), (4, 11), (11, 10), (21, 9), (10, 9), (9, 9), (8, 8), (2, 8),
        (7, 7), (1, 7), (20, 6), (14, 5), (13, 5), (15, 4), (3, 4), (17, 3), (16, 3), (19, 2), (18, 2),
    ],
    &[
        (0, -6), (6, 11), (5, 11), (12, 10), (11, 10), (0, 10), (21, 9), (10, 9), (4, 9), (3, 9), (9, 8), (8, 6),
        (2, 6), (7, 5), (1, 5), (18, 4), (17, 4), (16, 4), (15, 4), (19, 3), (14, 3), (13, 3), (20, 2),
    ],
];

/// ac_state_tab: (additional_code length, next VLC table) per group.
const AC_STATE: [(u32, usize); 22] = [
    (0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2),
    (6, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9), (8, 10), (0, 11),
];

/// ff_zigzag_direct
const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21,
    28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61,
    54, 47, 55, 62, 63,
];

/// ff_alternate_vertical_scan
const ALTERNATE_VERTICAL: [usize; 64] = [
    0, 8, 16, 24, 1, 9, 2, 10, 17, 25, 32, 40, 48, 56, 57, 49, 41, 33, 26, 18, 3, 11, 4, 12, 19, 27, 34, 42, 50, 58,
    35, 43, 51, 59, 20, 28, 5, 13, 6, 14, 21, 29, 36, 44, 52, 60, 37, 45, 53, 61, 22, 30, 7, 15, 23, 31, 38, 46, 54,
    62, 39, 47, 55, 63,
];

/// ff_mpeg2_non_linear_qscale
const NON_LINEAR_QSCALE: [i32; 32] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 72, 80, 88, 96,
    104, 112,
];

/// ff_mpeg4_default_intra_matrix, in raster order.
fn default_intra_matrix() -> [i32; 64] {
    let mut m = [0; 64];
    for (r, row) in crate::block::DEFAULT_INTRA_QUANT_MATRIX.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            m[r * 8 + c] = i32::from(v);
        }
    }
    m
}

/// One decoded studio picture, cropped to its visible size: 10-bit
/// samples, planes in the order the pixel format names them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StudioFrame {
    pub width: u32,
    pub height: u32,
    pub pixel_format: PixelFormat,
    /// Each plane's samples, row by row without padding.
    pub planes: [Vec<u16>; 3],
    /// Each plane's width in samples.
    pub plane_widths: [usize; 3],
    /// The pts of the packet the picture came from.
    pub pts: Option<i64>,
}

/// decode_studio_vol_header
#[derive(Debug, Clone, Copy)]
struct StudioVol {
    width: u32,
    height: u32,
    rgb: bool,
    chroma_format: u32,
    low_delay: bool,
    mpeg_quant: bool,
}

impl StudioVol {
    /// (chroma_x_shift, chroma_y_shift)
    fn chroma_shift(&self) -> (u32, u32) {
        if self.chroma_format == CHROMA_422 { (1, 0) } else { (0, 0) }
    }
}

/// The picture being decoded, in macroblock-aligned planes.
struct Picture {
    planes: [Vec<u16>; 3],
    strides: [usize; 3],
}

/// The studio-profile decoder state FFmpeg keeps between packets.
pub struct StudioDecoder {
    vol: Option<StudioVol>,
    profile: Option<u32>,
    intra_matrix: [i32; 64],
    chroma_intra_matrix: [i32; 64],
    alternate_scan: bool,
    dct_precision: u32,
    intra_dc_precision: u32,
    q_scale_type: bool,
    qscale: i32,
    last_dc: [i32; 3],
    /// dpcm_direction of the macroblock last decoded: 0 DCT, 1 or -1
    /// DPCM.
    dpcm_direction: i32,
    block32: [[i32; 64]; 12],
    dpcm_macroblock: [[u16; 256]; 3],
    dc_luma: Vlc,
    dc_chroma: Vlc,
    intra: Vec<Vlc>,
    /// The picture of a stream without low_delay, shown at the next.
    held: Option<StudioFrame>,
}

impl std::fmt::Debug for StudioDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StudioDecoder").field("vol", &self.vol).finish()
    }
}

impl Default for StudioDecoder {
    fn default() -> Self {
        Self {
            vol: None,
            profile: None,
            intra_matrix: default_intra_matrix(),
            chroma_intra_matrix: default_intra_matrix(),
            alternate_scan: false,
            dct_precision: 0,
            intra_dc_precision: 0,
            q_scale_type: false,
            qscale: 0,
            last_dc: [0; 3],
            dpcm_direction: 0,
            block32: [[0; 64]; 12],
            dpcm_macroblock: [[0; 256]; 3],
            dc_luma: Vlc::from_lengths(&DC_LUMA),
            dc_chroma: Vlc::from_lengths(&DC_CHROMA),
            intra: INTRA.iter().map(|t| Vlc::from_lengths(t)).collect(),
            held: None,
        }
    }
}

/// What a macroblock decode left: more macroblocks in the slice, or its
/// end (SLICE_OK, SLICE_END).
enum MbEnd {
    More,
    SliceEnd,
}

impl StudioDecoder {
    /// The visible size and pixel format of the pictures, once a VOL
    /// gave them.
    pub fn layout(&self) -> Option<((u32, u32), PixelFormat)> {
        self.vol.map(|v| ((v.width, v.height), pixel_format(&v)))
    }

    /// One packet: its headers and, from its first VOP, a picture (FFmpeg
    /// shows it now in a low-delay stream, else at the next picture or
    /// the flush).
    pub fn decode_packet(&mut self, data: &[u8], pts: Option<i64>) -> Result<Vec<StudioFrame>, StreamDecodeError> {
        let mut gb = Bits::new(data);
        let mut startcode: u32 = 0xFF;
        let mut vol_seen = false;
        loop {
            if gb.left() <= 0 {
                // No VOP: headers only.
                return Ok(Vec::new());
            }
            startcode = (startcode << 8) | gb.get(8);
            if startcode & 0xFFFF_FF00 != 0x100 {
                continue;
            }
            match startcode {
                0x120..=0x12F => {
                    if !vol_seen {
                        vol_seen = true;
                        self.decode_vol(&mut gb)?;
                    }
                }
                GOP_START_CODE => {
                    if gb.show(23) != 0 {
                        gb.skip(5 + 6 + 1 + 6 + 2);
                    }
                }
                VOS_START_CODE => {
                    let profile = gb.get(4);
                    let level = gb.get(4);
                    if profile != SIMPLE_STUDIO_PROFILE || !(1..9).contains(&level) {
                        return Err(ERR("a mix of studio and non-studio profile"));
                    }
                    gb.next_start_code();
                    self.extension_and_user_data(&mut gb, 0);
                    self.profile = Some(profile);
                }
                VISUAL_OBJECT_START_CODE => {
                    gb.skip(4); // visual_object_verid
                    if gb.get(4) != 1 {
                        return Err(ERR("visual object type other than video"));
                    }
                    gb.next_start_code();
                    self.extension_and_user_data(&mut gb, 1);
                }
                VOP_START_CODE => break,
                _ => {}
            }
            gb.align();
            startcode = 0xFF;
        }
        let vol = self.vol.ok_or(ERR("VOP before a VOL header"))?;
        self.decode_vop_header(&mut gb);

        let (mb_width, mb_height) = (vol.width.div_ceil(16) as usize, vol.height.div_ceil(16) as usize);
        let (hsub, vsub) = vol.chroma_shift();
        let strides = [mb_width * 16, (mb_width * 16) >> hsub, (mb_width * 16) >> hsub];
        let rows = [mb_height * 16, (mb_height * 16) >> vsub, (mb_height * 16) >> vsub];
        let mut picture = Picture {
            planes: [vec![0; strides[0] * rows[0]], vec![0; strides[1] * rows[1]], vec![0; strides[2] * rows[2]]],
            strides,
        };
        // ff_h263_decode_frame: the first slice at the reader, then each
        // slice ff_h263_resync finds. A slice that fails ends itself.
        let (mut mb_x, mut mb_y) = (0usize, 0usize);
        let _ = self.decode_slice(&mut gb, &vol, &mut picture, (mb_width, mb_height), &mut mb_x, &mut mb_y);
        while mb_y < mb_height {
            if !resync(&mut gb) {
                break;
            }
            let _ = self.decode_slice(&mut gb, &vol, &mut picture, (mb_width, mb_height), &mut mb_x, &mut mb_y);
        }
        let frame = crop(picture, &vol, pts);
        if vol.low_delay {
            Ok(vec![frame])
        } else {
            Ok(self.held.replace(frame).into_iter().collect())
        }
    }

    /// The picture a stream without low_delay still holds.
    pub fn flush(&mut self) -> Option<StudioFrame> {
        self.held.take()
    }

    /// decode_vol_header for a studio stream: decode_studio_vol_header
    /// with its trailing start code search and extensions.
    fn decode_vol(&mut self, gb: &mut Bits<'_>) -> Result<(), StreamDecodeError> {
        gb.skip(1); // random_accessible_vol
        let vo_type = gb.get(8);
        if vo_type != SIMPLE_STUDIO_VO_TYPE && vo_type != CORE_STUDIO_VO_TYPE {
            return Err(ERR("a mix of studio and non-studio profile"));
        }
        if self.profile.is_some_and(|p| p != SIMPLE_STUDIO_PROFILE) {
            return Err(ERR("a studio VOL in a non-studio profile"));
        }
        self.profile = Some(SIMPLE_STUDIO_PROFILE);
        gb.skip(4); // video_object_layer_verid
        let shape = gb.get(2);
        gb.skip(4 + 1); // shape extension, progressive_sequence
        if shape != 0 {
            return Err(ERR("non-rectangular shape"));
        }
        let rgb = gb.get(1) != 0;
        let chroma_format = gb.get(2);
        if chroma_format == 0 || chroma_format == 1 || (rgb && chroma_format == CHROMA_422) {
            return Err(ERR("illegal chroma format"));
        }
        if gb.get(4) != BITS {
            return Err(ERR("bit depth other than 10"));
        }
        gb.skip(1);
        let width = gb.get(14);
        gb.skip(1);
        let height = gb.get(14);
        gb.skip(1);
        let (width, height) = match self.vol {
            _ if width != 0 && height != 0 => (width, height),
            Some(v) => (v.width, v.height),
            None => return Err(ERR("VOL without a size")),
        };
        if gb.get(4) == 15 {
            gb.skip(16); // extended pixel aspect ratio
        }
        // frame_rate_code, bit rate and VBV fields with their markers
        gb.skip(4 + 15 + 1 + 15 + 1 + 15 + 1 + 3 + 11 + 1 + 15 + 1);
        let low_delay = gb.get(1) != 0;
        let mpeg_quant = gb.get(1) != 0; // mpeg2_stream
        self.vol = Some(StudioVol { width, height, rgb, chroma_format, low_delay, mpeg_quant });
        gb.next_start_code();
        self.extension_and_user_data(gb, 2);
        Ok(())
    }

    /// decode_studio_vop_header
    fn decode_vop_header(&mut self, gb: &mut Bits<'_>) {
        if gb.left() <= 32 {
            return;
        }
        // SMPTE time code with its markers, temporal_reference,
        // vop_structure
        gb.skip(16 + 1 + 16 + 1 + 16 + 1 + 16 + 1 + 4 + 10 + 2);
        let intra = gb.get(2) == 0; // vop_coding_type
        if gb.get(1) != 0 {
            gb.skip(3); // top_field_first, repeat_first_field, progressive_frame
        }
        if intra && gb.get(1) != 0 {
            self.reset_dc_predictors();
        }
        // The shape is rectangular.
        self.alternate_scan = gb.get(1) != 0;
        gb.skip(1); // frame_pred_frame_dct
        self.dct_precision = gb.get(2);
        self.intra_dc_precision = gb.get(2);
        self.q_scale_type = gb.get(1) != 0;
        // mpeg4_load_default_matrices: a VOP starts from the default
        // matrices; only its own extension changes them.
        self.intra_matrix = default_intra_matrix();
        self.chroma_intra_matrix = default_intra_matrix();
        gb.next_start_code();
        self.extension_and_user_data(gb, 4);
    }

    /// extension_and_user_data: a quantiser matrix extension after the
    /// VOL (`id` 2) or the VOP (4).
    fn extension_and_user_data(&mut self, gb: &mut Bits<'_>, id: u32) {
        if gb.show(32) == EXT_START_CODE && (id == 2 || id == 4) {
            gb.skip(32);
            if gb.get(4) == QUANT_MATRIX_EXT_ID {
                self.read_quant_matrix_ext(gb);
            }
        }
        let _ = USER_DATA_START_CODE;
    }

    /// read_quant_matrix_ext: matrices in zigzag order; the non-intra
    /// ones are read past.
    fn read_quant_matrix_ext(&mut self, gb: &mut Bits<'_>) {
        for target in 0..4 {
            if gb.get(1) == 0 {
                continue;
            }
            if gb.left() < 64 * 8 {
                return;
            }
            for &at in &ZIGZAG {
                let v = gb.get(8) as i32;
                match target {
                    0 => {
                        self.intra_matrix[at] = v;
                        self.chroma_intra_matrix[at] = v;
                    }
                    2 => self.chroma_intra_matrix[at] = v,
                    _ => {}
                }
            }
        }
        gb.next_start_code();
    }

    /// reset_studio_dc_predictors
    fn reset_dc_predictors(&mut self) {
        let dc = 1 << (BITS + self.dct_precision + self.intra_dc_precision - 1);
        self.last_dc = [dc; 3];
    }

    /// mpeg_get_qscale
    fn read_qscale(&self, gb: &mut Bits<'_>) -> i32 {
        let q = gb.get(5) as usize;
        if self.q_scale_type { NON_LINEAR_QSCALE[q] } else { (q as i32) << 1 }
    }

    /// decode_slice with ff_mpeg4_decode_studio_slice_header: the
    /// macroblocks from the one the slice header names until the slice
    /// ends.
    fn decode_slice(
        &mut self,
        gb: &mut Bits<'_>,
        vol: &StudioVol,
        picture: &mut Picture,
        (mb_width, mb_height): (usize, usize),
        mb_x: &mut usize,
        mb_y: &mut usize,
    ) -> Result<(), StreamDecodeError> {
        if !(gb.left() >= 32 && gb.get(32) == SLICE_START_CODE) {
            return Err(ERR("slice without a slice start code"));
        }
        let mb_count = mb_width * mb_height;
        let vlc_len = (usize::BITS - mb_count.leading_zeros()) as usize; // av_log2(mb_count) + 1
        let mb_num = gb.get(vlc_len) as usize;
        if mb_num >= mb_count {
            return Err(ERR("slice starts past the picture"));
        }
        *mb_x = mb_num % mb_width;
        *mb_y = mb_num / mb_width;
        self.qscale = self.read_qscale(gb);
        if gb.get(1) != 0 {
            // slice extension: intra_slice, slice_VOP_id_enable,
            // slice_VOP_id, then extra_information_slice bytes
            gb.skip(1 + 1 + 6);
            while gb.get(1) != 0 {
                gb.skip(8);
            }
        }
        self.reset_dc_predictors();
        while *mb_y < mb_height {
            while *mb_x < mb_width {
                let end = self.decode_mb(gb, vol)?;
                self.reconstruct(picture, vol, *mb_x, *mb_y);
                if let MbEnd::SliceEnd = end {
                    *mb_x += 1;
                    if *mb_x >= mb_width {
                        *mb_x = 0;
                        *mb_y += 1;
                    }
                    return Ok(());
                }
                *mb_x += 1;
            }
            *mb_x = 0;
            *mb_y += 1;
        }
        Ok(())
    }

    /// mpeg4_decode_studio_mb
    fn decode_mb(&mut self, gb: &mut Bits<'_>, vol: &StudioVol) -> Result<MbEnd, StreamDecodeError> {
        self.dpcm_direction = 0;
        if gb.get(1) != 0 {
            // DCT; macroblock_type: a quantiser change when it is 0
            if gb.get(1) == 0 {
                gb.skip(1);
                self.qscale = self.read_qscale(gb);
            }
            for n in 0..BLOCK_COUNT[vol.chroma_format as usize] {
                self.decode_block(gb, vol, n)?;
            }
        } else {
            gb.skip(1); // marker
            self.dpcm_direction = if gb.get(1) != 0 { -1 } else { 1 };
            for n in 0..3 {
                self.decode_dpcm_macroblock(gb, vol, n)?;
            }
        }
        let left = gb.left();
        if left >= 24 && gb.show(23) == 0 {
            gb.next_start_code();
            return Ok(MbEnd::SliceEnd);
        }
        if left == 0 || ((0..8).contains(&left) && gb.show(left as usize) == 0) {
            return Ok(MbEnd::SliceEnd);
        }
        Ok(MbEnd::More)
    }

    /// mpeg4_decode_studio_block: the dequantised coefficients of block
    /// `n` (0 to 3 luma, then alternating Cb and Cr).
    fn decode_block(&mut self, gb: &mut Bits<'_>, vol: &StudioVol, n: usize) -> Result<(), StreamDecodeError> {
        let min = -(1 << (BITS + 6));
        let max = (1 << (BITS + 6)) - 1;
        let shift = 3 - self.dct_precision as i32;
        let scan = if self.alternate_scan { &ALTERNATE_VERTICAL } else { &ZIGZAG };
        let (cc, dc_vlc, matrix) = if n < 4 {
            (0, &self.dc_luma, &self.intra_matrix)
        } else {
            ((n & 1) + 1, if vol.rgb { &self.dc_luma } else { &self.dc_chroma }, &self.chroma_intra_matrix)
        };
        let block = &mut self.block32[n];
        *block = [0; 64];
        let dct_dc_size = dc_vlc.read(gb).ok_or(ERR("illegal dc size"))?;
        let dct_diff = if dct_dc_size == 0 {
            0
        } else {
            let diff = gb.get_xbits(dct_dc_size as usize);
            if dct_dc_size > 8 && gb.get(1) == 0 {
                return Err(ERR("missing marker after a large dc"));
            }
            diff
        };
        self.last_dc[cc] = self.last_dc[cc].wrapping_add(dct_diff);
        block[0] = if vol.mpeg_quant {
            self.last_dc[cc].wrapping_mul(8 >> self.intra_dc_precision)
        } else {
            self.last_dc[cc].wrapping_mul(8 >> self.intra_dc_precision).wrapping_mul(8 >> self.dct_precision)
        };
        block[0] = block[0].clamp(min, max);
        let mut mismatch = 1 ^ block[0];
        let mut vlc = &self.intra[0];
        let mut idx = 1usize;
        loop {
            let group = vlc.read(gb).ok_or(ERR("illegal ac coefficient group"))? as usize;
            let (additional, next) = AC_STATE[group];
            vlc = &self.intra[next];
            let j;
            match group {
                0 => break,
                1..=6 => {
                    // zero run
                    let mut run = 1usize << additional;
                    if additional != 0 {
                        run += gb.get(additional as usize) as usize;
                    }
                    idx += run;
                    continue;
                }
                7..=12 => {
                    // zero run and a level of +-1
                    let code = gb.get(additional as usize) as usize;
                    let sign = code & 1;
                    idx += (1 << (additional - 1)) + (code >> 1);
                    if idx > 63 {
                        return Err(ERR("coefficient past the block"));
                    }
                    j = scan[idx];
                    idx += 1;
                    block[j] = if sign != 0 { 1 } else { -1 };
                }
                13..=20 => {
                    if idx > 63 {
                        return Err(ERR("coefficient past the block"));
                    }
                    j = scan[idx];
                    idx += 1;
                    block[j] = gb.get_xbits(additional as usize);
                }
                _ => {
                    // escape
                    if idx > 63 {
                        return Err(ERR("coefficient past the block"));
                    }
                    j = scan[idx];
                    idx += 1;
                    let len = (BITS + self.dct_precision + 4) as usize;
                    let flc = gb.get(len) as i32;
                    block[j] = if flc >> (len - 1) != 0 { -((flc ^ ((1 << len) - 1)) + 1) } else { flc };
                }
            }
            block[j] = block[j].wrapping_mul(matrix[j]).wrapping_mul(self.qscale).wrapping_mul(1 << shift) / 16;
            block[j] = block[j].clamp(min, max);
            mismatch ^= block[j];
        }
        block[63] ^= mismatch & 1;
        Ok(())
    }

    /// mpeg4_decode_dpcm_macroblock: component `n` of a DPCM macroblock,
    /// predicted from its left, top and top-left samples.
    fn decode_dpcm_macroblock(&mut self, gb: &mut Bits<'_>, vol: &StudioVol, n: usize) -> Result<(), StreamDecodeError> {
        let (hsub, vsub) = if n == 0 { (0, 0) } else { vol.chroma_shift() };
        let (height, width) = (16usize >> vsub, 16usize >> hsub);
        let block_mean = gb.get(BITS as usize) as i32;
        if block_mean == 0 {
            return Err(ERR("forbidden block_mean"));
        }
        self.last_dc[n] = block_mean * (1 << (self.dct_precision + self.intra_dc_precision));
        let mut rice_parameter = gb.get(4);
        if rice_parameter == 0 {
            return Err(ERR("forbidden rice_parameter"));
        }
        if rice_parameter == 15 {
            rice_parameter = 0;
        }
        if rice_parameter > 11 {
            return Err(ERR("forbidden rice_parameter"));
        }
        let mask = (1 << BITS) - 1;
        let macroblock = &mut self.dpcm_macroblock[n];
        let mut idx = 0usize;
        for i in 0..height {
            let mut output = 1 << (BITS - 1);
            let mut top = 1 << (BITS - 1);
            for _ in 0..width {
                let left = output;
                let topleft = top;
                let prefix = gb.unary(12);
                let mut residual = if prefix == 11 {
                    gb.get(BITS as usize) as i32
                } else {
                    if prefix == 12 {
                        return Err(ERR("forbidden rice_prefix_code"));
                    }
                    let suffix = gb.get(rice_parameter as usize) as i32;
                    ((prefix as i32) << rice_parameter) + suffix
                };
                residual = if residual & 1 != 0 { (-residual) >> 1 } else { residual >> 1 };
                if i != 0 {
                    top = i32::from(macroblock[idx - width]);
                }
                let min_left_top = left.min(top);
                let max_left_top = left.max(top);
                let p = (left + top - topleft).clamp(min_left_top, max_left_top);
                let mut p2 = (min_left_top.min(topleft) + max_left_top.max(topleft)) >> 1;
                if p2 == p {
                    p2 = block_mean;
                }
                if p2 > p {
                    residual = -residual;
                }
                output = (residual + p) & mask;
                macroblock[idx] = output as u16;
                idx += 1;
            }
        }
        Ok(())
    }

    /// ff_mpeg4_decode_studio: the macroblock into the picture.
    fn reconstruct(&self, picture: &mut Picture, vol: &StudioVol, mb_x: usize, mb_y: usize) {
        let (hsub, vsub) = vol.chroma_shift();
        if self.dpcm_direction == 0 {
            let y0 = (mb_x * 16, mb_y * 16);
            for (n, &(dx, dy)) in [(0, 0), (8, 0), (0, 8), (8, 8)].iter().enumerate() {
                idct_put(&mut picture.planes[0], picture.strides[0], (y0.0 + dx, y0.1 + dy), &self.block32[n]);
            }
            let c0 = ((mb_x * 16) >> hsub, (mb_y * 16) >> vsub);
            // Cb, Cr, Cb, Cr below, then (4:4:4) the right column
            let places: &[(usize, usize, usize)] = &[(1, 0, 0), (2, 0, 0), (1, 0, 8), (2, 0, 8), (1, 8, 0), (2, 8, 0), (1, 8, 8), (2, 8, 8)];
            for (k, &(plane, dx, dy)) in places.iter().take(BLOCK_COUNT[vol.chroma_format as usize] - 4).enumerate() {
                idct_put(&mut picture.planes[plane], picture.strides[plane], (c0.0 + dx, c0.1 + dy), &self.block32[4 + k]);
            }
            return;
        }
        for (i, src) in self.dpcm_macroblock.iter().enumerate() {
            let (hs, vs) = if i == 0 { (0, 0) } else { (hsub, vsub) };
            let (w, h) = (16usize >> hs, 16usize >> vs);
            let (x0, y0) = ((mb_x * 16) >> hs, (mb_y * 16) >> vs);
            let stride = picture.strides[i];
            for row in 0..h {
                for col in 0..w {
                    let (r, c) = if self.dpcm_direction == 1 { (row, col) } else { (h - 1 - row, w - 1 - col) };
                    picture.planes[i][(y0 + r) * stride + x0 + c] = src[row * w + col];
                }
            }
        }
    }
}

/// ff_h263_resync for the studio profile: the next slice start code, at
/// a byte boundary.
fn resync(gb: &mut Bits<'_>) -> bool {
    gb.align();
    while gb.left() >= 32 && gb.show(32) != SLICE_START_CODE {
        gb.skip(8);
    }
    gb.left() >= 32 && gb.show(32) == SLICE_START_CODE
}

fn pixel_format(vol: &StudioVol) -> PixelFormat {
    match (vol.rgb, vol.chroma_format) {
        (true, _) => PixelFormat::Gbrp10Le,
        (false, CHROMA_422) => PixelFormat::Yuv422P10Le,
        (false, _) => PixelFormat::Yuv444P10Le,
    }
}

/// The picture's visible part.
fn crop(picture: Picture, vol: &StudioVol, pts: Option<i64>) -> StudioFrame {
    let (hsub, vsub) = vol.chroma_shift();
    let (w, h) = (vol.width as usize, vol.height as usize);
    let dims = [(w, h), ((w + (1 << hsub) - 1) >> hsub, (h + (1 << vsub) - 1) >> vsub), ((w + (1 << hsub) - 1) >> hsub, (h + (1 << vsub) - 1) >> vsub)];
    let planes = std::array::from_fn(|p| {
        let (pw, ph) = dims[p];
        let stride = picture.strides[p];
        (0..ph).flat_map(|r| picture.planes[p][r * stride..r * stride + pw].iter().copied()).collect()
    });
    StudioFrame {
        width: vol.width,
        height: vol.height,
        pixel_format: pixel_format(vol),
        planes,
        plane_widths: [dims[0].0, dims[1].0, dims[2].0],
        pts,
    }
}

// ───── simple_idct_template.c, BIT_DEPTH 10, IN_IDCT_DEPTH 32 ─────

const W1: u32 = 22725;
const W2: u32 = 21407;
const W3: u32 = 19265;
const W4: u32 = 16384;
const W5: u32 = 12873;
const W6: u32 = 8867;
const W7: u32 = 4520;
const ROW_SHIFT: u32 = 13;
const COL_SHIFT: u32 = 21;

/// idctRowCondDC on one row of 32-bit coefficients, in unsigned
/// arithmetic as FFmpeg's SUINT.
fn idct_row(row: &mut [i32]) {
    let r = |k: usize| row[k] as u32;
    let mut a0 = W4.wrapping_mul(r(0)).wrapping_add(1 << (ROW_SHIFT - 1));
    let (mut a1, mut a2, mut a3) = (a0, a0, a0);
    a0 = a0.wrapping_add(W2.wrapping_mul(r(2)));
    a1 = a1.wrapping_add(W6.wrapping_mul(r(2)));
    a2 = a2.wrapping_sub(W6.wrapping_mul(r(2)));
    a3 = a3.wrapping_sub(W2.wrapping_mul(r(2)));
    let mut b0 = W1.wrapping_mul(r(1)).wrapping_add(W3.wrapping_mul(r(3)));
    let mut b1 = W3.wrapping_mul(r(1)).wrapping_sub(W7.wrapping_mul(r(3)));
    let mut b2 = W5.wrapping_mul(r(1)).wrapping_sub(W1.wrapping_mul(r(3)));
    let mut b3 = W7.wrapping_mul(r(1)).wrapping_sub(W5.wrapping_mul(r(3)));
    if row[4] != 0 || row[5] != 0 || row[6] != 0 || row[7] != 0 {
        a0 = a0.wrapping_add(W4.wrapping_mul(r(4)).wrapping_add(W6.wrapping_mul(r(6))));
        a1 = a1.wrapping_add(0u32.wrapping_sub(W4.wrapping_mul(r(4))).wrapping_sub(W2.wrapping_mul(r(6))));
        a2 = a2.wrapping_add(0u32.wrapping_sub(W4.wrapping_mul(r(4))).wrapping_add(W2.wrapping_mul(r(6))));
        a3 = a3.wrapping_add(W4.wrapping_mul(r(4)).wrapping_sub(W6.wrapping_mul(r(6))));
        b0 = b0.wrapping_add(W5.wrapping_mul(r(5))).wrapping_add(W7.wrapping_mul(r(7)));
        b1 = b1.wrapping_sub(W1.wrapping_mul(r(5))).wrapping_sub(W5.wrapping_mul(r(7)));
        b2 = b2.wrapping_add(W7.wrapping_mul(r(5))).wrapping_add(W3.wrapping_mul(r(7)));
        b3 = b3.wrapping_add(W3.wrapping_mul(r(5))).wrapping_sub(W1.wrapping_mul(r(7)));
    }
    let out = |v: u32| (v as i32) >> ROW_SHIFT;
    row[0] = out(a0.wrapping_add(b0));
    row[7] = out(a0.wrapping_sub(b0));
    row[1] = out(a1.wrapping_add(b1));
    row[6] = out(a1.wrapping_sub(b1));
    row[2] = out(a2.wrapping_add(b2));
    row[5] = out(a2.wrapping_sub(b2));
    row[3] = out(a3.wrapping_add(b3));
    row[4] = out(a3.wrapping_sub(b3));
}

/// idctSparseColPut for column `x` of `block` into `dest`, clipped to 10
/// bits.
fn idct_col_put(block: &[i32; 64], x: usize, dest: &mut [u16], stride: usize, (dx, dy): (usize, usize)) {
    let c = |k: usize| block[8 * k + x] as u32;
    let mut a0 = W4.wrapping_mul(c(0).wrapping_add((1u32 << (COL_SHIFT - 1)) / W4));
    let (mut a1, mut a2, mut a3) = (a0, a0, a0);
    a0 = a0.wrapping_add(W2.wrapping_mul(c(2)));
    a1 = a1.wrapping_add(W6.wrapping_mul(c(2)));
    a2 = a2.wrapping_sub(W6.wrapping_mul(c(2)));
    a3 = a3.wrapping_sub(W2.wrapping_mul(c(2)));
    let mut b0 = W1.wrapping_mul(c(1)).wrapping_add(W3.wrapping_mul(c(3)));
    let mut b1 = W3.wrapping_mul(c(1)).wrapping_sub(W7.wrapping_mul(c(3)));
    let mut b2 = W5.wrapping_mul(c(1)).wrapping_sub(W1.wrapping_mul(c(3)));
    let mut b3 = W7.wrapping_mul(c(1)).wrapping_sub(W5.wrapping_mul(c(3)));
    if block[32 + x] != 0 {
        a0 = a0.wrapping_add(W4.wrapping_mul(c(4)));
        a1 = a1.wrapping_sub(W4.wrapping_mul(c(4)));
        a2 = a2.wrapping_sub(W4.wrapping_mul(c(4)));
        a3 = a3.wrapping_add(W4.wrapping_mul(c(4)));
    }
    if block[40 + x] != 0 {
        b0 = b0.wrapping_add(W5.wrapping_mul(c(5)));
        b1 = b1.wrapping_sub(W1.wrapping_mul(c(5)));
        b2 = b2.wrapping_add(W7.wrapping_mul(c(5)));
        b3 = b3.wrapping_add(W3.wrapping_mul(c(5)));
    }
    if block[48 + x] != 0 {
        a0 = a0.wrapping_add(W6.wrapping_mul(c(6)));
        a1 = a1.wrapping_sub(W2.wrapping_mul(c(6)));
        a2 = a2.wrapping_add(W2.wrapping_mul(c(6)));
        a3 = a3.wrapping_sub(W6.wrapping_mul(c(6)));
    }
    if block[56 + x] != 0 {
        b0 = b0.wrapping_add(W7.wrapping_mul(c(7)));
        b1 = b1.wrapping_sub(W5.wrapping_mul(c(7)));
        b2 = b2.wrapping_add(W3.wrapping_mul(c(7)));
        b3 = b3.wrapping_sub(W1.wrapping_mul(c(7)));
    }
    let values = [
        a0.wrapping_add(b0),
        a1.wrapping_add(b1),
        a2.wrapping_add(b2),
        a3.wrapping_add(b3),
        a3.wrapping_sub(b3),
        a2.wrapping_sub(b2),
        a1.wrapping_sub(b1),
        a0.wrapping_sub(b0),
    ];
    for (k, v) in values.into_iter().enumerate() {
        dest[(dy + k) * stride + dx + x] = ((v as i32) >> COL_SHIFT).clamp(0, (1 << BITS) - 1) as u16;
    }
}

/// ff_simple_idct_put_int32_10bit: rows, then columns put at `at`.
fn idct_put(dest: &mut [u16], stride: usize, at: (usize, usize), block: &[i32; 64]) {
    let mut block = *block;
    for row in block.chunks_exact_mut(8) {
        idct_row(row);
    }
    for x in 0..8 {
        idct_col_put(&block, x, dest, stride, at);
    }
}
