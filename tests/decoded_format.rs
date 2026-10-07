//! The registry decoder reports the visible size and pixel layout of the
//! frame it last returned (oxideav-core `Decoder::output_video_dimensions`
//! / `output_pixel_format`): the VOL's `video_object_layer_width` ×
//! `_height`, odd sizes included, not the macroblock-padded planes. A new
//! VOL's size is reported with the first frame decoded under it, not when
//! the VOL arrives while earlier frames still wait in the reorder slot.
//!
//! `fixtures/decoded_format/*.m4v` are four-frame FFmpeg `mpeg4` streams
//! (`testsrc`, `-bf 2`: I B B P); `sh_128x96.h263` is a three-picture
//! short-header (H.263 baseline) stream.

use oxideav_core::{CodecParameters, Decoder, Error, Frame, Packet, PixelFormat, TimeBase};

const A: &[u8] = include_bytes!("fixtures/decoded_format/a_33x17.m4v");
const B: &[u8] = include_bytes!("fixtures/decoded_format/b_48x32.m4v");
const C: &[u8] = include_bytes!("fixtures/decoded_format/c_35x19.m4v");
const SHORT_HEADER: &[u8] = include_bytes!("fixtures/decoded_format/sh_128x96.h263");

fn decoder() -> Box<dyn Decoder> {
    oxideav_mpeg4video::make_decoder(&CodecParameters::video("mpeg4video".into())).expect("decoder")
}

fn report(dec: &dyn Decoder) -> Option<(u32, u32, PixelFormat)> {
    let (w, h) = dec.output_video_dimensions()?;
    Some((w, h, dec.output_pixel_format()?))
}

/// Receives every available frame; right after each, the report must be
/// that frame's size, and the planes must hold at least that much 4:2:0.
fn receive_all(dec: &mut dyn Decoder, expected: &[(u32, u32)], seen: &mut usize) {
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(frame)) => {
                let (w, h) = expected[*seen];
                assert_eq!(
                    report(dec),
                    Some((w, h, PixelFormat::Yuv420P)),
                    "report after frame {seen}"
                );
                let planes = frame.image_planes();
                assert_eq!(planes.len(), 3, "frame {seen}: planes");
                for (i, plane) in planes.iter().enumerate() {
                    let (pw, ph) = if i == 0 {
                        (w, h)
                    } else {
                        (w.div_ceil(2), h.div_ceil(2))
                    };
                    assert!(
                        plane.stride >= pw as usize,
                        "frame {seen}: plane {i} stride"
                    );
                    assert!(
                        plane.data.len() >= plane.stride * ph as usize,
                        "frame {seen}: plane {i} rows"
                    );
                }
                *seen += 1;
            }
            Ok(_) => panic!("non-video frame"),
            Err(Error::NeedMore | Error::Eof) => return,
            Err(e) => panic!("receive: {e}"),
        }
    }
}

/// Three VOLs of different sizes in one packet: every VOL is parsed
/// before the first frame comes out.
#[test]
fn each_frame_reports_its_own_vol_size() {
    let expected: Vec<(u32, u32)> = [(33, 17), (48, 32), (35, 19)]
        .iter()
        .flat_map(|&size| std::iter::repeat(size).take(4))
        .collect();
    let mut dec = decoder();
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), [A, B, C].concat()))
        .expect("send");
    assert_eq!(
        report(&*dec),
        Some((33, 17, PixelFormat::Yuv420P)),
        "report before the first frame"
    );
    let mut seen = 0;
    receive_all(&mut *dec, &expected, &mut seen);
    dec.flush().expect("flush");
    receive_all(&mut *dec, &expected, &mut seen);
    assert_eq!(seen, expected.len(), "frames");
}

/// One start-code unit per packet, receiving after each: a VOL arriving
/// while the previous VOL's anchor waits for display does not change the
/// report before that anchor comes out.
#[test]
fn reports_change_with_the_frame_that_carries_the_change() {
    let expected: Vec<(u32, u32)> = [(33, 17), (48, 32)]
        .iter()
        .flat_map(|&size| std::iter::repeat(size).take(4))
        .collect();
    let stream = [A, B].concat();
    let starts: Vec<usize> = (0..stream.len().saturating_sub(3))
        .filter(|&i| stream[i..i + 3] == [0, 0, 1])
        .collect();
    let mut dec = decoder();
    let mut seen = 0;
    for (k, &start) in starts.iter().enumerate() {
        let end = starts.get(k + 1).copied().unwrap_or(stream.len());
        dec.send_packet(&Packet::new(
            0,
            TimeBase::new(1, 25),
            stream[start..end].to_vec(),
        ))
        .expect("send");
        receive_all(&mut *dec, &expected, &mut seen);
    }
    dec.flush().expect("flush");
    receive_all(&mut *dec, &expected, &mut seen);
    assert_eq!(seen, expected.len(), "frames");
}

/// A short-header stream has no VOL: the size is the picture's source
/// format (sub-QCIF here).
#[test]
fn short_header_pictures_report_their_source_format() {
    let mut dec = decoder();
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), SHORT_HEADER.to_vec()))
        .expect("send");
    dec.flush().expect("flush");
    let expected = [(128, 96); 3];
    let mut seen = 0;
    receive_all(&mut *dec, &expected, &mut seen);
    assert_eq!(seen, 3, "frames");
}
