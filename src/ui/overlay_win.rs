//! Natywny, przezroczysty overlay dla Windows — omija transparency
//! viewportów `eframe` (niewiarygodne na Windows z backendem `glow`
//! *i* `wgpu`, patrz historia w `ui::app`), rysując radar programowo
//! przez `tiny-skia` na wprost do bufora DIB i wysyłając go na pulpit
//! przez `UpdateLayeredWindow`. To standardowa technika nakładek/HUD-ów
//! na Windows: kompozycja idzie przez zwykłą bitmapę z premultiplikowaną
//! alfą, bez negocjacji trybu alfa z żadnym swapchainem GPU.
//!
//! Okno jest tworzone na wątku głównym (tym samym, na którym działa
//! pętla zdarzeń `winit`/`eframe`) — Windows i tak dostarcza komunikaty
//! do zarejestrowanego `WNDPROC` niezależnie od tego, kto pompuje kolejkę
//! wątku, więc nie potrzebujemy tu własnej pętli `PeekMessage`.

use crate::sound::devices::RadarConfig;
use crate::ui::radar_egui::{draw_edge_glow_indicators, draw_radar, RadarCanvas, RadarState};
use egui::{Align2, Color32, FontId, Pos2, Rect, Stroke, Vec2};
use std::cell::RefCell;
use std::sync::Once;
use tiny_skia::{FillRule, Paint, PathBuilder, PixmapMut, Transform};
use windows::core::PCWSTR;
use windows::Win32::Foundation::{HINSTANCE, HWND, LPARAM, LRESULT, WPARAM};
use windows::Win32::Graphics::Gdi::{
    CreateCompatibleDC, CreateDIBSection, DeleteDC, DeleteObject, GetDC, ReleaseDC, SelectObject,
    AC_SRC_ALPHA, AC_SRC_OVER, BITMAPINFO, BITMAPINFOHEADER, BI_RGB, BLENDFUNCTION,
    DIB_RGB_COLORS, HBITMAP, HDC,
};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::Input::KeyboardAndMouse::ReleaseCapture;
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DestroyWindow, GetSystemMetrics, GetWindowLongPtrW,
    LoadCursorW, RegisterClassExW, SendMessageW, SetWindowLongPtrW, SetWindowPos, ShowWindow,
    UpdateLayeredWindow, CS_HREDRAW, CS_VREDRAW, GWL_EXSTYLE, HTCAPTION, IDC_ARROW,
    SET_WINDOW_POS_FLAGS, SM_CXSCREEN, SM_CYSCREEN, SWP_NOACTIVATE, SWP_NOZORDER, SW_HIDE,
    SW_SHOWNOACTIVATE, ULW_ALPHA, WM_LBUTTONDOWN, WM_NCLBUTTONDOWN, WNDCLASSEXW, WS_EX_LAYERED,
    WS_EX_TOOLWINDOW, WS_EX_TOPMOST, WS_EX_TRANSPARENT, WS_POPUP,
};

const CLASS_NAME: &str = "HearRadarOverlayWndClass";

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    // Okno nie ma paska tytułu (WS_POPUP) — to jedyny sposób, żeby dało się
    // je przeciągnąć: przekazujemy kliknięcie tak, jakby trafiło w pasek
    // tytułu. Wywoływane tylko gdy overlay NIE jest w trybie click-through
    // (wtedy ma WS_EX_TRANSPARENT i w ogóle nie dostaje zdarzeń myszy).
    if msg == WM_LBUTTONDOWN {
        unsafe {
            let _ = ReleaseCapture();
            let _ = SendMessageW(
                hwnd,
                WM_NCLBUTTONDOWN,
                Some(WPARAM(HTCAPTION as usize)),
                Some(LPARAM(0)),
            );
        }
        return LRESULT(0);
    }
    unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) }
}

fn ensure_class_registered(hinstance: HINSTANCE) {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let class_name = wide(CLASS_NAME);
        let cursor = unsafe { LoadCursorW(None, IDC_ARROW) }.unwrap_or_default();
        let wc = WNDCLASSEXW {
            cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
            style: CS_HREDRAW | CS_VREDRAW,
            lpfnWndProc: Some(wndproc),
            hInstance: hinstance,
            hCursor: cursor,
            lpszClassName: PCWSTR(class_name.as_ptr()),
            ..Default::default()
        };
        // RegisterClassExW kopiuje nazwę klasy do wewnętrznego rejestru
        // systemowego, więc `class_name` może bezpiecznie zniknąć po tym
        // wywołaniu.
        unsafe {
            RegisterClassExW(&wc);
        }
    });
}

/// Cienki adapter `RadarCanvas` -> `tiny_skia::PixmapMut`, żeby `draw_radar`
/// (patrz `ui::radar_egui`) rysował identyczną tarczę radaru niezależnie od
/// tego, czy celem jest `egui::Painter`, czy ten bufor programowy.
///
/// Zamiana kanałów R i B przy każdym kolorze to celowy trik: bufor DIB na
/// Windows dla 32-bpp oczekuje kolejności bajtów BGRA, a `tiny-skia`
/// zapisuje swój "RGBA" pod tymi samymi offsetami — więc podając jej R i B
/// zamienione miejscami, dostajemy poprawny BGRA bez dodatkowego
/// kopiowania/swizzlingu całej klatki.
struct SkiaCanvas<'a> {
    pixmap: RefCell<PixmapMut<'a>>,
}

/// `tiny-skia` routes any stroke with `width <= 1.0` (under an identity
/// transform) through its anti-aliased "hairline" scanline filler
/// (`treat_as_hairline` in `painter.rs`) instead of the normal stroke-to-fill
/// path. That filler has a fixed-point edge case (`scan/hairline_aa.rs`)
/// where an axis-aligned line landing on certain pixel-boundary fractions
/// computes a degenerate zero-width inner span and hits
/// `debug_assert!(false)` — crashing debug builds outright. `draw_radar`
/// draws mostly sub-1px, often perfectly axis-aligned strokes (the cross,
/// thin rings), so this was reliably reachable. Nudging just above the
/// hairline threshold keeps every stroke on the ordinary, bug-free path.
fn hairline_safe_width(width: f32) -> f32 {
    width.max(1.05)
}

fn skia_paint(color: Color32) -> Paint<'static> {
    let mut paint = Paint::default();
    paint.set_color_rgba8(color.b(), color.g(), color.r(), color.a());
    paint.anti_alias = true;
    paint
}

impl RadarCanvas for SkiaCanvas<'_> {
    fn circle_filled(&self, center: Pos2, radius: f32, color: Color32) {
        if radius <= 0.0 || color.a() == 0 {
            return;
        }
        let mut pb = PathBuilder::new();
        pb.push_circle(center.x, center.y, radius);
        let Some(path) = pb.finish() else { return };
        let paint = skia_paint(color);
        self.pixmap
            .borrow_mut()
            .fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
    }

    fn circle_stroke(&self, center: Pos2, radius: f32, stroke: Stroke) {
        if radius <= 0.0 || stroke.width <= 0.0 || stroke.color.a() == 0 {
            return;
        }
        let mut pb = PathBuilder::new();
        pb.push_circle(center.x, center.y, radius);
        let Some(path) = pb.finish() else { return };
        let paint = skia_paint(stroke.color);
        let sk_stroke = tiny_skia::Stroke { width: hairline_safe_width(stroke.width), ..Default::default() };
        self.pixmap
            .borrow_mut()
            .stroke_path(&path, &paint, &sk_stroke, Transform::identity(), None);
    }

    fn line_segment(&self, points: [Pos2; 2], stroke: Stroke) {
        if stroke.width <= 0.0 || stroke.color.a() == 0 {
            return;
        }
        let mut pb = PathBuilder::new();
        pb.move_to(points[0].x, points[0].y);
        pb.line_to(points[1].x, points[1].y);
        let Some(path) = pb.finish() else { return };
        let paint = skia_paint(stroke.color);
        let sk_stroke = tiny_skia::Stroke { width: hairline_safe_width(stroke.width), ..Default::default() };
        self.pixmap
            .borrow_mut()
            .stroke_path(&path, &paint, &sk_stroke, Transform::identity(), None);
    }

    fn rect_filled(&self, rect: Rect, _corner_radius: f32, color: Color32) {
        if color.a() == 0 {
            return;
        }
        let Some(sk_rect) =
            tiny_skia::Rect::from_xywh(rect.min.x, rect.min.y, rect.width(), rect.height())
        else {
            return;
        };
        let paint = skia_paint(color);
        // `PixmapMut::fill_rect` with AA and an identity transform (both true
        // here) routes through tiny-skia's `scan::hairline_aa::fill_dot8`,
        // which has a `debug_assert!(false)` that fires whenever a thin rect
        // (e.g. a sub-pixel-wide meter bar, like the LFE level indicator)
        // rounds to a zero-width inner pixel span in fixed point — the same
        // class of bug as `hairline_safe_width` above, but in the fill path
        // rather than the stroke path. Filling the rect as a path instead
        // goes through the ordinary (unaffected) path scan converter, same
        // as `circle_filled`/`triangle_filled` below.
        let mut pb = PathBuilder::new();
        pb.push_rect(sk_rect);
        let Some(path) = pb.finish() else { return };
        self.pixmap
            .borrow_mut()
            .fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
    }

    fn triangle_filled(&self, points: [Pos2; 3], color: Color32) {
        if color.a() == 0 {
            return;
        }
        let mut pb = PathBuilder::new();
        pb.move_to(points[0].x, points[0].y);
        pb.line_to(points[1].x, points[1].y);
        pb.line_to(points[2].x, points[2].y);
        pb.close();
        let Some(path) = pb.finish() else { return };
        let paint = skia_paint(color);
        self.pixmap
            .borrow_mut()
            .fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
    }

    fn text(&self, _pos: Pos2, _anchor: Align2, _text: &str, _font: FontId, _color: Color32) {
        // tiny-skia nie ma silnika tekstu — tekst (np. litery L/R przy łukach)
        // jest tu po prostu pomijany. Do rozważenia w przyszłości (np. przez
        // `ab_glyph`/`fontdue`), jeśli etykiety na overlay Windows okażą się potrzebne.
    }
}

/// Jedno natywne okno warstwowe (`WS_EX_LAYERED`) z własnym buforem DIB. Overlay składa się
/// z kilku takich okien: małego wokół tarczy radaru i wąskich pasków przy krawędziach ekranu na
/// poświatę krawędzi (patrz [`WinOverlay`]).
struct LayeredWindow {
    hwnd: HWND,
    mem_dc: HDC,
    bitmap: HBITMAP,
    pixels: *mut u8,
    width: i32,
    height: i32,
    visible: bool,
    click_through: bool,
}

impl LayeredWindow {
    fn new() -> windows::core::Result<Self> {
        let hinstance: HINSTANCE = unsafe { GetModuleHandleW(None)?.into() };
        ensure_class_registered(hinstance);

        let class_name = wide(CLASS_NAME);
        let title = wide("HearRadar Overlay");
        let hwnd = unsafe {
            CreateWindowExW(
                WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
                PCWSTR(class_name.as_ptr()),
                PCWSTR(title.as_ptr()),
                WS_POPUP,
                0,
                0,
                1,
                1,
                None,
                None,
                Some(hinstance),
                None,
            )?
        };

        let screen_dc = unsafe { GetDC(None) };
        let mem_dc = unsafe { CreateCompatibleDC(Some(screen_dc)) };
        unsafe {
            ReleaseDC(None, screen_dc);
        }

        let mut window = Self {
            hwnd,
            mem_dc,
            bitmap: HBITMAP::default(),
            pixels: std::ptr::null_mut(),
            width: 0,
            height: 0,
            visible: false,
            // Okno powstaje bez `WS_EX_TRANSPARENT`, więc stan startowy to „nie przepuszcza
            // kliknięć”. Dzięki temu pierwsze `set_click_through(true)` naprawdę ustawia styl.
            click_through: false,
        };
        window.resize(1, 1)?;
        Ok(window)
    }

    fn resize(&mut self, width: i32, height: i32) -> windows::core::Result<()> {
        let width = width.max(1);
        let height = height.max(1);
        if width == self.width && height == self.height && !self.bitmap.is_invalid() {
            return Ok(());
        }

        let mut bmi = BITMAPINFO::default();
        bmi.bmiHeader.biSize = std::mem::size_of::<BITMAPINFOHEADER>() as u32;
        bmi.bmiHeader.biWidth = width;
        // Ujemna wysokość = bitmapa "top-down" (wiersz 0 na górze), tak jak
        // `tiny_skia::Pixmap` oczekuje przy pisaniu po bajtach.
        bmi.bmiHeader.biHeight = -height;
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 32;
        bmi.bmiHeader.biCompression = BI_RGB.0;

        let mut pixels: *mut core::ffi::c_void = std::ptr::null_mut();
        let bitmap = unsafe {
            CreateDIBSection(Some(self.mem_dc), &bmi, DIB_RGB_COLORS, &mut pixels, None, 0)?
        };

        if !self.bitmap.is_invalid() {
            unsafe {
                let _ = DeleteObject(self.bitmap.into());
            }
        }
        unsafe {
            SelectObject(self.mem_dc, bitmap.into());
        }

        self.bitmap = bitmap;
        self.pixels = pixels as *mut u8;
        self.width = width;
        self.height = height;
        Ok(())
    }

    fn set_click_through(&mut self, click_through: bool) {
        if self.click_through == click_through {
            return;
        }
        self.click_through = click_through;
        let mut ex_style = unsafe { GetWindowLongPtrW(self.hwnd, GWL_EXSTYLE) };
        let transparent_bit = WS_EX_TRANSPARENT.0 as isize;
        if click_through {
            ex_style |= transparent_bit;
        } else {
            ex_style &= !transparent_bit;
        }
        unsafe {
            SetWindowLongPtrW(self.hwnd, GWL_EXSTYLE, ex_style);
        }
    }

    /// Rysuje klatkę `width`x`height` przez `draw` (na czystym, przezroczystym buforze) i wysyła ją
    /// na pulpit z lewym-górnym rogiem w `(x, y)`.
    fn draw_and_present(&mut self, x: i32, y: i32, width: i32, height: i32, draw: impl FnOnce(&SkiaCanvas<'_>)) {
        if self.resize(width, height).is_err() {
            return;
        }
        let len = (self.width as usize) * (self.height as usize) * 4;
        // Bezpieczne: `self.pixels` wskazuje na bufor DIB o dokładnie takiej
        // długości, żywy tak długo jak `self.bitmap` (odtwarzany w `resize`
        // razem z tym wskaźnikiem).
        let bytes = unsafe { std::slice::from_raw_parts_mut(self.pixels, len) };
        let Some(mut pixmap) = PixmapMut::from_bytes(bytes, self.width as u32, self.height as u32)
        else {
            return;
        };
        pixmap.fill(tiny_skia::Color::TRANSPARENT);
        draw(&SkiaCanvas { pixmap: RefCell::new(pixmap) });
        self.present(x, y);
    }

    fn present(&mut self, x: i32, y: i32) {
        unsafe {
            let _ = SetWindowPos(
                self.hwnd,
                None,
                x,
                y,
                self.width,
                self.height,
                SWP_NOACTIVATE | SWP_NOZORDER,
            );
        }

        if !self.visible {
            unsafe {
                let _ = ShowWindow(self.hwnd, SW_SHOWNOACTIVATE);
            }
            self.visible = true;
        }

        let src_pos = windows::Win32::Foundation::POINT { x: 0, y: 0 };
        let dst_pos = windows::Win32::Foundation::POINT { x, y };
        let win_size = windows::Win32::Foundation::SIZE { cx: self.width, cy: self.height };
        let blend = BLENDFUNCTION {
            BlendOp: AC_SRC_OVER as u8,
            BlendFlags: 0,
            SourceConstantAlpha: 255,
            AlphaFormat: AC_SRC_ALPHA as u8,
        };
        unsafe {
            let _ = UpdateLayeredWindow(
                self.hwnd,
                None,
                Some(&dst_pos),
                Some(&win_size),
                Some(self.mem_dc),
                Some(&src_pos),
                windows::Win32::Foundation::COLORREF(0),
                Some(&blend),
                ULW_ALPHA,
            );
        }
    }

    fn hide(&mut self) {
        if self.visible {
            unsafe {
                let _ = ShowWindow(self.hwnd, SW_HIDE);
            }
            self.visible = false;
        }
    }
}

impl Drop for LayeredWindow {
    fn drop(&mut self) {
        unsafe {
            let _ = ShowWindow(self.hwnd, SW_HIDE);
            if !self.bitmap.is_invalid() {
                let _ = DeleteObject(self.bitmap.into());
            }
            let _ = DeleteDC(self.mem_dc);
            let _ = DestroyWindow(self.hwnd);
        }
    }
}

/// Jeden wypełniony prostokąt poświaty krawędzi w układzie całego ekranu: (prostokąt, zaokrąglenie,
/// kolor). Poświata krawędzi rysuje wyłącznie prostokąty (patrz `radar_egui::draw_edge_glow_indicators`).
type GlowRect = (Rect, f32, Color32);

/// Płótno, które tylko zapamiętuje prostokąty poświaty krawędzi, zamiast rysować je na pełnym ekranie.
/// Inne kształty pomija, bo poświata krawędzi ich nie używa.
#[derive(Default)]
struct GlowRecorder {
    rects: RefCell<Vec<GlowRect>>,
}

impl RadarCanvas for GlowRecorder {
    fn circle_filled(&self, _: Pos2, _: f32, _: Color32) {}
    fn circle_stroke(&self, _: Pos2, _: f32, _: Stroke) {}
    fn line_segment(&self, _: [Pos2; 2], _: Stroke) {}
    fn rect_filled(&self, rect: Rect, corner_radius: f32, color: Color32) {
        if color.a() > 0 && rect.width() > 0.0 && rect.height() > 0.0 {
            self.rects.borrow_mut().push((rect, corner_radius, color));
        }
    }
    fn triangle_filled(&self, _: [Pos2; 3], _: Color32) {}
    fn text(&self, _: Pos2, _: Align2, _: &str, _: FontId, _: Color32) {}
}

/// Krawędź ekranu, przy której leży prostokąt poświaty: 0 lewa, 1 prawa, 2 góra, 3 dół. Pionowe pasy
/// (winieta, plama w stereo) dotykają lewej albo prawej krawędzi, więc te sprawdzamy najpierw.
fn glow_edge(rect: Rect, screen: Rect) -> usize {
    const TOUCH: f32 = 1.0;
    if rect.min.x <= screen.min.x + TOUCH {
        0
    } else if rect.max.x >= screen.max.x - TOUCH {
        1
    } else if rect.min.y <= screen.min.y + TOUCH {
        2
    } else {
        3
    }
}

/// O tyle może się zmienić krycie prostokąta (0..255), zanim pasek krawędzi zostanie narysowany
/// ponownie. Poświata gaśnie płynnie co klatkę, a tak mała różnica jest niewidoczna, za to oszczędza
/// większość odświeżeń w czasie zanikania.
const GLOW_ALPHA_TOLERANCE: u8 = 3;

/// Czy dwie klatki paska krawędzi wyglądają tak samo: te same prostokąty i kolory, krycie najwyżej
/// o [`GLOW_ALPHA_TOLERANCE`] inne.
fn glow_frames_match(a: &[GlowRect], b: &[GlowRect]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|((ra, za, ca), (rb, zb, cb))| {
            ra == rb && za == zb && ca.a().abs_diff(cb.a()) <= GLOW_ALPHA_TOLERANCE && {
                // Kolory są wstępnie pomnożone przez alfę, więc porównujemy barwę po przeskalowaniu.
                let hue = |c: &Color32| c.to_srgba_unmultiplied();
                let (ha, hb) = (hue(ca), hue(cb));
                ha[0].abs_diff(hb[0]) <= 8 && ha[1].abs_diff(hb[1]) <= 8 && ha[2].abs_diff(hb[2]) <= 8
            }
        })
}

/// Pasek przy jednej krawędzi ekranu z poświatą. Okno ma rozmiar dokładnie obejmujący prostokąty
/// poświaty tej krawędzi i odświeża się tylko, gdy obraz naprawdę się zmienił.
struct EdgeStrip {
    window: LayeredWindow,
    /// Prostokąty ostatnio wysłanej klatki, w układzie ekranu.
    shown: Vec<GlowRect>,
}

impl EdgeStrip {
    fn show(&mut self, screen: Rect, rects: Vec<GlowRect>) {
        if self.window.visible && glow_frames_match(&self.shown, &rects) {
            return;
        }
        let bounds = rects
            .iter()
            .fold(Rect::NOTHING, |acc, (r, _, _)| acc.union(*r))
            .intersect(screen);
        let x = bounds.min.x.floor() as i32;
        let y = bounds.min.y.floor() as i32;
        let width = (bounds.max.x.ceil() as i32 - x).max(1);
        let height = (bounds.max.y.ceil() as i32 - y).max(1);
        let offset = Vec2::new(x as f32, y as f32);
        self.window.draw_and_present(x, y, width, height, |canvas| {
            for (rect, corner_radius, color) in &rects {
                canvas.rect_filled(rect.translate(-offset), *corner_radius, *color);
            }
        });
        self.shown = rects;
    }

    fn hide(&mut self) {
        self.window.hide();
        self.shown.clear();
    }
}

/// Natywny overlay sterowany bezpośrednio przez WinAPI zamiast przez viewport `eframe`: małe okno
/// wokół tarczy radaru i, przy `edge_glow_enabled`, wąskie paski przy krawędziach ekranu na poświatę.
/// Dawniej poświata krawędzi rozciągała całe okno na monitor i co klatkę wysyłała pełnoekranowy bufor,
/// co zabierało grze dużo klatek i zasłaniało jej kursor. Tworzony leniwie, gdy użytkownik otwiera
/// overlay, i niszczony (`Drop`) gdy go zamyka.
pub struct WinOverlay {
    radar: LayeredWindow,
    /// Paski poświaty przy krawędziach (lewa, prawa, góra, dół), tworzone przy pierwszym użyciu.
    edges: [Option<EdgeStrip>; 4],
}

impl WinOverlay {
    pub fn new() -> windows::core::Result<Self> {
        Ok(Self { radar: LayeredWindow::new()?, edges: [None, None, None, None] })
    }

    /// Rozmiar głównego monitora w pikselach — poświata krawędzi liczy pozycje w układzie
    /// całego ekranu, analogicznie do `monitor_size` z `egui::ViewportInfo` na ścieżce
    /// nie-Windowsowej.
    fn screen_size() -> (i32, i32) {
        unsafe {
            (
                GetSystemMetrics(SM_CXSCREEN),
                GetSystemMetrics(SM_CYSCREEN),
            )
        }
    }

    /// Oblicza lewy-górny róg okna z `anchor_x`/`anchor_y` (0.0..1.0 =
    /// ułamek szerokości/wysokości głównego monitora) tak samo jak
    /// `HearRadarApp::overlay_anchored_position` dla ścieżki `eframe`.
    fn position_from_anchor(overlay_size: f32, anchor_x: f32, anchor_y: f32, margin: f32) -> (i32, i32) {
        let (screen_w, screen_h) = Self::screen_size();
        let usable_w = (screen_w as f32 - overlay_size - 2.0 * margin).max(0.0);
        let usable_h = (screen_h as f32 - overlay_size - 2.0 * margin).max(0.0);
        let x = margin + anchor_x * usable_w;
        let y = margin + anchor_y * usable_h;
        (x.round() as i32, y.round() as i32)
    }

    /// Rysuje jedną klatkę radaru w małym oknie wokół tarczy i, przy `edge_glow_enabled`, poświatę
    /// na krawędzi ekranu w osobnych paskach (patrz `radar_egui::draw_edge_glow_indicators`).
    /// Wołane co klatkę `eframe` (patrz `HearRadarApp::tick_native_overlay`), analogicznie do
    /// `ui_overlay_contents` w ścieżce nie-Windowsowej. `dial_size` to bok okna radaru: średnica
    /// tarczy + margines.
    #[allow(clippy::too_many_arguments)]
    pub fn render_and_present(
        &mut self,
        dial_size: f32,
        screen_margin: f32,
        cfg: &RadarConfig,
        state: &mut RadarState,
        channels: usize,
        now: f64,
        click_through: bool,
    ) {
        let size = dial_size.round() as i32;
        let (x, y) = Self::position_from_anchor(dial_size, cfg.anchor_x, cfg.anchor_y, screen_margin);
        let center = Pos2::new(size as f32 / 2.0, size as f32 / 2.0);
        self.radar.set_click_through(click_through);
        self.radar.draw_and_present(x, y, size, size, |canvas| {
            draw_radar(canvas, center, cfg, state, channels, now);
        });

        if !cfg.edge_glow_enabled {
            self.hide_edges();
            return;
        }
        let (screen_w, screen_h) = Self::screen_size();
        let screen = Rect::from_min_size(Pos2::ZERO, Vec2::new(screen_w as f32, screen_h as f32));
        let recorder = GlowRecorder::default();
        draw_edge_glow_indicators(&recorder, screen, state, now, cfg);
        let mut groups: [Vec<GlowRect>; 4] = Default::default();
        for glow in recorder.rects.into_inner() {
            groups[glow_edge(glow.0, screen)].push(glow);
        }
        for (slot, rects) in self.edges.iter_mut().zip(groups) {
            if rects.is_empty() {
                if let Some(strip) = slot {
                    strip.hide();
                }
                continue;
            }
            if slot.is_none() {
                // Pasek zawsze przepuszcza kliknięcia: nie ma czego przeciągać, a ma nie zasłaniać gry.
                let Ok(mut window) = LayeredWindow::new() else { continue };
                window.set_click_through(true);
                *slot = Some(EdgeStrip { window, shown: Vec::new() });
            }
            if let Some(strip) = slot {
                strip.show(screen, rects);
            }
        }
    }

    fn hide_edges(&mut self) {
        for strip in self.edges.iter_mut().flatten() {
            strip.hide();
        }
    }
}

// SAFETY: HWND/HDC/HBITMAP are just opaque handle values (pointer-sized
// integers as far as the Win32 API contract goes); `WinOverlay` is only ever
// touched from the single UI thread that owns the eframe/winit event loop,
// so no cross-thread synchronization is actually exercised here — this
// merely satisfies fields living inside `HearRadarApp`, which itself has no
// `Send`/`Sync` requirement but derives from types that assume it.
unsafe impl Send for WinOverlay {}

#[allow(unused)]
const _: () = {
    fn _assert_flags_type(_: SET_WINDOW_POS_FLAGS) {}
};

#[cfg(test)]
mod tests {
    use super::*;

    fn screen() -> Rect {
        Rect::from_min_size(Pos2::ZERO, Vec2::new(1920.0, 1080.0))
    }

    #[test]
    fn glow_rects_go_to_the_edge_they_touch() {
        let full_height_left = Rect::from_min_size(Pos2::ZERO, Vec2::new(300.0, 1080.0));
        assert_eq!(glow_edge(full_height_left, screen()), 0, "winieta po lewej, choć dotyka też góry i dołu");
        let right = Rect::from_min_max(Pos2::new(1760.0, 216.0), Pos2::new(1920.0, 864.0));
        assert_eq!(glow_edge(right, screen()), 1);
        let top = Rect::from_min_max(Pos2::new(800.0, 0.0), Pos2::new(1120.0, 160.0));
        assert_eq!(glow_edge(top, screen()), 2);
        let bottom = Rect::from_min_max(Pos2::new(800.0, 920.0), Pos2::new(1120.0, 1080.0));
        assert_eq!(glow_edge(bottom, screen()), 3);
    }

    #[test]
    fn tiny_alpha_changes_do_not_redraw_the_strip() {
        let rect = Rect::from_min_size(Pos2::ZERO, Vec2::new(300.0, 1080.0));
        let frame = |alpha: u8| vec![(rect, 0.0, Color32::from_rgba_unmultiplied(255, 40, 40, alpha))];
        assert!(glow_frames_match(&frame(70), &frame(68)), "różnica 2 w kryciu: bez odświeżenia");
        assert!(!glow_frames_match(&frame(70), &frame(60)), "wyraźnie bledsza: odświeżenie");
        assert!(!glow_frames_match(&frame(70), &[]), "zniknęła: odświeżenie");
        let wider = vec![(Rect::from_min_size(Pos2::ZERO, Vec2::new(320.0, 1080.0)), 0.0, frame(70)[0].2)];
        assert!(!glow_frames_match(&frame(70), &wider), "inny zasięg: odświeżenie");
    }
}
