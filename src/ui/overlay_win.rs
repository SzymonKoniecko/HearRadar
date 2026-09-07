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
use crate::ui::radar_egui::{draw_radar, RadarCanvas, RadarState};
use egui::{Align2, Color32, FontId, Pos2, Rect, Stroke};
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
        let sk_stroke = tiny_skia::Stroke { width: stroke.width, ..Default::default() };
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
        let sk_stroke = tiny_skia::Stroke { width: stroke.width, ..Default::default() };
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
        self.pixmap
            .borrow_mut()
            .fill_rect(sk_rect, &paint, Transform::identity(), None);
    }

    fn text(&self, _pos: Pos2, _anchor: Align2, _text: &str, _font: FontId, _color: Color32) {
        // tiny-skia nie ma silnika tekstu — etykiety kierunków (opcjonalne,
        // `RadarConfig::show_labels`) są tu po prostu pomijane. Prototype:
        // do rozważenia w przyszłości (np. przez `ab_glyph`/`fontdue`),
        // jeśli etykiety na overlay Windows okażą się potrzebne.
    }
}

/// Natywne okno warstwowe (`WS_EX_LAYERED`) z radarem, sterowane bezpośrednio
/// przez WinAPI zamiast przez viewport `eframe`. Tworzone leniwie, gdy
/// użytkownik otwiera overlay, i niszczone (`Drop`) gdy go zamyka.
pub struct WinOverlay {
    hwnd: HWND,
    mem_dc: HDC,
    bitmap: HBITMAP,
    pixels: *mut u8,
    size: i32,
    visible: bool,
    click_through: bool,
}

impl WinOverlay {
    pub fn new() -> windows::core::Result<Self> {
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

        let mut overlay = Self {
            hwnd,
            mem_dc,
            bitmap: HBITMAP::default(),
            pixels: std::ptr::null_mut(),
            size: 0,
            visible: false,
            click_through: true,
        };
        overlay.resize(1)?;
        Ok(overlay)
    }

    fn resize(&mut self, size: i32) -> windows::core::Result<()> {
        let size = size.max(1);
        if size == self.size && !self.bitmap.is_invalid() {
            return Ok(());
        }

        let mut bmi = BITMAPINFO::default();
        bmi.bmiHeader.biSize = std::mem::size_of::<BITMAPINFOHEADER>() as u32;
        bmi.bmiHeader.biWidth = size;
        // Ujemna wysokość = bitmapa "top-down" (wiersz 0 na górze), tak jak
        // `tiny_skia::Pixmap` oczekuje przy pisaniu po bajtach.
        bmi.bmiHeader.biHeight = -size;
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
        self.size = size;
        Ok(())
    }

    pub fn set_click_through(&mut self, click_through: bool) {
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

    /// Oblicza lewy-górny róg okna z `anchor_x`/`anchor_y` (0.0..1.0 =
    /// ułamek szerokości/wysokości głównego monitora) tak samo jak
    /// `HearRadarApp::overlay_anchored_position` dla ścieżki `eframe`.
    fn position_from_anchor(overlay_size: f32, anchor_x: f32, anchor_y: f32, margin: f32) -> (i32, i32) {
        let screen_w = unsafe { GetSystemMetrics(SM_CXSCREEN) } as f32;
        let screen_h = unsafe { GetSystemMetrics(SM_CYSCREEN) } as f32;
        let usable_w = (screen_w - overlay_size - 2.0 * margin).max(0.0);
        let usable_h = (screen_h - overlay_size - 2.0 * margin).max(0.0);
        let x = margin + anchor_x * usable_w;
        let y = margin + anchor_y * usable_h;
        (x.round() as i32, y.round() as i32)
    }

    /// Rysuje jedną klatkę radaru i wysyła ją na pulpit. Wołane co klatkę
    /// `eframe` (patrz `HearRadarApp::tick_native_overlay`), analogicznie do
    /// `ui_overlay_contents` w ścieżce nie-Windowsowej.
    pub fn render_and_present(
        &mut self,
        overlay_size: f32,
        screen_margin: f32,
        cfg: &RadarConfig,
        state: &mut RadarState,
        channels: usize,
        now: f64,
        click_through: bool,
    ) {
        let size = overlay_size.round() as i32;
        if self.resize(size).is_err() {
            return;
        }

        self.set_click_through(click_through);

        let len = (self.size as usize) * (self.size as usize) * 4;
        // Bezpieczne: `self.pixels` wskazuje na bufor DIB o dokładnie takiej
        // długości, żywy tak długo jak `self.bitmap` (odtwarzany w `resize`
        // razem z tym wskaźnikiem).
        let bytes = unsafe { std::slice::from_raw_parts_mut(self.pixels, len) };
        let Some(mut pixmap) = PixmapMut::from_bytes(bytes, self.size as u32, self.size as u32)
        else {
            return;
        };
        pixmap.fill(tiny_skia::Color::TRANSPARENT);

        let canvas = SkiaCanvas { pixmap: RefCell::new(pixmap) };
        let center = Pos2::new(self.size as f32 / 2.0, self.size as f32 / 2.0);
        draw_radar(&canvas, center, cfg, state, channels, now);

        let (x, y) = Self::position_from_anchor(overlay_size, cfg.anchor_x, cfg.anchor_y, screen_margin);

        unsafe {
            let _ = SetWindowPos(
                self.hwnd,
                None,
                x,
                y,
                self.size,
                self.size,
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
        let win_size = windows::Win32::Foundation::SIZE { cx: self.size, cy: self.size };
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
}

impl Drop for WinOverlay {
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
