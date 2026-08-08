//! Generate the capability showcase: process real FITS images (e.g. from a
//! Seestar) through timed pipeline variants and emit a small static site
//! with slider + side-by-side comparisons and a FITS-style processing log.
//!
//! Drop inputs into `showcase/input/`:
//!   - `<target>.fit` / `.fits` / `.fts` — the raw stacked FITS
//!   - `<target>.jpg` (optional)        — the Seestar's own on-device JPEG
//!
//! Usage:
//!     cargo run --release --example showcase [input_dir] [site_dir]
//!
//! Defaults: input `showcase/input`, site `showcase`. Outputs land in
//! `<site_dir>/index.html`, `<site_dir>/<target>.html`, `<site_dir>/img/`,
//! and `<site_dir>/results.json`.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use processinator::stretch::StretchAlgorithm;
use processinator::{process, read_fits, to_dynamic_image, PipelineConfig};

struct VariantSpec {
    key: &'static str,
    /// Caption used in the stage grid (mono, uppercase).
    caption: &'static str,
    /// Label used in selects and the processing log.
    label: &'static str,
    config: PipelineConfig,
}

fn variant_specs() -> Vec<VariantSpec> {
    vec![
        VariantSpec {
            key: "linear",
            caption: "S0 LINEAR",
            label: "linear (no stretch)",
            config: PipelineConfig {
                gradient_removal: false,
                stretch: StretchAlgorithm::Linear {
                    low_percentile: 0.1,
                    high_percentile: 99.99,
                },
                // Raw-data preview: show what the sensor recorded, untouched
                green_removal: 0.0,
                saturation: 1.0,
                ..Default::default()
            },
        },
        VariantSpec {
            key: "stretch",
            caption: "S1 MTF STRETCH",
            label: "MTF stretch",
            config: PipelineConfig {
                gradient_removal: false,
                ..Default::default()
            },
        },
        VariantSpec {
            key: "gradient",
            caption: "S2 + GRADIENT",
            label: "+ gradient removal",
            config: PipelineConfig::default(),
        },
        VariantSpec {
            key: "full",
            caption: "S3 + DENOISE",
            label: "+ starlet denoise",
            config: PipelineConfig {
                denoise: true,
                ..Default::default()
            },
        },
    ]
}

struct VariantResult {
    key: &'static str,
    caption: &'static str,
    label: &'static str,
    pipeline_ms: f64,
    encode_ms: f64,
    /// Site-relative path of the rendered PNG.
    src: String,
}

struct TargetResult {
    name: String,
    slug: String,
    fits_file: String,
    width: usize,
    height: usize,
    channels: usize,
    read_ms: f64,
    /// Site-relative path of the copied Seestar JPEG, when one was found.
    seestar_src: Option<String>,
    variants: Vec<VariantResult>,
}

impl TargetResult {
    fn megapixels(&self) -> f64 {
        (self.width * self.height) as f64 / 1e6
    }

    fn full_ms(&self) -> f64 {
        self.variants
            .iter()
            .find(|v| v.key == "full")
            .map(|v| v.pipeline_ms)
            .unwrap_or(0.0)
    }
}

struct Machine {
    cpu: String,
    threads: usize,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let input_dir = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("showcase/input"));
    let site_dir = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("showcase"));
    let img_dir = site_dir.join("img");
    std::fs::create_dir_all(&input_dir).expect("create input directory");
    std::fs::create_dir_all(&img_dir).expect("create image output directory");

    let machine = detect_machine();
    let fits_files = collect_fits(&input_dir);

    if fits_files.is_empty() {
        println!(
            "No FITS files in {} — generating an empty observation log.",
            input_dir.display()
        );
        println!("Drop a Seestar .fit (and optional matching .jpg) there and re-run.");
    }

    let mut targets = Vec::new();
    for path in &fits_files {
        match process_target(path, &img_dir) {
            Ok(target) => {
                println!(
                    "  {}  {}x{}  read {:.1} ms  full pipeline {:.1} ms",
                    target.name,
                    target.width,
                    target.height,
                    target.read_ms,
                    target.full_ms(),
                );
                targets.push(target);
            }
            Err(err) => eprintln!("  skipping {}: {err}", path.display()),
        }
    }

    for target in &targets {
        let page = render_target_page(target, &machine);
        let path = site_dir.join(format!("{}.html", target.slug));
        std::fs::write(&path, page).expect("write target page");
    }

    let index = render_index(&targets, &machine);
    std::fs::write(site_dir.join("index.html"), index).expect("write index page");

    let results = render_results_json(&targets, &machine);
    std::fs::write(site_dir.join("results.json"), results).expect("write results.json");

    println!(
        "\n{} target{} → {}/index.html",
        targets.len(),
        if targets.len() == 1 { "" } else { "s" },
        site_dir.display()
    );
}

fn collect_fits(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    p.extension()
                        .and_then(|e| e.to_str())
                        .map(|e| {
                            let e = e.to_ascii_lowercase();
                            e == "fit" || e == "fits" || e == "fts"
                        })
                        .unwrap_or(false)
                })
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    files
}

/// Find `<stem>.jpg` / `<stem>.jpeg` next to the FITS file, case-insensitive.
fn find_sibling_jpg(fits_path: &Path) -> Option<PathBuf> {
    let stem = fits_path.file_stem()?.to_str()?.to_ascii_lowercase();
    let dir = fits_path.parent()?;
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .find_map(|e| {
            let p = e.path();
            let ext = p.extension()?.to_str()?.to_ascii_lowercase();
            if ext != "jpg" && ext != "jpeg" {
                return None;
            }
            (p.file_stem()?.to_str()?.to_ascii_lowercase() == stem).then_some(p)
        })
}

fn process_target(fits_path: &Path, img_dir: &Path) -> Result<TargetResult, String> {
    let stem = fits_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "target".to_string());
    let slug = slugify(&stem);

    let t = Instant::now();
    let data = read_fits(fits_path).map_err(|e| e.to_string())?;
    let read_ms = t.elapsed().as_secs_f64() * 1e3;

    let seestar_src = find_sibling_jpg(fits_path).and_then(|jpg| {
        let dest_name = format!("{slug}-seestar.jpg");
        std::fs::copy(&jpg, img_dir.join(&dest_name))
            .map(|_| format!("img/{dest_name}"))
            .map_err(|e| eprintln!("  could not copy {}: {e}", jpg.display()))
            .ok()
    });

    let mut variants = Vec::new();
    for spec in variant_specs() {
        let t = Instant::now();
        let processed = process(data.clone(), &spec.config);
        let img = to_dynamic_image(&processed);
        let pipeline_ms = t.elapsed().as_secs_f64() * 1e3;

        let file_name = format!("{slug}-{}.png", spec.key);
        let t = Instant::now();
        img.save(img_dir.join(&file_name))
            .map_err(|e| e.to_string())?;
        let encode_ms = t.elapsed().as_secs_f64() * 1e3;

        variants.push(VariantResult {
            key: spec.key,
            caption: spec.caption,
            label: spec.label,
            pipeline_ms,
            encode_ms,
            src: format!("img/{file_name}"),
        });
    }

    Ok(TargetResult {
        name: stem,
        slug,
        fits_file: fits_path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default(),
        width: data.width(),
        height: data.height(),
        channels: data.num_channels(),
        read_ms,
        seestar_src,
        variants,
    })
}

fn detect_machine() -> Machine {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let cpu = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|v| v.trim().to_string())
        })
        .unwrap_or_else(|| std::env::consts::ARCH.to_string());
    Machine { cpu, threads }
}

fn slugify(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.is_empty() && !out.ends_with('-') {
            out.push('-');
        }
    }
    let out = out.trim_end_matches('-').to_string();
    if out.is_empty() {
        "target".to_string()
    } else {
        out
    }
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn channels_label(channels: usize) -> &'static str {
    if channels == 3 {
        "RGB"
    } else {
        "mono"
    }
}

// ---------------------------------------------------------------------------
// Page rendering
// ---------------------------------------------------------------------------

const SHARED_CSS: &str = r#"
:root {
  --sky: #0a0e1a;
  --panel: #121828;
  --line: #202b47;
  --text: #d8dee9;
  --muted: #8791a7;
  --amber: #e9a83a;
  --mono: ui-monospace, 'Cascadia Code', 'JetBrains Mono', Menlo, Consolas, monospace;
  --serif: Georgia, 'Times New Roman', serif;
  --sans: system-ui, -apple-system, 'Segoe UI', sans-serif;
}
* { box-sizing: border-box; }
html { scrollbar-color: var(--line) var(--sky); }
body {
  margin: 0;
  background: var(--sky);
  background-image: radial-gradient(ellipse 90% 40% at 30% -5%, #16203c 0%, transparent 60%);
  background-repeat: no-repeat;
  color: var(--text);
  font-family: var(--sans);
  line-height: 1.55;
}
main { max-width: 1060px; margin: 0 auto; padding: 0 24px 48px; }
a { color: var(--amber); }
:focus-visible { outline: 2px solid var(--amber); outline-offset: 2px; }
.masthead { padding: 52px 0 4px; }
.eyebrow {
  font-family: var(--mono);
  font-size: 12px;
  letter-spacing: .18em;
  text-transform: uppercase;
  color: var(--amber);
  margin: 0;
}
h1 {
  font-family: var(--serif);
  font-style: italic;
  font-weight: 400;
  font-size: clamp(28px, 4.5vw, 44px);
  margin: .3em 0 .25em;
}
h2 {
  font-family: var(--mono);
  font-size: 12px;
  font-weight: 400;
  letter-spacing: .16em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 44px 0 14px;
}
.meta { font-family: var(--mono); font-size: 12.5px; color: var(--muted); margin: 0 0 6px; }
.meta b { color: var(--text); font-weight: 400; }
.lede { max-width: 62ch; color: var(--text); margin: 10px 0 0; }
.fitslog {
  font-family: var(--mono);
  font-size: 13px;
  line-height: 1.75;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 16px 20px;
  overflow-x: auto;
  white-space: pre;
}
.fitslog .kw { color: var(--muted); }
.fitslog .val { color: var(--amber); }
footer {
  font-family: var(--mono);
  font-size: 11.5px;
  color: var(--muted);
  border-top: 1px solid var(--line);
  margin-top: 56px;
  padding-top: 18px;
}
footer p { margin: 4px 0; }
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { transition: none !important; animation: none !important; }
}
"#;

const INDEX_CSS: &str = r#"
.cards {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 18px;
  margin-top: 20px;
}
.card {
  display: block;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
  text-decoration: none;
  color: inherit;
  transition: border-color .15s ease;
}
.card:hover { border-color: var(--amber); }
.card img { display: block; width: 100%; aspect-ratio: 3 / 2; object-fit: cover; background: #000; }
.card-caption { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; padding: 10px 14px; }
.card-caption .name { font-family: var(--serif); font-style: italic; font-size: 19px; }
.card-caption .ms { font-family: var(--mono); font-size: 11.5px; color: var(--amber); white-space: nowrap; }
.empty {
  border: 1px dashed var(--line);
  border-radius: 8px;
  padding: 36px 28px;
  margin-top: 20px;
  font-family: var(--mono);
  font-size: 13px;
  color: var(--muted);
  line-height: 2;
}
.empty code { color: var(--amber); }
"#;

const TARGET_CSS: &str = r#"
.back { font-family: var(--mono); font-size: 12px; text-decoration: none; }
.compare { margin-top: 6px; }
.compare-frame {
  position: relative;
  width: 100%;
  overflow: hidden;
  background: #000;
  border: 1px solid var(--line);
  border-radius: 6px;
  touch-action: none;
  cursor: col-resize;
}
.compare-frame img {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
  user-select: none;
  -webkit-user-drag: none;
  pointer-events: none;
}
.compare-divider {
  position: absolute;
  top: 0;
  bottom: 0;
  left: 50%;
  width: 1px;
  background: rgba(233, 168, 58, .85);
  pointer-events: none;
}
.compare-divider::before {
  content: "";
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 46px;
  height: 46px;
  border: 1px solid var(--amber);
  border-radius: 50%;
  background: rgba(10, 14, 26, .55);
}
.compare-divider::after {
  content: "";
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 70px;
  height: 1px;
  background: linear-gradient(90deg,
    var(--amber) 0 12px, transparent 12px calc(100% - 12px),
    var(--amber) calc(100% - 12px));
}
.tag {
  position: absolute;
  top: 10px;
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: .08em;
  padding: 3px 9px;
  background: rgba(10, 14, 26, .72);
  border: 1px solid var(--line);
  border-radius: 3px;
  pointer-events: none;
}
.tag-a { left: 10px; }
.tag-b { right: 10px; }
.compare-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
  margin-top: 10px;
  font-family: var(--mono);
  font-size: 12px;
  color: var(--muted);
}
.compare-controls select {
  background: var(--panel);
  color: var(--text);
  border: 1px solid var(--line);
  border-radius: 4px;
  padding: 6px 8px;
  font-family: var(--mono);
  font-size: 12px;
}
.compare-range { flex: 1; min-width: 160px; accent-color: var(--amber); }
.stage-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 14px;
}
.stage-grid figure { margin: 0; background: var(--panel); border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
.stage-grid a { display: block; line-height: 0; }
.stage-grid img { width: 100%; height: auto; display: block; background: #000; }
.stage-grid figcaption {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  padding: 8px 12px;
  font-family: var(--mono);
  font-size: 11.5px;
  letter-spacing: .06em;
  color: var(--muted);
}
.stage-grid .ms { color: var(--amber); white-space: nowrap; }
"#;

const COMPARE_JS: &str = r#"
document.querySelectorAll('.compare').forEach((root) => {
  const data = JSON.parse(root.dataset.renditions);
  const frame = root.querySelector('.compare-frame');
  const imgA = root.querySelector('.img-a');
  const imgB = root.querySelector('.img-b');
  const tagA = root.querySelector('.tag-a');
  const tagB = root.querySelector('.tag-b');
  const divider = root.querySelector('.compare-divider');
  const selA = root.querySelector('.sel-a');
  const selB = root.querySelector('.sel-b');
  const range = root.querySelector('.compare-range');

  data.forEach((r, i) => {
    selA.add(new Option(r.label, i));
    selB.add(new Option(r.label, i));
  });
  selA.value = root.dataset.defaultA;
  selB.value = root.dataset.defaultB;

  function sync() {
    imgA.src = data[selA.value].src;
    imgB.src = data[selB.value].src;
    tagA.textContent = data[selA.value].label;
    tagB.textContent = data[selB.value].label;
  }
  function setPos(p) {
    p = Math.max(0, Math.min(100, p));
    imgA.style.clipPath = `inset(0 ${100 - p}% 0 0)`;
    divider.style.left = p + '%';
    range.value = p;
  }
  function fromPointer(e) {
    const r = frame.getBoundingClientRect();
    setPos(((e.clientX - r.left) / r.width) * 100);
  }
  frame.addEventListener('pointerdown', (e) => {
    frame.setPointerCapture(e.pointerId);
    fromPointer(e);
  });
  frame.addEventListener('pointermove', (e) => {
    if (e.buttons) fromPointer(e);
  });
  range.addEventListener('input', () => setPos(+range.value));
  selA.addEventListener('change', sync);
  selB.addEventListener('change', sync);
  sync();
  setPos(50);
});
"#;

fn page_shell(title: &str, extra_css: &str, body: &str, script: &str) -> String {
    let mut page = String::new();
    page.push_str("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n");
    page.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
    page.push_str(
        "<link rel=\"icon\" href=\"data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20\
         viewBox='0%200%2016%2016'%3E%3Ccircle%20cx='8'%20cy='8'%20r='4'%20fill='%23e9a83a'/%3E%3C/svg%3E\">\n",
    );
    let _ = writeln!(page, "<title>{}</title>", html_escape(title));
    page.push_str("<style>");
    page.push_str(SHARED_CSS);
    page.push_str(extra_css);
    page.push_str("</style>\n</head>\n<body>\n<main>\n");
    page.push_str(body);
    page.push_str("</main>\n");
    if !script.is_empty() {
        page.push_str("<script>");
        page.push_str(script);
        page.push_str("</script>\n");
    }
    page.push_str("</body>\n</html>\n");
    page
}

fn footer_html(machine: &Machine) -> String {
    format!(
        "<footer>\n<p>SIMPLE = T — generated by <code>cargo run --release --example showcase</code>, \
         processinator {}</p>\n<p>{} &middot; {} threads</p>\n</footer>",
        env!("CARGO_PKG_VERSION"),
        html_escape(&machine.cpu),
        machine.threads,
    )
}

fn render_index(targets: &[TargetResult], machine: &Machine) -> String {
    let mut body = String::new();
    body.push_str("<header class=\"masthead\">\n");
    body.push_str("<p class=\"eyebrow\">processinator &middot; observation log</p>\n");
    body.push_str("<h1>What the telescope saw &mdash; and what the pipeline made of it</h1>\n");
    body.push_str(
        "<p class=\"lede\">processinator is a Rust library that turns linear FITS stacks into \
         finished images: MTF, arcsinh, log, linear, and statistical stretches, polynomial \
         gradient removal, starlet wavelet denoising, and stacking-edge detection, parallelized \
         with rayon. Every image below was processed on this machine, and timed.</p>\n",
    );
    let _ = writeln!(
        body,
        "<p class=\"meta\" style=\"margin-top:14px\">{} &middot; {} threads &middot; release build</p>",
        html_escape(&machine.cpu),
        machine.threads,
    );
    body.push_str("</header>\n");

    if targets.is_empty() {
        body.push_str(
            "<div class=\"empty\">No observations yet.<br>\
             Drop a FITS file (and, if you have it, the matching Seestar JPEG) into \
             <code>showcase/input/</code>, then run:<br>\
             <code>cargo run --release --example showcase</code></div>\n",
        );
    } else {
        body.push_str("<h2>Targets</h2>\n<div class=\"cards\">\n");
        for t in targets {
            let thumb = t
                .variants
                .iter()
                .find(|v| v.key == "full")
                .or_else(|| t.variants.last())
                .map(|v| v.src.as_str())
                .unwrap_or("");
            let _ = writeln!(
                body,
                "<a class=\"card\" href=\"{slug}.html\">\n\
                 <img src=\"{thumb}\" alt=\"{name} processed with the full pipeline\" loading=\"lazy\">\n\
                 <div class=\"card-caption\"><span class=\"name\">{name}</span>\
                 <span class=\"ms\">full pipeline {ms:.1} ms</span></div>\n</a>",
                slug = t.slug,
                thumb = html_escape(thumb),
                name = html_escape(&t.name),
                ms = t.full_ms(),
            );
        }
        body.push_str("</div>\n");
    }

    body.push_str(&footer_html(machine));
    page_shell("processinator — observation log", INDEX_CSS, &body, "")
}

fn render_target_page(target: &TargetResult, machine: &Machine) -> String {
    // Renditions offered in the comparison widget: Seestar JPEG first (when
    // present), then the pipeline variants in stage order.
    let mut renditions: Vec<(String, String)> = Vec::new();
    if let Some(src) = &target.seestar_src {
        renditions.push(("Seestar on-device JPG".to_string(), src.clone()));
    }
    for v in &target.variants {
        renditions.push((v.label.to_string(), v.src.clone()));
    }

    let renditions_json = {
        let items: Vec<String> = renditions
            .iter()
            .map(|(label, src)| {
                format!(
                    "{{\"label\":\"{}\",\"src\":\"{}\"}}",
                    json_escape(label),
                    json_escape(src)
                )
            })
            .collect();
        format!("[{}]", items.join(","))
    };

    // Default comparison: Seestar JPG vs full pipeline when the JPG exists,
    // otherwise linear (raw) vs full pipeline.
    let default_a = 0;
    let default_b = renditions.len().saturating_sub(1);
    let initial_a = &renditions[default_a];
    let initial_b = &renditions[default_b];

    let mut body = String::new();
    body.push_str("<header class=\"masthead\">\n");
    body.push_str("<p><a class=\"back\" href=\"index.html\">&larr; observation log</a></p>\n");
    let _ = writeln!(body, "<h1>{}</h1>", html_escape(&target.name));
    let _ = writeln!(
        body,
        "<p class=\"meta\"><b>{w} &times; {h}</b> &middot; {ch} &middot; {fits} &middot; \
         read in {read:.1} ms</p>",
        w = target.width,
        h = target.height,
        ch = channels_label(target.channels),
        fits = html_escape(&target.fits_file),
        read = target.read_ms,
    );
    body.push_str("</header>\n");

    // --- Interactive comparison -------------------------------------------
    body.push_str("<h2>Compare</h2>\n");
    let _ = writeln!(
        body,
        "<div class=\"compare\" data-renditions='{json}' data-default-a=\"{da}\" data-default-b=\"{db}\">\n\
         <div class=\"compare-frame\" style=\"aspect-ratio: {w} / {h}\">\n\
         <img class=\"img-b\" src=\"{srcb}\" alt=\"{name}, right side of comparison\">\n\
         <img class=\"img-a\" src=\"{srca}\" alt=\"{name}, left side of comparison\">\n\
         <div class=\"compare-divider\"></div>\n\
         <span class=\"tag tag-a\"></span>\n\
         <span class=\"tag tag-b\"></span>\n\
         </div>\n\
         <div class=\"compare-controls\">\n\
         <label>A <select class=\"sel-a\" aria-label=\"Left image\"></select></label>\n\
         <span>vs</span>\n\
         <label>B <select class=\"sel-b\" aria-label=\"Right image\"></select></label>\n\
         <input class=\"compare-range\" type=\"range\" min=\"0\" max=\"100\" value=\"50\" \
         aria-label=\"Comparison divider position\">\n\
         </div>\n</div>",
        json = renditions_json,
        da = default_a,
        db = default_b,
        w = target.width,
        h = target.height,
        srca = html_escape(&initial_a.1),
        srcb = html_escape(&initial_b.1),
        name = html_escape(&target.name),
    );

    // --- Side-by-side stage grid ------------------------------------------
    body.push_str("<h2>Pipeline stages</h2>\n<div class=\"stage-grid\">\n");
    if let Some(src) = &target.seestar_src {
        let _ = writeln!(
            body,
            "<figure><a href=\"{src}\"><img src=\"{src}\" \
             alt=\"{name}, Seestar on-device JPG\" loading=\"lazy\"></a>\
             <figcaption><span>SEESTAR JPG</span><span class=\"ms\">on-device</span></figcaption></figure>",
            src = html_escape(src),
            name = html_escape(&target.name),
        );
    }
    for v in &target.variants {
        let _ = writeln!(
            body,
            "<figure><a href=\"{src}\"><img src=\"{src}\" \
             alt=\"{name}, {label}\" loading=\"lazy\"></a>\
             <figcaption><span>{caption}</span><span class=\"ms\">{ms:.1} ms</span></figcaption></figure>",
            src = html_escape(&v.src),
            name = html_escape(&target.name),
            label = html_escape(v.label),
            caption = html_escape(v.caption),
            ms = v.pipeline_ms,
        );
    }
    body.push_str("</div>\n");

    // --- FITS-style processing log ----------------------------------------
    body.push_str("<h2>Processing log</h2>\n");
    body.push_str(&render_fits_log(target, machine));

    body.push_str(&footer_html(machine));
    page_shell(
        &format!("{} — processinator showcase", target.name),
        TARGET_CSS,
        &body,
        COMPARE_JS,
    )
}

/// The timing readout, styled after the HISTORY/COMMENT cards a FITS header
/// uses to record processing applied to the data.
fn render_fits_log(target: &TargetResult, machine: &Machine) -> String {
    let mut log = String::new();
    log.push_str("<div class=\"fitslog\">");
    let _ = writeln!(
        log,
        "<span class=\"kw\">SIMPLE </span> =                    T / conforms to FITS standard"
    );
    let _ = writeln!(
        log,
        "<span class=\"kw\">COMMENT</span>  processinator {} &middot; release build",
        env!("CARGO_PKG_VERSION")
    );
    let _ = writeln!(
        log,
        "<span class=\"kw\">COMMENT</span>  target: {}  ({} x {}, {})",
        html_escape(&target.name),
        target.width,
        target.height,
        channels_label(target.channels),
    );
    let _ = writeln!(
        log,
        "<span class=\"kw\">HISTORY</span>  {:.<34} <span class=\"val\">{:>8.1} ms</span>",
        "read FITS ", target.read_ms
    );
    for v in &target.variants {
        let _ = writeln!(
            log,
            "<span class=\"kw\">HISTORY</span>  {:.<34} <span class=\"val\">{:>8.1} ms</span>  ({:.1} MP/s)",
            format!("{} ", v.label),
            v.pipeline_ms,
            target.megapixels() / (v.pipeline_ms / 1e3),
        );
    }
    let _ = writeln!(
        log,
        "<span class=\"kw\">COMMENT</span>  encode PNG: {:.1} ms (full-pipeline variant)",
        target
            .variants
            .iter()
            .find(|v| v.key == "full")
            .map(|v| v.encode_ms)
            .unwrap_or(0.0),
    );
    let _ = writeln!(
        log,
        "<span class=\"kw\">COMMENT</span>  cpu: {} ({} threads)",
        html_escape(&machine.cpu),
        machine.threads,
    );
    log.push_str("<span class=\"kw\">END</span>");
    log.push_str("</div>\n");
    log
}

fn render_results_json(targets: &[TargetResult], machine: &Machine) -> String {
    let epoch = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let mut out = String::new();
    out.push_str("{\n");
    let _ = writeln!(out, "  \"generated_at_epoch\": {epoch},");
    let _ = writeln!(
        out,
        "  \"processinator_version\": \"{}\",",
        json_escape(env!("CARGO_PKG_VERSION"))
    );
    let _ = writeln!(
        out,
        "  \"machine\": {{ \"cpu\": \"{}\", \"threads\": {} }},",
        json_escape(&machine.cpu),
        machine.threads
    );
    out.push_str("  \"targets\": [\n");
    for (i, t) in targets.iter().enumerate() {
        let _ = writeln!(out, "    {{");
        let _ = writeln!(out, "      \"name\": \"{}\",", json_escape(&t.name));
        let _ = writeln!(out, "      \"slug\": \"{}\",", json_escape(&t.slug));
        let _ = writeln!(
            out,
            "      \"fits_file\": \"{}\",",
            json_escape(&t.fits_file)
        );
        let _ = writeln!(
            out,
            "      \"width\": {}, \"height\": {}, \"channels\": {},",
            t.width, t.height, t.channels
        );
        let _ = writeln!(out, "      \"megapixels\": {:.3},", t.megapixels());
        let _ = writeln!(out, "      \"read_ms\": {:.2},", t.read_ms);
        let _ = writeln!(
            out,
            "      \"seestar_jpg\": {},",
            t.seestar_src
                .as_ref()
                .map(|s| format!("\"{}\"", json_escape(s)))
                .unwrap_or_else(|| "null".to_string())
        );
        out.push_str("      \"variants\": [\n");
        for (j, v) in t.variants.iter().enumerate() {
            let _ = writeln!(
                out,
                "        {{ \"key\": \"{}\", \"label\": \"{}\", \"pipeline_ms\": {:.2}, \
                 \"encode_ms\": {:.2}, \"image\": \"{}\" }}{}",
                v.key,
                json_escape(v.label),
                v.pipeline_ms,
                v.encode_ms,
                json_escape(&v.src),
                if j + 1 < t.variants.len() { "," } else { "" },
            );
        }
        out.push_str("      ]\n");
        let _ = writeln!(
            out,
            "    }}{}",
            if i + 1 < targets.len() { "," } else { "" }
        );
    }
    out.push_str("  ]\n}\n");
    out
}
