
// NeuroAnalyse — Shared Components & Data
// ─────────────────────────────────────────

// ── Mock patient data ────────────────────────────────────────────────
const MOCK_PATIENTS = [
  {
    id: "10000000146", name: "Ahmet Yilmaz", age: 84, gender: "E",
    lastScan: "2006-07-06", clinician: "Test Kullanıcı",
    centiloid: -2.5, risk: "negative", modality: "MRI",
    scanDate: "2006-07-06", status: "completed"
  },
  {
    id: "10000000214", name: "Fatma Sahin", age: 76, gender: "K",
    lastScan: "2006-11-09", clinician: "Test Kullanıcı",
    centiloid: 9.0, risk: "negative", modality: "MRI",
    scanDate: "2006-11-09", status: "completed"
  },
  {
    id: "10000000382", name: "Mehmet Kaya", age: 71, gender: "E",
    lastScan: "2017-04-24", clinician: "Test Kullanıcı",
    centiloid: -14.0, risk: "negative", modality: "MRI",
    scanDate: "2017-04-24", status: "completed"
  },
  {
    id: "10000000450", name: "Ayse Demir", age: 78, gender: "K",
    lastScan: "2017-09-20", clinician: "Test Kullanıcı",
    centiloid: -8.0, risk: "negative", modality: "MRI",
    scanDate: "2017-09-20", status: "completed"
  },
  {
    id: "10000000528", name: "Hasan Ozturk", age: 65, gender: "E",
    lastScan: "2017-05-22", clinician: "Test Kullanıcı",
    centiloid: 68.0, risk: "elevated", modality: "MRI",
    scanDate: "2017-05-22", status: "completed"
  },
  {
    id: "10000000528", name: "Hasan Ozturk", age: 65, gender: "E",
    lastScan: "2012-06-21", clinician: "Test Kullanıcı",
    centiloid: 32.0, risk: "borderline", modality: "MRI",
    scanDate: "2012-06-21", status: "completed"
  },
  {
    id: "10000000696", name: "Zeynep Arslan", age: 68, gender: "K",
    lastScan: "2017-06-21", clinician: "Test Kullanıcı",
    centiloid: 16.0, risk: "borderline", modality: "MRI",
    scanDate: "2017-06-21", status: "completed"
  },
];

const RISK_META = {
  negative:   { label_tr: "Negatif",    label_en: "Negative",   color: "#15803d", bg: "#f0fdf4", border: "#86efac", range: "< 25 CL" },
  borderline: { label_tr: "Sınırda",    label_en: "Borderline", color: "#92400e", bg: "#fffbeb", border: "#fcd34d", range: "25–50 CL" },
  elevated:   { label_tr: "Yüksek",     label_en: "Elevated",   color: "#9a3412", bg: "#fff7ed", border: "#fdba74", range: "50–100 CL" },
  high:       { label_tr: "Kritik",     label_en: "High",       color: "#7f1d1d", bg: "#fef2f2", border: "#fca5a5", range: "> 100 CL" },
};

// T and INTERP are defined in NeuroAnalyse.html plain <script> (synchronous).

// ── Gauge SVG Component ───────────────────────────────────────────────
function GaugeSVG({ score, size = 320 }) {
  const clamp = Math.max(0, Math.min(score || 0, 150));
  const angleDeg = 135 + (clamp / 150) * 270;
  const rotation = angleDeg - 270;

  const cx = 170, cy = 158, r = 130;
  const [displayScore, setDisplayScore] = React.useState(0);

  React.useEffect(() => {
    let start = null;
    const duration = 1200;
    const target = clamp;
    function animate(ts) {
      if (!start) start = ts;
      const p = Math.min((ts - start) / duration, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      setDisplayScore(target * eased);
      if (p < 1) requestAnimationFrame(animate);
    }
    requestAnimationFrame(animate);
  }, [score]);

  const dispRot = 135 + (Math.max(0, Math.min(displayScore, 150)) / 150) * 270 - 270;

  return (
    <svg viewBox="0 0 340 220" style={{ width: size, maxWidth: "100%" }} xmlns="http://www.w3.org/2000/svg">
      {/* Track */}
      <path d="M 78 250 A 130 130 0 1 0 262 250"
        fill="none" stroke="#e2e8f0" strokeWidth="16" strokeLinecap="round"/>
      {/* Green 0→25 */}
      <path d="M 78 250 A 130 130 0 0 0 40 158"
        fill="none" stroke="#16a34a" strokeWidth="16" strokeLinecap="butt" opacity="0.9"/>
      {/* Amber 25→50 */}
      <path d="M 40 158 A 130 130 0 0 0 78 66"
        fill="none" stroke="#d97706" strokeWidth="16" strokeLinecap="butt" opacity="0.9"/>
      {/* Orange 50→100 */}
      <path d="M 78 66 A 130 130 0 0 0 262 66"
        fill="none" stroke="#ea580c" strokeWidth="16" strokeLinecap="butt" opacity="0.9"/>
      {/* Red 100→150 */}
      <path d="M 262 66 A 130 130 0 0 0 262 250"
        fill="none" stroke="#dc2626" strokeWidth="16" strokeLinecap="butt" opacity="0.9"/>

      {/* Threshold ticks */}
      <line x1="40" y1="158" x2="26" y2="158" stroke="#94a3b8" strokeWidth="1.5"/>
      <line x1="78" y1="66" x2="68" y2="54" stroke="#94a3b8" strokeWidth="1.5"/>
      <line x1="262" y1="66" x2="272" y2="54" stroke="#94a3b8" strokeWidth="1.5"/>

      {/* Labels */}
      <text x="170" y="200" textAnchor="middle" fontSize="11" fill="#94a3b8" fontFamily="sans-serif">0</text>
      <text x="18" y="163" textAnchor="end" fontSize="11" fill="#94a3b8" fontFamily="sans-serif">25</text>
      <text x="66" y="46" textAnchor="middle" fontSize="11" fill="#94a3b8" fontFamily="sans-serif">50</text>
      <text x="274" y="46" textAnchor="middle" fontSize="11" fill="#94a3b8" fontFamily="sans-serif">100</text>
      <text x="322" y="163" textAnchor="start" fontSize="11" fill="#94a3b8" fontFamily="sans-serif">150</text>

      {/* Needle */}
      <line x1="170" y1="158" x2="170" y2="38"
        stroke="#1e293b" strokeWidth="3" strokeLinecap="round"
        transform={`rotate(${dispRot} 170 158)`}
        style={{ transition: "transform 0.05s linear" }}/>
      <circle cx="170" cy="158" r="9" fill="#1e293b"/>
      <circle cx="170" cy="158" r="4" fill="#f8fafc"/>

      {/* Centre score */}
      <text x="170" y="145" textAnchor="middle" fontSize="28" fontWeight="700"
        fill="#0f2744" fontFamily="sans-serif">{displayScore.toFixed(1)}</text>
      <text x="170" y="162" textAnchor="middle" fontSize="11" fill="#94a3b8" fontFamily="sans-serif">CL</text>
    </svg>
  );
}

// ── Risk Badge ────────────────────────────────────────────────────────
function RiskBadge({ risk, lang, size = "md" }) {
  const meta = RISK_META[risk] || RISK_META.negative;
  const label = lang === "tr" ? meta.label_tr : meta.label_en;
  const pads = size === "sm" ? "3px 10px" : "7px 16px";
  const fs = size === "sm" ? 11 : 13;
  return (
    <span style={{
      display: "inline-block",
      padding: pads,
      borderRadius: 8,
      fontSize: fs,
      fontWeight: 700,
      letterSpacing: 0.3,
      background: meta.bg,
      color: meta.color,
      border: `1px solid ${meta.border}`,
    }}>{label}</span>
  );
}

// ── Mini spark bar ────────────────────────────────────────────────────
function CentiloidBar({ value }) {
  const pct = Math.min((value / 150) * 100, 100);
  const color = value < 25 ? "#16a34a" : value < 50 ? "#d97706" : value < 100 ? "#ea580c" : "#dc2626";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{ flex: 1, height: 6, background: "#e2e8f0", borderRadius: 4, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 4, transition: "width 0.6s ease" }}></div>
      </div>
      <span style={{ fontSize: 12, fontWeight: 600, color: "#334155", minWidth: 36, textAlign: "right" }}>{value}</span>
    </div>
  );
}

// ── Stat Card ─────────────────────────────────────────────────────────
function StatCard({ label, value, color }) {
  return (
    <div style={{
      background: "#fff",
      border: "1px solid #e2e8f0",
      borderRadius: 12,
      padding: "20px 24px",
      display: "flex",
      alignItems: "center",
      gap: 16,
      boxShadow: "0 1px 3px rgba(0,0,0,0.05)"
    }}>
      <div style={{
        width: 44, height: 44, borderRadius: 10,
        background: color + "18",
        display: "flex", alignItems: "center", justifyContent: "center",
        flexShrink: 0
      }}>
        <span style={{
          width: 18,
          height: 18,
          borderRadius: 5,
          border: `2px solid ${color}`,
          display: "block",
          position: "relative",
        }}>
          <span style={{
            position: "absolute",
            left: 4,
            right: 4,
            bottom: 4,
            height: 5,
            borderRadius: 3,
            background: color,
            opacity: 0.72,
          }}></span>
        </span>
      </div>
      <div>
        <div style={{ fontSize: 26, fontWeight: 700, color: "#0f2744", lineHeight: 1 }}>{value}</div>
        <div style={{ fontSize: 12, color: "#64748b", marginTop: 3 }}>{label}</div>
      </div>
    </div>
  );
}

// ── Mock MRI slice canvases ────────────────────────────────────────────
function BrainSlice({ type, risk, src }) {
  // If the API returned a real MRI slice, show it directly
  if (src) {
    return (
      <img
        src={src}
        alt={type}
        style={{
          display: "block",
          width: "100%",
          height: "auto",
          background: "#0a0a0a",
          borderRadius: 4,
        }}
      />
    );
  }

  const canvasRef = React.useRef(null);
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    ctx.fillStyle = "#0a0a0a";
    ctx.fillRect(0, 0, w, h);

    const cx = w / 2, cy = h / 2;
    const intensity = risk === "high" ? 0.95 : risk === "elevated" ? 0.75 : risk === "borderline" ? 0.55 : 0.35;

    // Brain outline
    ctx.save();
    if (type === "axial") {
      ctx.beginPath();
      ctx.ellipse(cx, cy, w * 0.38, h * 0.44, 0, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${Math.round(180 * intensity)}, ${Math.round(160 * intensity)}, ${Math.round(140 * intensity)}, 0.9)`;
      ctx.fill();
      // Sulci
      for (let i = 0; i < 8; i++) {
        const a = (i / 8) * Math.PI * 2;
        ctx.beginPath();
        ctx.moveTo(cx + Math.cos(a) * w * 0.15, cy + Math.sin(a) * h * 0.18);
        ctx.lineTo(cx + Math.cos(a) * w * 0.34, cy + Math.sin(a) * h * 0.38);
        ctx.strokeStyle = "rgba(20,20,20,0.6)";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
      // Lateral ventricles
      ctx.beginPath();
      ctx.ellipse(cx - 12, cy, 10, 18, -0.3, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(10,10,30,0.8)";
      ctx.fill();
      ctx.beginPath();
      ctx.ellipse(cx + 12, cy, 10, 18, 0.3, 0, Math.PI * 2);
      ctx.fill();
    } else if (type === "coronal") {
      ctx.beginPath();
      ctx.ellipse(cx, cy - 5, w * 0.36, h * 0.42, 0, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${Math.round(175 * intensity)}, ${Math.round(155 * intensity)}, ${Math.round(135 * intensity)}, 0.9)`;
      ctx.fill();
      // Corpus callosum area
      ctx.beginPath();
      ctx.ellipse(cx, cy - 15, w * 0.1, h * 0.06, 0, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(200,180,160,0.4)";
      ctx.fill();
      // Cerebellum
      ctx.beginPath();
      ctx.ellipse(cx, cy + h * 0.3, w * 0.25, h * 0.15, 0, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${Math.round(155 * intensity)}, ${Math.round(135 * intensity)}, ${Math.round(115 * intensity)}, 0.85)`;
      ctx.fill();
    } else {
      // Sagittal
      ctx.beginPath();
      ctx.moveTo(cx - w * 0.05, cy - h * 0.44);
      ctx.bezierCurveTo(cx + w * 0.4, cy - h * 0.44, cx + w * 0.44, cy - h * 0.1, cx + w * 0.44, cy + h * 0.1);
      ctx.bezierCurveTo(cx + w * 0.44, cy + h * 0.3, cx + w * 0.2, cy + h * 0.44, cx - w * 0.1, cy + h * 0.44);
      ctx.bezierCurveTo(cx - w * 0.35, cy + h * 0.44, cx - w * 0.44, cy + h * 0.1, cx - w * 0.44, cy - h * 0.05);
      ctx.bezierCurveTo(cx - w * 0.44, cy - h * 0.3, cx - w * 0.2, cy - h * 0.44, cx - w * 0.05, cy - h * 0.44);
      ctx.fillStyle = `rgba(${Math.round(170 * intensity)}, ${Math.round(150 * intensity)}, ${Math.round(130 * intensity)}, 0.9)`;
      ctx.fill();
      // Brain stem
      ctx.beginPath();
      ctx.ellipse(cx + w * 0.1, cy + h * 0.35, w * 0.08, h * 0.12, 0.2, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${Math.round(140 * intensity)}, ${Math.round(120 * intensity)}, ${Math.round(100 * intensity)}, 0.9)`;
      ctx.fill();
    }
    ctx.restore();

    // Amyloid hotspot overlay for elevated/high
    if (risk === "elevated" || risk === "high") {
      const hotColor = risk === "high" ? "rgba(220,38,38,0.25)" : "rgba(234,88,12,0.18)";
      ctx.beginPath();
      ctx.ellipse(cx, cy - h * 0.1, w * 0.22, h * 0.25, 0, 0, Math.PI * 2);
      ctx.fillStyle = hotColor;
      ctx.fill();
    }

    // Label
    const labels = { axial: "AX", coronal: "COR", sagittal: "SAG" };
    ctx.fillStyle = "rgba(255,255,255,0.5)";
    ctx.font = "bold 10px monospace";
    ctx.fillText(labels[type], 8, 18);
  }, [type, risk]);

  return <canvas ref={canvasRef} width={200} height={200} style={{ display: "block", width: "100%", height: "auto" }} />;
}

Object.assign(window, {
  MOCK_PATIENTS, RISK_META,
  GaugeSVG, RiskBadge, CentiloidBar, StatCard, BrainSlice
});
