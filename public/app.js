/**
 * Ứng dụng phân tích bóng đá — chạy hoàn toàn phía trình duyệt.
 *
 * Mọi dữ liệu là file JSON tĩnh do pipeline Python sinh ra sẵn. App không
 * gọi API bóng đá nào, nên không bao giờ chạm giới hạn tốc độ, tải gần như
 * tức thì, và xem được offline sau lần mở đầu tiên.
 */

const DATA = "data";
const state = { league: null, leagues: [], tab: "fixtures", cache: new Map() };

/* ---------------- Tiện ích ---------------- */

const pct = (v) => `${(v * 100).toFixed(0)}%`;
const pct1 = (v) => `${(v * 100).toFixed(1)}%`;

const WEEKDAYS = ["Chủ nhật", "Thứ hai", "Thứ ba", "Thứ tư", "Thứ năm", "Thứ sáu", "Thứ bảy"];

/** Tên vòng đấu Champions League — nguồn trả về mã tiếng Anh viết hoa. */
const STAGES = {
  LEAGUE_STAGE: "Vòng bảng",
  PLAYOFFS: "Play-off",
  LAST_16: "Vòng 1/8",
  QUARTER_FINALS: "Tứ kết",
  SEMI_FINALS: "Bán kết",
  FINAL: "Chung kết",
};

/** Tên giải quốc nội, dùng trong bảng sức mạnh Champions League. */
const LEAGUE_LABELS = {
  "premier-league": "Anh", "la-liga": "TBN", "serie-a": "Ý",
  "bundesliga": "Đức", "ligue-1": "Pháp", "eredivisie": "Hà Lan",
  "primeira-liga": "BĐN", "scottish-premiership": "Scotland",
  "turkish-super-lig": "Thổ", "belgian-pro-league": "Bỉ",
  "greek-super-league": "Hy Lạp", "austrian-bundesliga": "Áo",
  unknown: "—",
};

const roundLabel = (r) => STAGES[r] || r || "";

/**
 * Ngày thi đấu theo múi giờ của thiết bị.
 *
 * Luôn ưu tiên kickoff_utc: giờ ghi trong nguồn là giờ sân vận động, nên một
 * trận Ngoại hạng Anh 20:00 giờ London là 02:00 sáng hôm sau ở Việt Nam.
 * Nhóm theo ngày địa phương mới đúng với cảm nhận của người xem.
 */
function localDate(fixture) {
  if (fixture.kickoff_utc) {
    const d = new Date(fixture.kickoff_utc);
    if (!Number.isNaN(d.getTime())) return d;
  }
  return new Date(`${fixture.date}T00:00:00`);
}

function dayKey(fixture) {
  const d = localDate(fixture);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

function formatDay(key) {
  const d = new Date(`${key}T00:00:00`);
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const diff = Math.round((d - today) / 86400000);
  if (diff === 0) return "Hôm nay";
  if (diff === 1) return "Ngày mai";
  return `${WEEKDAYS[d.getDay()]}, ${d.getDate()}/${d.getMonth() + 1}`;
}

/** Giờ bóng lăn theo múi giờ thiết bị. Trả chuỗi rỗng nếu nguồn thiếu giờ. */
function formatTime(fixture) {
  if (!fixture.kickoff_utc) return "";
  const d = new Date(fixture.kickoff_utc);
  if (Number.isNaN(d.getTime())) return "";
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

/**
 * Rút gọn tên đội cho vừa màn hình hẹp.
 *
 * Mô hình Champions League làm việc trên tên đã chuẩn hoá (viết thường, bỏ
 * hậu tố) để ghép được giữa các nguồn, nên tên đến đây có thể toàn chữ
 * thường. Viết hoa lại đầu từ trước khi hiển thị.
 */
function titleCase(name) {
  if (name !== name.toLowerCase()) return name;
  return name.replace(/\b[a-zà-ỹ]/g, (c) => c.toUpperCase());
}

function shortName(name) {
  return titleCase(name)
    .replace(/\s+(FC|CF|AFC|SAD|BC)$/i, "")
    // Chỉ bỏ tiền tố pháp lý, GIỮ phần phân biệt. Trước đây luật này biến
    // "Club Atlético de Madrid" thành "Madrid" — nằm ngay cạnh Real Madrid
    // trong bảng Champions League thì không ai đọc ra là đội nào.
    .replace(/^(FC|CF|AFC|SS|SL|AC|Club)\s+/i, "")
    .replace(/^Club Atlético de\s+/i, "Atlético ")
    .replace(/^Atlético de\s+/i, "Atlético ")
    .replace(/\s+Balompié$/i, "")
    .replace(/\s+de Fútbol$/i, "")
    .replace(/\s+CF$/i, "")
    .trim();
}

const escapeHtml = (s) =>
  String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function loadJSON(path) {
  if (state.cache.has(path)) return state.cache.get(path);
  const res = await fetch(path, { cache: "no-cache" });
  if (!res.ok) throw new Error(`${res.status} ${path}`);
  const data = await res.json();
  state.cache.set(path, data);
  return data;
}

const el = (id) => document.getElementById(id);
const view = () => el("view");

/* ---------------- Lịch thi đấu ---------------- */

function confidenceClass(level) {
  return level === "cao" ? "cao" : level === "trung bình" ? "trung" : "thap";
}

function signalClass(strength) {
  return strength === "rõ" ? "ro" : strength === "vừa" ? "vua" : "mo";
}

/**
 * Khối trả lời câu hỏi "nhìn bảng này thì tin được cái gì".
 *
 * Đặt ngay đầu bảng chi tiết vì đó là thứ người dùng cần trước tiên; các con
 * số xác suất chi tiết chỉ có ích sau khi đã biết trận này có tín hiệu hay không.
 */
function signalSection(p) {
  const sig = p.signal;
  if (!sig) return "";

  const b = sig.best;
  const verdict = {
    "rõ": "Trận này có tín hiệu rõ.",
    "vừa": "Trận này có tín hiệu vừa phải.",
    "mơ hồ": "Trận này không có tín hiệu rõ ràng.",
  }[sig.strength] || "";

  const sc = sig.score;
  const mismatch = sig.score_matches_pick === false;

  const rows = sig.ranked.slice(0, 4).map((o) => `
    <div class="kv">
      <span>${escapeHtml(o.pick)}</span>
      <b>${pct(o.probability)}
        <span style="color:var(--text-faint);font-weight:500;font-size:12px">
          (nền ${pct(o.base_rate)}, ${o.clarity >= 0 ? "+" : ""}${(o.clarity * 100).toFixed(0)})
        </span>
      </b>
    </div>`).join("");

  return `
    <div class="verdict verdict-${signalClass(sig.strength)}">
      <div class="verdict-head">${escapeHtml(verdict)}</div>

      <div class="verdict-grid">
        <div class="verdict-cell">
          <span class="verdict-label">Kèo</span>
          <b class="verdict-value">${escapeHtml(b.pick)}</b>
          <span class="verdict-prob">${pct1(b.probability)}</span>
        </div>
        ${sc ? `
        <div class="verdict-cell">
          <span class="verdict-label">Tỷ số</span>
          <b class="verdict-value">${sc.top.home}–${sc.top.away}</b>
          <span class="verdict-prob">${pct1(sc.top.probability)}</span>
        </div>` : ""}
      </div>

      ${sc ? `<div class="verdict-note">
        Ba tỷ số ${sc.top3.map((x) => `${x.home}–${x.away}`).join(", ")} gộp lại chiếm
        <strong>${pct(sc.top3_combined)}</strong> khả năng. Một tỷ số đơn lẻ hiếm khi
        vượt 12%, nên con số gộp mới là mức đáng dùng.
      </div>` : ""}

      <div class="verdict-note">
        Cửa "${escapeHtml(b.pick)}" bình thường xảy ra ${pct(b.base_rate)} số trận,
        ở đây là ${pct(b.probability)} — cao hơn mức thường
        ${(b.clarity * 100).toFixed(0)} điểm phần trăm.
      </div>

      ${mismatch ? `<div class="verdict-warn">⚠ Tỷ số khả dĩ nhất
        (${sc.top.home}–${sc.top.away}) không cùng hướng với cửa được chọn.
        Dấu hiệu trận này thực sự khó đoán.</div>` : ""}
      ${sig.warnings.length ? `<div class="verdict-warn">⚠ ${sig.warnings.map(escapeHtml).join(" · ")}</div>` : ""}

      <div class="verdict-fail">Vẫn hỏng khoảng <strong>${pct(sig.fails_about)}</strong> số lần.
        Đây là công cụ đọc số, không phải gợi ý đặt cược.</div>
    </div>

    <div class="section-label">Các cửa, xếp theo mức khác thường</div>
    ${rows}
    <div class="note">Cột trong ngoặc là tần suất cửa đó xảy ra ở một trận
      <strong>bất kỳ</strong> của giải này. Chênh lệch so với nó mới là thông tin —
      "Tài 1.5" luôn khoảng 75% ở mọi trận nên dù cao vẫn không nói lên điều gì riêng.</div>
  `;
}

/** Nguồn dự đoán tốt nhất cho một trận: thị trường nếu có kèo, nếu không thì mô hình. */
function bestSource(p) {
  if (p.market?.available) {
    return { outcome: p.market.prediction.outcome, eg: p.market.expected_goals, fromMarket: true };
  }
  return { outcome: p.prediction.outcome, eg: p.expected_goals, fromMarket: false };
}

function matchCard(p, index) {
  const src = bestSource(p);
  const o = src.outcome;
  const eg = src.eg;
  const homeLeads = o.home >= o.away;

  return `
  <article class="match" data-idx="${index}">
    <div class="match-meta">
      <span>${escapeHtml(roundLabel(p.fixture.round))}${formatTime(p.fixture) ? " · " + formatTime(p.fixture) : ""}</span>
      <span>
        ${src.fromMarket ? '<span class="badge market">kèo</span>' : ""}
        ${p.signal ? `<span class="badge sig-${signalClass(p.signal.strength)}">${escapeHtml(p.signal.strength)}</span>` : ""}
        <span class="badge ${confidenceClass(p.confidence.level)}">${escapeHtml(p.confidence.level)}</span>
      </span>
    </div>
    <div class="teams">
      <div class="team-row">
        <span class="team-name ${homeLeads ? "leader" : "trailer"}">${escapeHtml(shortName(p.home_team))}</span>
        <span class="xg">${eg.home.toFixed(2)}</span>
      </div>
      <div class="team-row">
        <span class="team-name ${!homeLeads ? "leader" : "trailer"}">${escapeHtml(shortName(p.away_team))}</span>
        <span class="xg">${eg.away.toFixed(2)}</span>
      </div>
    </div>
    <div class="probbar" role="img" aria-label="Thắng nhà ${pct(o.home)}, hoà ${pct(o.draw)}, thắng khách ${pct(o.away)}">
      <span class="p-home" style="width:${o.home * 100}%"></span>
      <span class="p-draw" style="width:${o.draw * 100}%"></span>
      <span class="p-away" style="width:${o.away * 100}%"></span>
    </div>
    <div class="problabels">
      <span class="l-home">Nhà <b>${pct(o.home)}</b></span>
      <span class="l-draw">Hoà <b>${pct(o.draw)}</b></span>
      <span class="l-away">Khách <b>${pct(o.away)}</b></span>
    </div>
    ${p.signal ? `<div class="card-verdict">
      <span>${escapeHtml(p.signal.best.pick)} <b>${pct(p.signal.best.probability)}</b></span>
      ${p.signal.score ? `<span>${p.signal.score.top.home}–${p.signal.score.top.away}
        <b>${pct(p.signal.score.top.probability)}</b></span>` : ""}
    </div>` : ""}
  </article>`;
}

async function renderFixtures() {
  const preds = await loadJSON(`${DATA}/${state.league}/predictions.json`);
  if (!preds.length) {
    view().innerHTML = `<div class="empty">Không có trận nào trong 14 ngày tới.</div>`;
    return;
  }

  const byDay = new Map();
  preds.forEach((p, i) => {
    const day = dayKey(p.fixture);
    if (!byDay.has(day)) byDay.set(day, []);
    byDay.get(day).push({ p, i });
  });

  // Nhóm lại theo ngày địa phương có thể làm lệch thứ tự so với thứ tự gốc,
  // nên sắp xếp lại theo ngày trước khi hiển thị.
  const html = [...byDay.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([day, items]) =>
      `<h2 class="day-header">${escapeHtml(formatDay(day))}</h2>` +
      items.map(({ p, i }) => matchCard(p, i)).join(""))
    .join("");

  view().innerHTML = html;
  view().querySelectorAll(".match").forEach((node) =>
    node.addEventListener("click", () => openSheet(preds[+node.dataset.idx])));
}

/* ---------------- Chi tiết trận ---------------- */

/** Khối so sánh mô hình với thị trường, chỉ hiện khi có kèo. */
function marketSection(p) {
  if (!p.market?.available) {
    return `<div class="note">Chưa có kèo cho trận này, nên các con số trên đây
      hoàn toàn từ mô hình. Nhà cái thường chỉ treo kèo trước khoảng 10 ngày.</div>`;
  }

  const mk = p.market;
  const mo = mk.prediction.outcome;
  const bo = p.prediction.outcome;
  const d = mk.disagreement;
  const oo = mk.offered_odds;

  const row = (label, m, k) => `
    <tr>
      <td style="text-align:left">${label}</td>
      <td class="num">${pct1(m)}</td>
      <td class="num">${pct1(k)}</td>
      <td class="num" style="color:${Math.abs(m - k) >= 0.05 ? "var(--draw)" : "var(--text-faint)"}">
        ${(m - k >= 0 ? "+" : "") + ((m - k) * 100).toFixed(1)}
      </td>
    </tr>`;

  const best = d?.available ? d.best_edge : null;

  return `
    <div class="section-label">Mô hình so với thị trường</div>
    <div class="card" style="margin-bottom:12px">
      <table>
        <thead><tr>
          <th style="text-align:left;padding-left:12px">Cửa</th>
          <th>Mô hình</th><th>Thị trường</th><th style="padding-right:12px">Lệch</th>
        </tr></thead>
        <tbody>
          ${row("Thắng nhà", bo.home, mo.home)}
          ${row("Hoà", bo.draw, mo.draw)}
          ${row("Thắng khách", bo.away, mo.away)}
        </tbody>
      </table>
    </div>
    <div class="kv"><span>Kèo nhà cái</span><b>${oo.home} / ${oo.draw} / ${oo.away}</b></div>
    ${mk.total_line ? `<div class="kv"><span>Kèo tài/xỉu ${mk.total_line.point}</span>
      <b>${mk.total_line.over} / ${mk.total_line.under}</b></div>` : ""}
    <div class="kv"><span>Nguồn kèo</span><b>${escapeHtml(mk.source)}</b></div>
    ${d?.available ? `<div class="kv"><span>Biên nhà cái</span><b>${((d.overround - 1) * 100).toFixed(1)}%</b></div>` : ""}
    ${best && best.edge > 0.03 ? `<div class="note"><strong>Mô hình bất đồng ở cửa
      ${best.outcome === "home" ? "thắng nhà" : best.outcome === "draw" ? "hoà" : "thắng khách"}.</strong>
      Mô hình cho <strong>${pct1(best.model_prob)}</strong>, trong khi kèo
      <strong>${best.offered_odds}</strong> chỉ hàm ý <strong>${pct1(best.market_prob)}</strong>.
      <br><br>Lưu ý quan trọng: trên 2.270 trận đã kiểm định, khi hai bên bất đồng thì
      <em>thị trường thường đúng hơn</em>. Chênh lệch kiểu này hay là dấu hiệu mô hình
      đang thiếu thông tin (xoay tua đội hình, thông tin phút chót) chứ không phải nhà cái ra giá sai.</div>` : ""}
  `;
}

/**
 * Phần cầu thủ ghi bàn. Có hai chế độ vì dữ liệu đến ở hai thời điểm khác nhau.
 *
 * ESPN chỉ công bố đội hình khoảng một tiếng trước giờ bóng lăn, trong khi app
 * hiển thị lịch tới 14 ngày. Nếu chỉ dựa vào đội hình thì phần lớn thời gian
 * mục này trống trơn — nên trước đó hiện danh sách theo phong độ ghi bàn, và
 * ghi rõ đó chưa phải xác suất.
 */
function scorersSection(p) {
  const lu = p.lineup;
  if (lu?.available) {
    const blocks = ["home", "away"].map((side) => {
      const d = lu.sides[side];
      if (!d) return "";
      return `
      <div class="lineup">
        <div class="lineup-head">${escapeHtml(shortName(d.team))}
          ${d.formation ? `<span>${escapeHtml(d.formation)}</span>` : ""}</div>
        <div class="lineup-xi">${d.starters.map((x) =>
          escapeHtml(x.name.split(" ").slice(-1)[0])).join(" · ")}</div>
        ${d.scorers?.length ? `<div class="lineup-scorers">
          ${d.scorers.slice(0, 4).map((x) => `<div class="kv">
            <span>${escapeHtml(x.player)} <span style="color:var(--text-faint)">${escapeHtml(x.position || "")}</span></span>
            <b>${pct(x.probability)}</b></div>`).join("")}
        </div>` : ""}
      </div>`;
    }).join("");

    return `
    <div class="section-label">Đội hình ra sân · xác suất ghi bàn</div>
    ${blocks}
    <div class="note"><strong>Ước lượng thô.</strong> Tính từ số bàn của cầu thủ
      mùa này và mùa trước, cộng mức nền theo vị trí, rồi chuẩn hoá theo bàn
      thắng kỳ vọng của đội. Kiểm chứng ngược trên trận đã đá: người ghi bàn
      thật xếp hạng 1 và 2 trên 11. Chỉ tính <strong>11 người đá chính</strong> —
      cầu thủ vào sân từ ghế dự bị không có trong danh sách.</div>`;
  }

  const form = p.in_form_scorers;
  const hasAny = form && ((form.home || []).length || (form.away || []).length);
  if (!hasAny) return "";

  const block = (side, team) => {
    const rows = form[side] || [];
    if (!rows.length) {
      return `<div class="lineup"><div class="lineup-head">${escapeHtml(shortName(team))}</div>
        <div class="lineup-xi">Chưa có cầu thủ nào ghi bàn mùa này.</div></div>`;
    }
    return `
    <div class="lineup">
      <div class="lineup-head">${escapeHtml(shortName(team))}</div>
      <div class="lineup-scorers">
        ${rows.map((x) => `<div class="kv">
          <span>${escapeHtml(x.player)}</span>
          <b>${x.goals_this_season} bàn
            <span style="color:var(--text-faint);font-weight:500;font-size:12px">
              (${x.per_match.toFixed(2)}/trận)</span></b>
        </div>`).join("")}
      </div>
    </div>`;
  };

  return `
    <div class="section-label">Cầu thủ đang ghi bàn tốt nhất</div>
    ${block("home", p.home_team)}
    ${block("away", p.away_team)}
    <div class="note"><strong>Đây chưa phải xác suất ghi bàn.</strong> Đội hình ra
      sân chỉ được công bố khoảng một tiếng trước giờ bóng lăn. Khi có đội hình,
      mục này tự chuyển thành xác suất tính trên 11 người đá chính.</div>`;
}

function openSheet(p) {
  const src = bestSource(p);
  const o = src.outcome;
  const ou = (p.market?.available ? p.market.prediction : p.prediction).over_under["2.5"];
  const h2h = p.head_to_head;
  const exH = p.explain.home;
  const exA = p.explain.away;

  const pred = p.market?.available ? p.market.prediction : p.prediction;
  const scores = pred.correct_score.slice(0, 3).map((s) =>
    `<div class="score-chip"><b>${s.home}-${s.away}</b><span>${pct1(s.probability)}</span></div>`).join("");

  const h2hRows = h2h.matches.length
    ? h2h.matches.map((m) => `
        <div class="h2h-row">
          <span class="h2h-date">${escapeHtml(m.date)}</span>
          <span class="h2h-teams">${escapeHtml(shortName(m.home_team))} – ${escapeHtml(shortName(m.away_team))}</span>
          <span class="h2h-score">${m.home_goals}-${m.away_goals}</span>
        </div>`).join("")
    : `<div class="empty" style="padding:20px">Chưa có lịch sử đối đầu trong dữ liệu.</div>`;

  // Chỉ hiện phần hiệu chỉnh khi nó thực sự thay đổi kết quả đáng kể.
  const adjRows = [];
  const addAdj = (label, side) => {
    if (Math.abs(side.total_multiplier - 1) < 0.005) return;
    const dir = side.total_multiplier > 1 ? "+" : "";
    adjRows.push(`<div class="kv"><span>${label}</span><b>${dir}${((side.total_multiplier - 1) * 100).toFixed(1)}%</b></div>`);
  };
  addAdj(`${escapeHtml(shortName(p.home_team))} (sân nhà)`, exH);
  addAdj(`${escapeHtml(shortName(p.away_team))} (sân khách)`, exA);

  const adjSection = adjRows.length
    ? `<div class="section-label">Hiệu chỉnh bối cảnh</div>${adjRows.join("")}
       <div class="note">Hiệu chỉnh gồm ngày nghỉ và động lực theo vị trí
       trên bảng xếp hạng.</div>`
    : "";

  el("sheetBody").innerHTML = `
    <h2 class="sheet-title">${escapeHtml(shortName(p.home_team))} vs ${escapeHtml(shortName(p.away_team))}</h2>
    <p class="sheet-sub">${escapeHtml(formatDay(dayKey(p.fixture)))}${formatTime(p.fixture) ? " · " + formatTime(p.fixture) : ""} · ${escapeHtml(roundLabel(p.fixture.round))}</p>

    <div class="probbar" style="height:8px">
      <span class="p-home" style="width:${o.home * 100}%"></span>
      <span class="p-draw" style="width:${o.draw * 100}%"></span>
      <span class="p-away" style="width:${o.away * 100}%"></span>
    </div>
    <div class="problabels" style="margin-top:7px">
      <span class="l-home">Nhà <b>${pct1(o.home)}</b></span>
      <span class="l-draw">Hoà <b>${pct1(o.draw)}</b></span>
      <span class="l-away">Khách <b>${pct1(o.away)}</b></span>
    </div>

    ${signalSection(p)}

    <div class="section-label">Bàn thắng kỳ vọng${src.fromMarket ? " (theo kèo)" : ""}</div>
    <div class="kv"><span>${escapeHtml(shortName(p.home_team))}</span><b>${src.eg.home.toFixed(2)}</b></div>
    <div class="kv"><span>${escapeHtml(shortName(p.away_team))}</span><b>${src.eg.away.toFixed(2)}</b></div>
    <div class="kv"><span>Tổng bàn kỳ vọng</span><b>${pred.expected_total_goals.toFixed(2)}</b></div>

    <div class="section-label">Thị trường khác</div>
    <div class="kv"><span>Tài 2.5</span><b>${pct1(ou.over)}</b></div>
    <div class="kv"><span>Xỉu 2.5</span><b>${pct1(ou.under)}</b></div>
    <div class="kv"><span>Cả hai đội ghi bàn</span><b>${pct1(pred.btts.yes)}</b></div>

    ${scorersSection(p)}

    ${p.first_half ? `
    <div class="section-label">Hiệp 1</div>
    <div class="kv"><span>Thắng nhà / Hoà / Thắng khách</span>
      <b>${pct(p.first_half.outcome.home)} · ${pct(p.first_half.outcome.draw)} · ${pct(p.first_half.outcome.away)}</b></div>
    <div class="kv"><span>Có bàn trong hiệp 1</span>
      <b>${pct1(p.first_half.over_under["0.5"].over)}</b></div>
    <div class="kv"><span>Tài 1.5 hiệp 1</span>
      <b>${pct1(p.first_half.over_under["1.5"].over)}</b></div>
    <div class="kv"><span>Bàn kỳ vọng hiệp 1</span>
      <b>${p.first_half.expected_goals.home.toFixed(2)} – ${p.first_half.expected_goals.away.toFixed(2)}</b></div>
    <div class="note">Kèo hiệp 1 do mô hình suy ra, không lấy từ nhà cái:
      ${pct(p.first_half.share_used)} số bàn của giải này được ghi trong hiệp 1.
      Đã đối chiếu với kèo thật ở một trận mẫu, lệch khoảng 1 điểm phần trăm.</div>` : ""}

    <div class="section-label">Ba tỷ số khả dĩ nhất</div>
    <div class="scores scores-3">${scores}</div>
    ${p.market?.available && p.market.calibrated_on?.length > 1
      ? `<div class="note">Ba tỷ số này được tính từ ma trận đã ép khớp
         <strong>cả 1X2 lẫn kèo tài/xỉu</strong> của nhà cái, nên chúng phản ánh
         cả tương quan hai đội lẫn kỳ vọng tổng bàn của thị trường.</div>`
      : ""}

    ${marketSection(p)}

    ${adjSection}

    <div class="section-label">Đối đầu gần đây</div>
    ${h2hRows}

    <div class="section-label">Độ tin cậy</div>
    <div class="kv"><span>Mức</span><b>${escapeHtml(p.confidence.level)}</b></div>
    <div class="kv"><span>Số trận dữ liệu ${escapeHtml(shortName(p.home_team))}</span><b>${p.confidence.home_matches}</b></div>
    <div class="kv"><span>Số trận dữ liệu ${escapeHtml(shortName(p.away_team))}</span><b>${p.confidence.away_matches}</b></div>
    ${p.confidence.level !== "cao" ? `<div class="note"><strong>Cẩn trọng.</strong>
      ${escapeHtml(shortName(p.confidence.limiting_team))} có ít dữ liệu ở giải này, nên mô hình
      kéo mạnh ước lượng về mức trung bình. Dự đoán kém tin cậy hơn bình thường.</div>` : ""}

    <div class="note">Tỷ lệ công bằng theo mô hình (chưa tính biên nhà cái):
      <strong>${p.prediction.fair_odds.home} / ${p.prediction.fair_odds.draw} / ${p.prediction.fair_odds.away}</strong>.</div>
  `;
  el("sheet").hidden = false;
  document.body.style.overflow = "hidden";
}

function closeSheet() {
  el("sheet").hidden = true;
  document.body.style.overflow = "";
}

/* ---------------- Lịch sử ---------------- */

/**
 * Trang lịch sử: trận đã đá, kết quả thật, người ghi bàn, và dự đoán đã đưa ra.
 *
 * Mục đích chính không phải xem lại tỷ số — chỗ nào cũng có — mà để đối chiếu
 * dự đoán với thực tế. Một app dự đoán không cho xem lại nó đã sai thế nào thì
 * không có cách nào để tin được.
 */
async function renderHistory() {
  const rows = await loadJSON(`${DATA}/${state.league}/history.json`);
  if (!rows.length) {
    view().innerHTML = `<div class="empty">Chưa có trận nào được lưu.<br>
      <small>Lịch sử bắt đầu tích luỹ từ khi trận đầu tiên đá xong.</small></div>`;
    return;
  }

  const graded = rows.filter((r) => r.grade);
  const correct = graded.filter((r) => r.grade.top_pick_correct).length;
  const scored = graded.filter((r) => r.grade.score_correct !== null);
  const scoreHit = scored.filter((r) => r.grade.score_correct).length;
  const brier = graded.length
    ? graded.reduce((a, r) => a + r.grade.brier, 0) / graded.length : 0;

  const card = rows.map((r) => {
    const a = r.actual;
    const g = r.grade || {};
    const homeWon = a.home_goals > a.away_goals;
    const awayWon = a.home_goals < a.away_goals;

    // Ghi rõ tên đội thay vì mũi tên: với trận có 5 bàn của cả hai bên,
    // ký hiệu ◄ ► buộc người đọc phải dịch trong đầu từng dòng một.
    const teamTag = (side) => {
      const name = shortName(side === "home" ? r.home_team : r.away_team);
      const color = side === "home" ? "var(--home)" : "var(--away)";
      return `<span style="color:${color};font-size:11px;font-weight:650"> ${escapeHtml(name)}</span>`;
    };

    const goals = (a.goals || []).length
      ? `<div class="hist-goals">${a.goals.map((x) => `
          <div><i>${escapeHtml(x.minute || "")}</i>${escapeHtml(x.scorer || "?")}${teamTag(x.side)}
            ${x.assist ? `<span style="color:var(--text-faint);font-size:11.5px"> ← ${escapeHtml(x.assist)}</span>` : ""}
          </div>`).join("")}</div>`
      : "";

    const pick = g.signal_pick || g.top_pick;
    const ok = g.signal_correct !== null && g.signal_correct !== undefined
      ? g.signal_correct : g.top_pick_correct;

    return `
    <article class="hist">
      <div class="hist-top">
        <span>${escapeHtml(formatDay(dayKey(r.fixture)))} · ${escapeHtml(roundLabel(r.fixture.round))}</span>
        ${a.half_time ? `<span>HT ${a.half_time.home}–${a.half_time.away}</span>` : ""}
      </div>
      <div class="hist-score">
        <div class="hist-row ${homeWon ? "won" : "lost"}">
          <span>${escapeHtml(shortName(r.home_team))}</span><b>${a.home_goals}</b>
        </div>
        <div class="hist-row ${awayWon ? "won" : "lost"}">
          <span>${escapeHtml(shortName(r.away_team))}</span><b>${a.away_goals}</b>
        </div>
      </div>
      <div class="hist-check">
        <span class="dim">Dự đoán:</span>
        <span class="${ok ? "ok" : "bad"}">${escapeHtml(pick || "—")} ${ok ? "✓" : "✕"}</span>
        ${g.score_predicted ? `<span class="dim">· tỷ số ${escapeHtml(g.score_predicted)}
          ${g.score_correct ? '<span class="ok">✓</span>' : ""}</span>` : ""}
        <span class="dim">· gán ${pct(g.probability_of_actual || 0)} cho kết quả thật</span>
      </div>
      ${goals}
    </article>`;
  }).join("");

  view().innerHTML = `
    <div class="card">
      <div class="card-title">Đối chiếu dự đoán — ${graded.length} trận</div>
      <div class="acc-grid">
        <div class="acc-cell"><b>${graded.length ? pct(correct / graded.length) : "—"}</b>
          <span>đúng cửa 1X2</span></div>
        <div class="acc-cell"><b>${scored.length ? pct(scoreHit / scored.length) : "—"}</b>
          <span>đúng tỷ số</span></div>
        <div class="acc-cell"><b>${brier.toFixed(2)}</b><span>Brier (thấp = tốt)</span></div>
      </div>
    </div>
    <div class="note">Mẫu còn nhỏ nên các con số này dao động mạnh. Chỉ sau vài
      chục trận chúng mới đáng tin. Brier dưới 0.60 là khá, dưới 0.55 là tốt.</div>
    ${card}`;
}

/* ---------------- Bảng xếp hạng ---------------- */

/**
 * Vùng xếp hạng. Giải quốc nội và Champions League có luật hoàn toàn khác:
 * giải quốc nội 20 đội lấy 4 suất C1 và xuống hạng 3 đội cuối, còn vòng bảng
 * C1 có 36 đội, 8 đội đầu vào thẳng vòng 1/8, hạng 9-24 đá play-off, từ 25
 * trở xuống bị loại.
 */
function zoneClass(pos, total) {
  if (total > 30) {
    if (pos <= 8) return "zone-ucl";
    if (pos <= 24) return "zone-uel";
    return "zone-rel";
  }
  if (pos <= 4) return "zone-ucl";
  if (pos <= 6) return "zone-uel";
  if (pos > total - 3) return "zone-rel";
  return "";
}

async function renderStandings() {
  const rows = await loadJSON(`${DATA}/${state.league}/standings.json`);
  if (!rows.length) {
    view().innerHTML = `<div class="empty">Mùa giải chưa bắt đầu.</div>`;
    return;
  }

  const body = rows.map((r) => `
    <tr class="${zoneClass(r.position, rows.length)}">
      <td class="pos">${r.position}</td>
      <td>${escapeHtml(shortName(r.team))}</td>
      <td class="num">${r.played}</td>
      <td class="num">${r.goal_difference > 0 ? "+" : ""}${r.goal_difference}</td>
      <td class="pts">${r.points}</td>
      <td style="padding-right:12px"><span class="form">${(r.form || []).map((f) => `<i class="${f}">${f}</i>`).join("")}</span></td>
    </tr>`).join("");

  view().innerHTML = `
    <div class="card">
      <div class="card-title">Bảng xếp hạng — mùa 2026/27</div>
      <table>
        <thead><tr><th></th><th>Đội</th><th>Tr</th><th>HS</th><th>Đ</th><th style="padding-right:12px">Phong độ</th></tr></thead>
        <tbody>${body}</tbody>
      </table>
    </div>
    <div class="note">${rows.length > 30
      ? "Vòng bảng Champions League 36 đội. Viền xanh lá = vào thẳng vòng 1/8 (hạng 1-8), xanh dương = đá play-off (9-24), đỏ = bị loại (25-36)."
      : "Viền xanh lá = suất Champions League, xanh dương = Europa League, đỏ = xuống hạng."}</div>`;
}

/* ---------------- Sức mạnh đội bóng ---------------- */

async function renderStrength() {
  const rows = await loadJSON(`${DATA}/${state.league}/strength.json`);
  const model = await loadJSON(`${DATA}/${state.league}/model.json`);

  // Bảng Champions League có thêm cột giải quốc nội — không có nó thì
  // không hiểu vì sao một CLB Scotland lại xếp thấp hơn CLB Anh cùng chỉ số.
  const showLeague = rows.some((r) => r.league);
  const maxAbs = Math.max(...rows.map((r) => Math.max(Math.abs(r.attack_pct), Math.abs(r.defence_pct))), 1);
  const bar = (v, color) => {
    const w = (Math.abs(v) / maxAbs) * 100;
    return `<span class="strength-bar">
      <span class="strength-val" style="color:${v >= 0 ? color : "var(--danger)"}">${v > 0 ? "+" : ""}${v.toFixed(0)}%</span>
      <span class="strength-track"><i style="width:${w}%;background:${v >= 0 ? color : "var(--danger)"}"></i></span>
    </span>`;
  };

  const thin = (r) => r.bridge_matches !== undefined
    ? r.bridge_matches === 0
    : r.matches < 15;

  const body = rows.map((r) => `
    <tr>
      <td class="pos">${r.rank}</td>
      <td>${escapeHtml(shortName(r.team))}${thin(r) ? ' <span style="color:var(--danger);font-size:10px">●</span>' : ""}
        ${showLeague ? `<span style="color:var(--text-faint);font-size:10.5px"> ${escapeHtml(LEAGUE_LABELS[r.league] || r.league)}</span>` : ""}</td>
      <td>${bar(r.attack_pct, "var(--home)")}</td>
      <td>${bar(r.defence_pct, "var(--away)")}</td>
    </tr>`).join("");

  view().innerHTML = `
    <div class="card">
      <div class="card-title">${showLeague ? "Sức mạnh trên thang chung châu Âu" : "Sức mạnh so với trung bình giải"}</div>
      <div class="table-scroll">
        <table class="strength-table">
          <thead><tr><th></th><th>Đội</th><th>Tấn công</th><th>Phòng ngự</th></tr></thead>
          <tbody>${body}</tbody>
        </table>
      </div>
    </div>
    <div class="note">
      ${showLeague ? `Đây là thang <strong>chung cho cả châu Âu</strong>: sức mạnh trong giải
      đã được cộng thêm mặt bằng của chính giải đó, nên một CLB Scotland và một CLB Anh
      cùng chỉ số nghĩa là thực sự ngang nhau.<br><br>` : ""}
      <strong>Tấn công</strong> là mức ghi bàn cao hơn/thấp hơn đội trung bình.
      <strong>Phòng ngự</strong> là mức thủng lưới ít hơn/nhiều hơn — số dương luôn là tốt.
      Dấu <span style="color:var(--danger)">●</span> đánh dấu đội thiếu dữ liệu, ước lượng kém chắc chắn.
    </div>
    <div class="note">Mô hình khớp trên ${model.trained_on_matches} trận.
      Lợi thế sân nhà giải này: nhân <strong>${Math.exp(model.home_advantage).toFixed(2)}</strong> lần bàn thắng kỳ vọng.</div>`;
}

/* ---------------- Trang giải thích mô hình ---------------- */

async function renderAbout() {
  const index = await loadJSON(`${DATA}/index.json`);
  const model = await loadJSON(`${DATA}/${state.league}/model.json`);
  const built = new Date(index.generated_at);
  const ageHours = (Date.now() - built.getTime()) / 3600000;
  const src = index.sources || {};

  const freshness = ageHours < 14
    ? { label: "mới", cls: "cao" }
    : ageHours < 36
      ? { label: "hơi cũ", cls: "trung" }
      : { label: "đã cũ", cls: "thap" };

  const sourceRow = (key, name) => {
    const s = src[key];
    if (!s) return "";
    const ok = s.configured && !s.blocked;
    const state_ = s.blocked ? "bị chặn" : s.configured ? "đang dùng" : "chưa có key";
    const cls = ok ? "cao" : s.configured ? "trung" : "thap";
    return `<div class="kv">
      <span>${escapeHtml(name)}<br>
        <small style="color:var(--text-faint)">${escapeHtml(s.purpose)}</small></span>
      <b><span class="badge ${cls}">${escapeHtml(state_)}</span></b>
    </div>`;
  };

  view().innerHTML = `
    <div class="card"><div class="card-title">Trạng thái dữ liệu</div>
      <div style="padding:10px 14px">
        <div class="kv"><span>Cập nhật lần cuối</span>
          <b>${built.toLocaleString("vi-VN")} <span class="badge ${freshness.cls}">${freshness.label}</span></b></div>
        <div class="kv"><span>Cách đây</span><b>${ageHours < 1 ? "dưới 1 giờ" : Math.round(ageHours) + " giờ"}</b></div>
        ${index.leagues.map((l) => `<div class="kv"><span>${escapeHtml(l.name)}</span>
          <b>${l.fixtures} trận${l.teams ? " · " + l.teams + " đội" : ""}</b></div>`).join("")}
      </div>
    </div>

    <div class="card"><div class="card-title">Nguồn dữ liệu</div>
      <div style="padding:10px 14px">
        ${sourceRow("openfootball", "openfootball")}
        ${sourceRow("odds_api", "The Odds API")}
        ${sourceRow("football_data_org", "football-data.org")}
      </div>
    </div>


    <div class="card"><div class="card-title">API key đặt ở đâu</div>
      <div class="prose" style="padding:12px 14px">
        <p>App này <strong>không có ô nhập key</strong>, và đó là chủ ý. Trang web
        chỉ đọc các file JSON dựng sẵn — nó không bao giờ tự gọi API bóng đá.
        Người gọi API là pipeline Python chạy trên máy bạn hoặc trên GitHub Actions.</p>

        <p>Nếu đặt key vào giao diện, key sẽ nằm trong mã trang mà <strong>bất kỳ ai
        mở link đều đọc được</strong>, và mỗi người xem lại tiêu một phần quota của bạn.
        Nên key phải ở phía dựng dữ liệu:</p>

        <ul>
          <li><strong>Chạy trên máy</strong> — file <code>.env</code> ở thư mục dự án</li>
          <li><strong>Chạy trên GitHub</strong> — Settings → Secrets and variables → Actions</li>
        </ul>

        <p>Sau khi đổi key, chạy lại <code>python pipeline/build.py</code> rồi tải lại trang.</p>
      </div>
    </div>

    <div class="card"><div class="card-title">Mô hình đang dùng</div>
      <div style="padding:10px 14px">
        <div class="kv"><span>Loại</span><b>Dixon-Coles</b></div>
        <div class="kv"><span>Số trận huấn luyện</span><b>${model.trained_on_matches}</b></div>
        <div class="kv"><span>Lợi thế sân nhà</span><b>×${Math.exp(model.home_advantage).toFixed(3)}</b></div>
        <div class="kv"><span>Hệ số rho</span><b>${model.rho}</b></div>
        <div class="kv"><span>Suy giảm thời gian (xi)</span><b>${model.xi}</b></div>
        <div class="kv"><span>Co ngót (ridge)</span><b>${model.ridge}</b></div>
        ${model.bridge_matches ? `<div class="kv"><span>Trận cầu nối liên giải</span><b>${model.bridge_matches}</b></div>` : ""}
      </div>
    </div>

    <div class="prose">
      <h3>Mô hình hoạt động thế nào</h3>
      <p>Mỗi đội có hai chỉ số: <strong>sức tấn công</strong> và <strong>sức phòng ngự</strong>,
      ước lượng từ toàn bộ kết quả các mùa gần nhất. Trận càng cũ trọng số càng nhỏ,
      nên phong độ gần đây có ảnh hưởng lớn hơn.</p>

      <p>Từ đó mô hình dựng <strong>ma trận xác suất cho mọi tỷ số</strong>. Mọi con số
      1X2, tài/xỉu, cả hai đội ghi bàn đều suy ra từ chính ma trận đó nên luôn nhất quán.</p>

      <h3>Khi có kèo nhà cái</h3>
      <p>Trận nào có kèo, app lấy xác suất từ <strong>thị trường</strong> làm con số chính.
      Kiểm định trên 2.270 trận: kèo đóng cửa đoán đúng <strong>54.9%</strong>, mô hình
      <strong>51.2%</strong>. Trộn mô hình vào giá thị trường làm dự báo <em>tệ đi</em>,
      nên app không trộn.</p>

      <p>Kèo cho ba con số 1X2 và một con số tài/xỉu. App giải ngược ra bộ ba
      (bàn thắng kỳ vọng nhà, khách, hệ số rho) tạo đúng cả bốn con số đó, rồi suy lại
      toàn bộ tỷ số — nên phần dự đoán tỷ số cũng mang thông tin thị trường.</p>

      <h3>Đọc phần nhận định</h3>
      <p>Mỗi cửa được so với <strong>tần suất nền</strong> của chính giải đó, không phải
      so với 50%. "Tài 1.5" đạt 86% nghe rất cao nhưng nền của nó đã là 80%, nên nó
      gần như không mang thông tin. Chỉ phần vượt lên trên mức thường mới đáng đọc.</p>

      <h3>Giới hạn — nên đọc</h3>
      <ul>
        <li>Mô hình <strong>không</strong> đánh bại được nhà cái. Điều này đã kiểm định trực tiếp.</li>
        <li>Khi mô hình bất đồng với thị trường, <strong>thị trường thường đúng hơn</strong>.</li>
        <li>Trận gắn nhãn "mơ hồ" là trận thực sự khó đoán — phần lớn trận bóng đá là như vậy.</li>
        <li>Bóng đá phương sai rất lớn. Dự đoán đúng 65% vẫn sai hơn 3 trên 10 lần.</li>
      </ul>
    </div>`;
}

/* ---------------- Điều hướng ---------------- */

const RENDERERS = {
  fixtures: renderFixtures,
  history: renderHistory,
  standings: renderStandings,
  strength: renderStrength,
  about: renderAbout,
};

async function render() {
  view().innerHTML = `<div class="loading"><div class="spinner"></div><p>Đang tải…</p></div>`;
  try {
    await RENDERERS[state.tab]();
  } catch (err) {
    view().innerHTML = `<div class="empty">Không tải được dữ liệu.<br><small>${escapeHtml(err.message)}</small></div>`;
  }
}

function renderLeagueTabs() {
  el("leagueTabs").innerHTML = state.leagues.map((l) =>
    `<button class="league-tab" role="tab" data-league="${escapeHtml(l.id)}"
       aria-selected="${l.id === state.league}">${escapeHtml(l.name)}</button>`).join("");

  el("leagueTabs").querySelectorAll(".league-tab").forEach((btn) =>
    btn.addEventListener("click", () => {
      state.league = btn.dataset.league;
      renderLeagueTabs();
      render();
    }));
}

async function init() {
  try {
    const index = await loadJSON(`${DATA}/index.json`);
    state.leagues = index.leagues;
    state.league = index.leagues[0]?.id;
    if (!state.league) throw new Error("Chưa có giải nào trong dữ liệu");
    renderLeagueTabs();
    await render();
  } catch (err) {
    view().innerHTML = `<div class="empty">Chưa có dữ liệu.<br>
      <small>Chạy <code>python pipeline/build.py</code> để sinh dữ liệu.</small><br>
      <small>${escapeHtml(err.message)}</small></div>`;
  }

  document.querySelectorAll(".tab").forEach((btn) =>
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.tab = btn.dataset.tab;
      render();
    }));

  el("sheet").addEventListener("click", (e) => {
    if (e.target.hasAttribute("data-close")) closeSheet();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !el("sheet").hidden) closeSheet();
  });

  el("refreshBtn").addEventListener("click", async () => {
    const btn = el("refreshBtn");
    btn.classList.add("spinning");
    state.cache.clear();
    await render();
    btn.classList.remove("spinning");
  });

  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("sw.js").catch(() => {
      // Service worker chỉ để xem offline; thất bại không ảnh hưởng app.
    });
  }
}

init();
