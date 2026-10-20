const BaseElement = globalThis.HTMLElement ?? class {};

const CSS = `
:host{--墨:#09090c;--纸:#faf8f2;--橙:#ff7a2f;--玻璃:rgba(8,8,12,.96);--弱:rgba(250,248,242,.57);--studio-rail:clamp(430px,33.333vw,640px);--studio-bottom:clamp(250px,28.15vh,304px);position:fixed;z-index:2147483000;right:16px;top:16px;color:var(--纸);font:12px/1.45 "Microsoft YaHei UI","PingFang SC","Noto Sans CJK SC",sans-serif;letter-spacing:.015em}
*{box-sizing:border-box}.launcher{position:relative;width:52px;height:52px;border:2px solid var(--墨);border-radius:0;background:var(--橙);color:var(--墨);box-shadow:0 0 0 2px var(--纸),4px 4px 0 var(--墨);cursor:pointer;font:900 20px/1 "Microsoft YaHei UI",sans-serif}.launcher::after{content:"遥测";position:absolute;right:-6px;bottom:-9px;padding:2px 4px;border:1px solid var(--墨);background:var(--纸);font-size:8px;letter-spacing:.08em}.launcher:focus-visible,.icon:focus-visible,select:focus-visible{outline:2px dashed var(--橙);outline-offset:3px}
.panel{position:relative;width:min(500px,calc(100vw - 32px));max-height:calc(100vh - 32px);overflow:auto;border:3px solid rgba(250,248,242,.94);border-radius:0;background:var(--玻璃);box-shadow:inset 0 0 0 1px var(--墨),5px 5px 0 rgba(9,9,12,.92);scrollbar-color:var(--橙) #17171a;scrollbar-width:thin}.panel::before{content:"物理分析";position:absolute;z-index:5;left:-1px;top:-1px;padding:4px 8px 3px;border-right:2px solid var(--墨);border-bottom:2px solid var(--墨);background:var(--橙);color:var(--墨);box-shadow:2px 2px 0 var(--墨);font-size:9px;font-weight:900;letter-spacing:.12em}.hidden{display:none}
.head{position:sticky;top:0;z-index:4;display:flex;align-items:center;gap:10px;padding:25px 12px 10px;border-bottom:1px solid rgba(250,248,242,.24);background:rgba(8,8,12,.94);backdrop-filter:blur(12px)}.signal{width:8px;height:8px;background:#65e493;box-shadow:0 0 11px #65e493}.titles{min-width:0;flex:1}.title{font-size:14px;font-weight:900;letter-spacing:.08em}.source{margin-top:3px;color:var(--弱);font:9px/1.35 "Cascadia Mono","Microsoft YaHei UI",monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.icon{width:48px;height:30px;border:1px solid rgba(250,248,242,.32);border-radius:0;background:transparent;color:var(--纸);cursor:pointer;font:800 9px "Microsoft YaHei UI",sans-serif;letter-spacing:.08em}.icon:hover{border-color:var(--橙);color:var(--橙)}
.body{padding:11px}.strip{display:grid;grid-template-columns:1fr auto;gap:10px;align-items:end;margin-bottom:7px}.label{display:block;margin-bottom:5px;color:var(--弱);font-size:9px;font-weight:800;letter-spacing:.08em}select{width:100%;height:32px;border:1px solid rgba(250,248,242,.38);border-radius:0;background:#111115;color:var(--纸);padding:0 9px;font:600 11px "Microsoft YaHei UI","Cascadia Mono",sans-serif}.rate{padding-bottom:7px;color:var(--橙);font:700 10px "Cascadia Mono","Microsoft YaHei UI",monospace}.pair{display:grid;grid-template-columns:1fr 1fr;gap:6px}.pair .section{margin-top:6px}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:5px}.grid.four{grid-template-columns:repeat(4,1fr)}.metric{min-width:0;padding:7px;border:1px solid rgba(250,248,242,.2);border-radius:0;background:rgba(9,9,12,.66)}.metric strong{display:block;margin-top:3px;color:var(--纸);font:800 14px/1 "Cascadia Mono",monospace}.metric span{color:var(--弱);font-size:8px;letter-spacing:.04em}.section{position:relative;margin-top:6px;padding:8px;border:1px solid rgba(250,248,242,.24);border-radius:0;background:rgba(8,8,12,.5)}.section-head{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:6px}.section h3{margin:0;font-size:10px;font-weight:900;letter-spacing:.06em}.badge{padding:2px 5px;border:1px solid rgba(250,248,242,.25);border-radius:0;color:var(--弱);font-size:8px}.plots{display:grid;grid-template-columns:repeat(3,1fr);gap:5px}.plot{min-width:0}.plot-top{display:flex;justify-content:space-between;color:rgba(250,248,242,.68);font:8px "Cascadia Mono",monospace}.plot svg{display:block;width:100%;height:45px;margin-top:3px;border:1px solid rgba(250,248,242,.08);background-color:#07070a;background-image:linear-gradient(rgba(250,248,242,.045) 1px,transparent 1px),linear-gradient(90deg,rgba(250,248,242,.045) 1px,transparent 1px);background-size:12px 12px}.plot path{fill:none;stroke-width:1.8;vector-effect:non-scaling-stroke}.force-stage{width:100%;height:145px;border:1px solid rgba(250,248,242,.12);background-color:#07070a;background-image:radial-gradient(rgba(250,248,242,.12) 1px,transparent 1px);background-size:14px 14px}.force-stage .axis{stroke:rgba(250,248,242,.2);stroke-width:1}.force-stage .body{fill:#f2f0e9;stroke:#09090c;stroke-width:3}.force-stage .force{stroke:#63dff0;stroke-width:5}.force-stage .torque{fill:none;stroke:#ff9a57;stroke-width:4}.force-stage text{fill:rgba(250,248,242,.7);font:9px "Cascadia Mono","Microsoft YaHei UI",monospace}.force-summary{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin-top:5px}.contact-row{display:grid;grid-template-columns:auto 1fr auto;gap:8px;align-items:center;color:rgba(250,248,242,.66);font-size:9px}.heat{height:9px;border:1px solid var(--墨);background:linear-gradient(90deg,#3178ff 0%,#6ad5ff 43%,#f6c45f 70%,#ff5c54 100%);position:relative;overflow:hidden}.heat::after{content:"";position:absolute;inset:0;background:#101014;transform-origin:right;transform:scaleX(var(--empty,.5))}.foot{display:flex;justify-content:space-between;gap:12px;margin-top:7px;color:rgba(250,248,242,.43);font:8px/1.35 "Cascadia Mono","Microsoft YaHei UI",monospace}.diagnostic{margin-top:6px;padding:6px;border-left:3px solid var(--橙);background:rgba(255,122,47,.12);color:#ffd0b8}.unavailable{color:rgba(250,248,242,.34)}
.diagnostics,.forces{display:contents}.studio-legend,.studio-notes,.studio-plots{display:none}.legend-list{display:grid;gap:8px;margin:0;padding:0;list-style:none;color:rgba(250,248,242,.72);font-size:9px}.legend-list i{display:inline-block;width:8px;height:8px;margin-right:8px;border-radius:50%;background:var(--dot,#63dff0);box-shadow:0 0 8px color-mix(in srgb,var(--dot,#63dff0) 55%,transparent)}
@media(min-width:1181px){
  :host([layout="studio"][open]){inset:0;width:100%;height:100%;pointer-events:none}
  :host([layout="studio"][open]) .panel{position:fixed;inset:0;width:100%;height:100%;max-height:none;overflow:hidden;border:0;background:transparent;box-shadow:none;pointer-events:none}
  :host([layout="studio"][open]) .panel::before{display:none}
  :host([layout="studio"][open]) .head{position:fixed;top:0;right:0;width:var(--studio-rail);height:58px;padding:0 18px;border-bottom:1px solid #2a3b4c;border-left:1px solid #2a3b4c;background:#091827;backdrop-filter:none;pointer-events:auto}
  :host([layout="studio"][open]) .title{font-size:18px}
  :host([layout="studio"][open]) .source{max-width:270px}
  :host([layout="studio"][open]) .body{position:fixed;inset:58px 0 0;display:grid;grid-template-columns:minmax(0,1fr) var(--studio-rail);grid-template-rows:minmax(0,1fr) var(--studio-bottom);padding:0;pointer-events:none}
  :host([layout="studio"][open]) .diagnostics{display:block;grid-column:2;grid-row:1/3;min-height:0;padding:12px 14px 34px;overflow:auto;border-left:1px solid #2a3b4c;background:#091827;pointer-events:auto;scrollbar-color:#466075 #091827}
  :host([layout="studio"][open]) .forces{display:grid;grid-column:1;grid-row:2;grid-template-columns:minmax(0,1.15fr) minmax(280px,.85fr);gap:12px;min-width:0;padding:10px 16px 12px;border-top:1px solid #2a3b4c;background:#091827;pointer-events:auto}
  :host([layout="studio"][open]) .forces .section{min-width:0;margin:0;background:#07121e}
  :host([layout="studio"][open]) .force-stage{height:135px}
  :host([layout="studio"][open]) .plots{display:none}
  :host([layout="studio"][open]) .studio-plots{display:grid;gap:9px}
  :host([layout="studio"][open]) .studio-chart{padding:9px 10px;border:1px solid #2a3b4c;background:#07121e}
  :host([layout="studio"][open]) .studio-chart-head,:host([layout="studio"][open]) .studio-chart-legend{display:flex;align-items:center;justify-content:space-between;gap:8px}
  :host([layout="studio"][open]) .studio-chart-head{font-size:11px;font-weight:800}.studio-chart-head small{color:var(--弱);font:8px "Cascadia Mono",monospace}
  :host([layout="studio"][open]) .studio-chart svg{display:block;width:100%;height:92px;margin:7px 0;border-top:1px solid #203448;border-bottom:1px solid #203448;background-image:linear-gradient(rgba(130,160,185,.08) 1px,transparent 1px);background-size:100% 23px}
  :host([layout="studio"][open]) .studio-chart path{fill:none;stroke-width:1.7;vector-effect:non-scaling-stroke}
  :host([layout="studio"][open]) .studio-chart-legend{justify-content:flex-start;color:rgba(250,248,242,.64);font:8px "Cascadia Mono",monospace}.studio-chart-legend span{display:flex;align-items:center;gap:4px}.studio-chart-legend i{width:10px;height:2px;background:var(--line)}
  :host([layout="studio"][open]) .studio-legend,:host([layout="studio"][open]) .studio-notes{display:block}
  :host([layout="studio"][open]) .studio-notes{margin-top:15px;color:rgba(250,248,242,.58);font-size:9px;line-height:1.8}
  :host([layout="studio"][open]) #diagnostic{position:fixed;right:14px;bottom:30px;width:calc(var(--studio-rail) - 28px);pointer-events:auto}
  :host([layout="studio"][open]) .foot{position:fixed;right:16px;bottom:9px;width:calc(var(--studio-rail) - 32px);margin:0;pointer-events:none}
}
@media(max-width:620px){:host{right:12px;top:auto;bottom:14px}.panel{max-height:calc(100vh - 28px)}.grid.four{grid-template-columns:1fr 1fr}.pair{grid-template-columns:1fr}.plots{grid-template-columns:1fr 1fr}.force-stage{height:160px}}
@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}
`;

const PHASE_LABELS = {
  settle: "稳定姿态",
  long_approach: "远距离轮滑加速",
  carve_turn: "连续快速甩弯",
  corner_exit: "保持速度出弯",
  crouch_gate: "运动中低头过竿",
  exit_gate: "过竿后连续起身",
  spin_360: "原地 360° 旋转",
  hard_brake: "急停制动",
  head_tilt_finish: "歪头定格",
  training: "训练采样",
  policy_run: "权重运行",
};

const JOINT_LABELS = {
  head_pitch: "头部俯仰",
  left_hip_pitch: "左髋俯仰",
  left_knee: "左膝",
  left_ankle: "左踝",
  right_hip_pitch: "右髋俯仰",
  right_knee: "右膝",
  right_ankle: "右踝",
  passive_LF_wheel: "左前轮（从动）",
  passive_LR_wheel: "左后轮（从动）",
  passive_RF_wheel: "右前轮（从动）",
  passive_RR_wheel: "右后轮（从动）",
};

const finite = (value) => Number.isFinite(value);
const fmt = (value, digits = 2) => finite(value) ? Number(value).toFixed(digits) : "—";
const phaseLabel = (value) => PHASE_LABELS[value] ?? value?.replaceAll?.("_", " ") ?? "";
const jointLabel = (value) => JOINT_LABELS[value] ?? value;
const sourceLabel = (source = "MuJoCo 遥测") => source
  .replace("Microduck roller showcase", "Microduck 轮滑实测")
  .replace("deterministic MuJoCo replay", "可复现 MuJoCo 回放")
  .replace("MuJoCo PhysicsRecorder", "MuJoCo 物理记录器");

function pathFor(values, width = 150, height = 42) {
  const valid = values.filter(finite);
  if (valid.length < 2) return "";
  let min = Math.min(...valid);
  let max = Math.max(...valid);
  if (Math.abs(max - min) < 1e-9) { min -= 1; max += 1; }
  const denominator = Math.max(1, values.length - 1);
  let drawing = false;
  return values.map((value, index) => {
    if (!finite(value)) { drawing = false; return ""; }
    const x = (index / denominator) * width;
    const y = height - ((value - min) / (max - min)) * height;
    const op = drawing ? "L" : "M";
    drawing = true;
    return `${op}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
}

function alignedPaths(series, width = 360, height = 96) {
  const valid = series.flat().filter(finite);
  if (!valid.length) return series.map(() => "");
  let min = Math.min(...valid);
  let max = Math.max(...valid);
  if (Math.abs(max - min) < 1e-9) { min -= 1; max += 1; }
  return series.map((values) => {
    const denominator = Math.max(1, values.length - 1);
    let drawing = false;
    return values.map((value, index) => {
      if (!finite(value)) { drawing = false; return ""; }
      const x = (index / denominator) * width;
      const y = height - ((value - min) / (max - min)) * height;
      const op = drawing ? "L" : "M";
      drawing = true;
      return `${op}${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
  });
}

function axisValue(joint, kind, index) {
  if (!joint) return null;
  return kind === "force"
    ? joint.wrench?.forceN?.[index]
    : joint.wrench?.torqueNm?.[index];
}

export class RLPhysicsOverlay extends BaseElement {
  constructor() {
    super();
    this._open = true;
    this._bus = null;
    this._unsubscribe = null;
    this._pending = false;
    this._frame = null;
    this._diagnostic = "";
    this._selectedJoint = "";
    this.attachShadow?.({ mode: "open" });
    if (this.shadowRoot) this.shadowRoot.innerHTML = `<style>${CSS}</style><div id="root"></div>`;
  }

  connectedCallback() { this.renderShell(); }

  set open(value) {
    this._open = Boolean(value);
    this.renderShell();
  }

  get open() { return this._open; }

  connect(bus) {
    this.disconnect();
    this._bus = bus;
    this._unsubscribe = bus.subscribe((frame) => {
      this._frame = frame;
      if (this._pending) return;
      this._pending = true;
      requestAnimationFrame(() => {
        this._pending = false;
        this.update();
      });
    });
    this.renderShell();
    return this;
  }

  disconnect() {
    this._unsubscribe?.();
    this._unsubscribe = null;
    this._bus = null;
  }

  disconnectedCallback() { this.disconnect(); }

  setDiagnostic(message = "") {
    this._diagnostic = message;
    this.update();
  }

  renderShell() {
    const root = this.shadowRoot?.getElementById("root");
    if (!root) return;
    this.toggleAttribute?.("open", this._open);
    if (!this._open) {
      root.innerHTML = `<button class="launcher" type="button" aria-label="打开物理遥测">力</button>`;
      root.querySelector("button")?.addEventListener("click", () => { this.open = true; });
      return;
    }
    root.innerHTML = `
      <aside class="panel" data-testid="rl-overlay-panel" aria-label="强化学习物理遥测">
        <header class="head"><span class="signal" aria-hidden="true"></span><div class="titles"><div class="title">关节诊断</div><div class="source" id="source">等待 MuJoCo 数据…</div></div><button class="icon" type="button" aria-label="收起物理遥测">收起</button></header>
        <div class="body">
          <div class="diagnostics">
          <div class="strip"><label><span class="label">查看关节</span><select id="joint" aria-label="选择关节"></select></label><div class="rate" id="rate">采样 0 Hz</div></div>
          <div class="grid four overview-only">
            <div class="metric"><span>位置 · rad</span><strong id="position">—</strong></div>
            <div class="metric"><span>位移 · rad</span><strong id="displacement">—</strong></div>
            <div class="metric"><span>速度 · rad/s</span><strong id="velocity">—</strong></div>
            <div class="metric"><span>驱动力矩 · N·m</span><strong id="torque">—</strong></div>
          </div>
          <div class="pair overview-only"><section class="section"><div class="section-head"><h3>子刚体姿态</h3><span class="badge">世界坐标 · deg</span></div><div class="grid"><div class="metric"><span>滚转 R</span><strong id="roll">—</strong></div><div class="metric"><span>俯仰 P</span><strong id="pitch">—</strong></div><div class="metric"><span>偏航 Y</span><strong id="yaw">—</strong></div></div></section><section class="section"><div class="section-head"><h3>广义关节力</h3><span class="badge">关节空间</span></div><div class="grid"><div class="metric"><span>驱动</span><strong id="q-act">—</strong></div><div class="metric"><span id="q-inertia-label">惯性</span><strong id="q-inertia">—</strong></div><div class="metric"><span>约束</span><strong id="q-con">—</strong></div></div></section></div>
          <section class="section curves-only"><div class="section-head"><h3>六维外力 / 力矩时域曲线</h3><span class="badge">仅真实物理通道</span></div><div class="plots" id="plots"></div><div class="studio-plots"><article class="studio-chart"><div class="studio-chart-head"><span>子刚体外力 · Force</span><small id="studio-force-range">—</small></div><svg viewBox="0 0 360 96" preserveAspectRatio="none" role="img" aria-label="Fx、Fy、Fz 合并时域曲线"><path id="studio-fx" stroke="#51b9f6"/><path id="studio-fy" stroke="#50e3b3"/><path id="studio-fz" stroke="#ff6b8b"/></svg><div class="studio-chart-legend"><span><i style="--line:#51b9f6"></i>Fx</span><span><i style="--line:#50e3b3"></i>Fy</span><span><i style="--line:#ff6b8b"></i>Fz</span></div></article><article class="studio-chart"><div class="studio-chart-head"><span>子刚体外力矩 · Torque</span><small id="studio-torque-range">—</small></div><svg viewBox="0 0 360 96" preserveAspectRatio="none" role="img" aria-label="Tx、Ty、Tz 合并时域曲线"><path id="studio-tx" stroke="#ffad32"/><path id="studio-ty" stroke="#9a6cff"/><path id="studio-tz" stroke="#ff6542"/></svg><div class="studio-chart-legend"><span><i style="--line:#ffad32"></i>Tx</span><span><i style="--line:#9a6cff"></i>Ty</span><span><i style="--line:#ff6542"></i>Tz</span></div></article></div></section>
          <section class="section studio-legend"><div class="section-head"><h3>图层说明</h3><span class="badge">实时物理量</span></div><ul class="legend-list"><li><i style="--dot:#63dff0"></i>青色：子刚体外力与平动分量</li><li><i style="--dot:#ff9a57"></i>橙色：驱动力矩与旋转方向</li><li><i style="--dot:#8a6cff"></i>紫色：惯性与约束力矩</li><li><i style="--dot:#ff5c72"></i>红蓝热力：接触摩擦强度</li></ul></section>
          </div>
          <div class="forces">
          <section class="section forces-only"><div class="section-head"><h3>子刚体外力矢量</h3><span class="badge">c-frame · 对数箭头</span></div><svg class="force-stage" viewBox="0 0 320 220" role="img" aria-label="当前关节对应子刚体的外力与外力矩方向"><defs><marker id="force-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7Z" fill="#63dff0"/></marker><marker id="torque-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7Z" fill="#ff9a57"/></marker></defs><line class="axis" x1="25" y1="110" x2="295" y2="110"/><line class="axis" x1="160" y1="20" x2="160" y2="200"/><text x="294" y="103">+X</text><text x="166" y="25">+Z</text><circle class="body" cx="160" cy="110" r="27"/><line id="force-vector" class="force" x1="160" y1="110" x2="160" y2="45" marker-end="url(#force-arrow)"/><path id="torque-vector" class="torque" d="M105 132 A62 62 0 0 1 215 132" marker-end="url(#torque-arrow)"/><text id="force-label" x="12" y="18">合力 — N</text><text id="torque-label" x="12" y="207">合力矩 — N·m</text></svg><div class="force-summary"><div class="metric"><span>Fx</span><strong id="field-fx">—</strong></div><div class="metric"><span>Fz</span><strong id="field-fz">—</strong></div><div class="metric"><span>T 合量</span><strong id="field-torque">—</strong></div></div></section>
          <section class="section forces-only"><div class="section-head"><h3>接触力分布</h3><span class="badge" id="contact-count">0 个接触点</span></div><div class="contact-row"><span>切向摩擦力</span><div class="heat" id="heat"></div><span id="friction">未提供</span></div><div class="studio-notes">原始值：qpos / qvel / qfrc / cfrc / contact<br>力箭头：方向真实、长度采用 log1p 映射<br>热力颜色：表示离散接触点的摩擦强度<br>缺失通道会明确标记，不使用动画伪造</div></section>
          </div>
          <div id="diagnostic"></div>
          <footer class="foot"><span id="clock">时间 0.00 s</span><span id="stats">有界缓存</span></footer>
        </div>
      </aside>`;
    root.querySelector(".icon")?.addEventListener("click", () => { this.open = false; });
    root.querySelector("#joint")?.addEventListener("change", (event) => {
      this._selectedJoint = event.target.value;
      this.update();
    });
    this.update();
  }

  update() {
    const root = this.shadowRoot;
    const frame = this._frame;
    if (!root || !this._open || !frame) return;
    const history = this._bus?.snapshot?.() ?? [frame];
    const select = root.getElementById("joint");
    const names = frame.joints?.map(({ name }) => name) ?? [];
    if (!names.includes(this._selectedJoint)) this._selectedJoint = names[0] ?? "";
    if (select && select.dataset.names !== names.join("|")) {
      select.replaceChildren(...names.map((name) => {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = jointLabel(name);
        return option;
      }));
      select.dataset.names = names.join("|");
    }
    if (select) select.value = this._selectedJoint;
    const joint = frame.joints?.find(({ name }) => name === this._selectedJoint);
    root.getElementById("source").textContent = frame.phase
      ? `${sourceLabel(frame.source)} · ${phaseLabel(frame.phase)}`
      : sourceLabel(frame.source);
    root.getElementById("position").textContent = fmt(joint?.positionRad, 3);
    root.getElementById("displacement").textContent = fmt(joint?.displacementRad, 3);
    root.getElementById("velocity").textContent = fmt(joint?.velocityRadps, 3);
    root.getElementById("torque").textContent = fmt(joint?.actuatorTorqueNm, 3);
    root.getElementById("roll").textContent = fmt(joint?.bodyRpyDeg?.[0], 1);
    root.getElementById("pitch").textContent = fmt(joint?.bodyRpyDeg?.[1], 1);
    root.getElementById("yaw").textContent = fmt(joint?.bodyRpyDeg?.[2], 1);
    root.getElementById("q-act").textContent = fmt(joint?.generalized?.actuator, 3);
    const inertiaAvailable = finite(joint?.generalized?.inertia);
    root.getElementById("q-inertia-label").textContent = inertiaAvailable ? "惯性" : "偏置";
    root.getElementById("q-inertia").textContent = fmt(
      inertiaAvailable ? joint.generalized.inertia : joint?.generalized?.bias,
      3,
    );
    root.getElementById("q-con").textContent = fmt(joint?.generalized?.constraint, 3);
    root.getElementById("clock").textContent = `时间 ${fmt(frame.timeS)} s`;
    const stats = this._bus?.stats?.() ?? {};
    root.getElementById("rate").textContent = `采样 ${stats.maxHz ?? 0} Hz`;
    root.getElementById("stats").textContent = `${stats.buffered ?? 0}/${stats.capacity ?? 0} 样本 · 跳过 ${stats.dropped ?? 0}`;
    const contactCount = frame.contacts?.count;
    root.getElementById("contact-count").textContent = finite(contactCount)
      ? `${contactCount} 个接触点`
      : "接触点未提供";
    const friction = frame.contacts?.frictionForceN;
    root.getElementById("friction").textContent = finite(friction) ? `${fmt(friction)} N` : "未提供";
    const heat = root.getElementById("heat");
    heat?.style.setProperty("--empty", finite(friction) ? String(Math.max(0, 1 - Math.log1p(friction) / 6)) : "1");

    const forceValues = joint?.wrench?.forceN ?? [];
    const torqueValues = joint?.wrench?.torqueNm ?? [];
    const forceAvailable = forceValues.some(finite);
    const torqueAvailable = torqueValues.some(finite);
    const fx = finite(forceValues[0]) ? forceValues[0] : 0;
    const fz = finite(forceValues[2]) ? forceValues[2] : 0;
    const projectedForce = Math.hypot(fx, fz);
    const totalForce = Math.hypot(...forceValues.map((value) => finite(value) ? value : 0));
    const totalTorque = Math.hypot(...torqueValues.map((value) => finite(value) ? value : 0));
    const forceLength = forceAvailable ? Math.min(100, 28 + Math.log1p(projectedForce) * 34) : 0;
    const forceLine = root.getElementById("force-vector");
    const forceDenominator = projectedForce || 1;
    forceLine?.setAttribute("x2", String(160 + (fx / forceDenominator) * forceLength));
    forceLine?.setAttribute("y2", String(110 - (fz / forceDenominator) * forceLength));
    forceLine?.setAttribute("opacity", forceAvailable ? "1" : "0.15");
    const torquePath = root.getElementById("torque-vector");
    torquePath?.setAttribute("d", (torqueValues[1] ?? 0) >= 0
      ? "M105 132 A62 62 0 0 1 215 132"
      : "M215 132 A62 62 0 0 0 105 132");
    torquePath?.setAttribute("opacity", torqueAvailable ? String(Math.min(1, 0.3 + Math.log1p(totalTorque) * 2)) : "0.15");
    root.getElementById("force-label").textContent = forceAvailable ? `合力 ${fmt(totalForce)} N` : "合力 未提供";
    root.getElementById("torque-label").textContent = torqueAvailable ? `合力矩 ${fmt(totalTorque)} N·m` : "合力矩 未提供";
    root.getElementById("field-fx").textContent = finite(forceValues[0]) ? fmt(forceValues[0]) : "—";
    root.getElementById("field-fz").textContent = finite(forceValues[2]) ? fmt(forceValues[2]) : "—";
    root.getElementById("field-torque").textContent = torqueAvailable ? fmt(totalTorque) : "—";

    const axes = [
      ["Fx", "force", 0, "#51b9f6"], ["Fy", "force", 1, "#50e3b3"], ["Fz", "force", 2, "#ff6b8b"],
      ["Tx", "torque", 0, "#ffad32"], ["Ty", "torque", 1, "#9a6cff"], ["Tz", "torque", 2, "#ff6542"],
    ];
    const series = axes.map(([, kind, index]) => history.map((sample) => axisValue(
        sample.joints?.find(({ name }) => name === this._selectedJoint), kind, index,
      )));
    root.getElementById("plots").innerHTML = axes.map(([label, kind, , color], seriesIndex) => {
      const values = series[seriesIndex];
      const current = values.at(-1);
      const unit = kind === "force" ? "N" : "N·m";
      return `<div class="plot"><div class="plot-top"><span>${label}</span><span class="${finite(current) ? "" : "unavailable"}">${fmt(current)} ${unit}</span></div><svg viewBox="0 0 150 42" role="img" aria-label="${label} 近期时域曲线"><path d="${pathFor(values)}" stroke="${color}"/></svg></div>`;
    }).join("");
    const studioPaths = [...alignedPaths(series.slice(0, 3)), ...alignedPaths(series.slice(3, 6))];
    ["studio-fx", "studio-fy", "studio-fz", "studio-tx", "studio-ty", "studio-tz"].forEach((id, index) => {
      root.getElementById(id)?.setAttribute("d", studioPaths[index]);
    });
    const forceHistory = series.slice(0, 3).flat().filter(finite);
    const torqueHistory = series.slice(3, 6).flat().filter(finite);
    root.getElementById("studio-force-range").textContent = forceHistory.length
      ? `±${fmt(Math.max(...forceHistory.map(Math.abs)))} N`
      : "未提供";
    root.getElementById("studio-torque-range").textContent = torqueHistory.length
      ? `±${fmt(Math.max(...torqueHistory.map(Math.abs)))} N·m`
      : "未提供";
    const diagnostic = root.getElementById("diagnostic");
    diagnostic.replaceChildren();
    if (this._diagnostic) {
      const message = document.createElement("div");
      message.className = "diagnostic";
      message.textContent = this._diagnostic;
      diagnostic.append(message);
    }
  }
}

export function definePhysicsOverlay(tagName = "rl-physics-overlay") {
  if (!globalThis.customElements) return RLPhysicsOverlay;
  if (!customElements.get(tagName)) customElements.define(tagName, RLPhysicsOverlay);
  return customElements.get(tagName);
}
