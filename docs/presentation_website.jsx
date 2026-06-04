import { useState, useEffect, useRef } from "react";
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, ReferenceLine, Tooltip, ResponsiveContainer, Cell } from "recharts";

// ─── Data ────────────────────────────────────────────────────────────────────

const FRONTIER = [
  { id:1, E:0.9684, sigma:0.0176, label:"Wise/USD (6 steps, 3 FX)", regime:"european", steps:6 },
  { id:2, E:0.9923, sigma:0.0183, label:"Revolut route (6 steps, 3 FX)", regime:"european", steps:6 },
  { id:3, E:0.9927, sigma:0.0224, label:"N26 route (6 steps, 3 FX)", regime:"european", steps:6 },
  { id:4, E:0.9961, sigma:0.0252, label:"Santander route (5 steps, 3 FX)", regime:"european", steps:5 },
  { id:5, E:1.1326, sigma:0.0344, label:"BancoDoBrasil route (5 steps, 2 FX)", regime:"brl", steps:5 },
  { id:6, E:1.1345, sigma:0.0392, label:"BancoNacional route — greedy ★ (4 steps, 2 FX)", regime:"brl", steps:4 },
];

const DOMINATED = [
  { E:0.963, sigma:0.022 }, { E:0.962, sigma:0.023 }, { E:0.961, sigma:0.025 }, { E:0.965, sigma:0.023 },
  { E:1.127, sigma:0.038 }, { E:1.124, sigma:0.040 }, { E:1.126, sigma:0.041 },
  { E:1.124, sigma:0.043 }, { E:1.125, sigma:0.050 }, { E:1.124, sigma:0.055 },
];

const SECTIONS = ["Problem","Phase 1","Phase 2","Phase 3","Phase 4","Results"];
const SECTION_IDS = ["problem","phase1","phase2","phase3","phase4","results"];

// ─── Helpers ─────────────────────────────────────────────────────────────────

const getOptimal = (lambda) =>
  FRONTIER.reduce((best, p) => {
    const s = p.E - (lambda / 2) * p.sigma ** 2;
    const bs = best.E - (lambda / 2) * best.sigma ** 2;
    return s > bs ? p : best;
  }, FRONTIER[0]);

// ─── Micro-components ─────────────────────────────────────────────────────────

const Tag = ({ phase }) => {
  const colors = ["","#534AB7","#1D9E75","#D85A30","#BA7517"];
  const bgs = ["","#EEEDFE","#E1F5EE","#FAECE7","#FAEEDA"];
  return (
    <span style={{ display:"inline-block", padding:"2px 10px", borderRadius:20,
      fontSize:11, fontWeight:500, letterSpacing:".05em",
      background:bgs[phase], color:colors[phase] }}>
      PHASE {phase}
    </span>
  );
};

const Eq = ({ label, text, accent="#534AB7" }) => (
  <div style={{ borderLeft:`3px solid ${accent}`, paddingLeft:12, margin:"12px 0",
    background:"var(--color-background-secondary)", borderRadius:"0 var(--border-radius-md) var(--border-radius-md) 0",
    padding:"10px 12px 10px 14px" }}>
    {label && <div style={{ fontSize:11, color:accent, fontWeight:500, marginBottom:4 }}>{label}</div>}
    <div style={{ fontFamily:"var(--font-mono)", fontSize:13, lineHeight:1.6, color:"var(--color-text-primary)" }}>{text}</div>
  </div>
);

const Stat = ({ label, value, accent="#534AB7" }) => (
  <div style={{ textAlign:"center", padding:"14px 20px",
    background:"var(--color-background-secondary)", borderRadius:"var(--border-radius-lg)",
    borderTop:`3px solid ${accent}` }}>
    <div style={{ fontSize:22, fontWeight:500, color:accent, fontFamily:"var(--font-mono)" }}>{value}</div>
    <div style={{ fontSize:12, color:"var(--color-text-secondary)", marginTop:2 }}>{label}</div>
  </div>
);

const SectionWrap = ({ id, children }) => (
  <section id={id} style={{ padding:"3rem 0", borderBottom:"0.5px solid var(--color-border-tertiary)" }}>
    {children}
  </section>
);

const H2 = ({ children }) => (
  <h2 style={{ fontFamily:"var(--font-serif)", fontSize:22, fontWeight:500,
    color:"var(--color-text-primary)", margin:"0 0 6px" }}>{children}</h2>
);

const Sub = ({ children }) => (
  <p style={{ fontSize:13, color:"var(--color-text-secondary)", margin:"0 0 18px" }}>{children}</p>
);

const FigBox = ({ name }) => (
  <div style={{ border:"1px dashed var(--color-border-secondary)", borderRadius:"var(--border-radius-md)",
    background:"var(--color-background-secondary)", padding:"16px", textAlign:"center",
    display:"flex", flexDirection:"column", alignItems:"center", gap:6, minHeight:80 }}>
    <i className="ti ti-photo" style={{ fontSize:20, color:"var(--color-text-tertiary)" }} aria-hidden="true"/>
    <span style={{ fontFamily:"var(--font-mono)", fontSize:11, color:"var(--color-text-secondary)" }}>
      {name}
    </span>
  </div>
);

// ─── Nav ─────────────────────────────────────────────────────────────────────

const Nav = ({ active }) => {
  const scroll = (id) => {
    document.getElementById(id)?.scrollIntoView({ behavior:"smooth" });
  };
  return (
    <nav style={{ position:"sticky", top:0, zIndex:50,
      background:"var(--color-background-primary)", borderBottom:"0.5px solid var(--color-border-tertiary)",
      display:"flex", alignItems:"center", justifyContent:"space-between", padding:"0 1rem",
      height:44, gap:8 }}>
      <span style={{ fontFamily:"var(--font-serif)", fontSize:13, fontWeight:500, color:"var(--color-text-primary)", whiteSpace:"nowrap" }}>
        CS109 · Marco Paes
      </span>
      <div style={{ display:"flex", gap:4 }}>
        {SECTIONS.map((s,i) => (
          <button key={s} onClick={() => scroll(SECTION_IDS[i])} style={{
            background: active===i ? "var(--color-background-secondary)" : "transparent",
            border: active===i ? "0.5px solid var(--color-border-secondary)" : "none",
            borderRadius:"var(--border-radius-md)", padding:"4px 10px", cursor:"pointer",
            fontSize:12, color: active===i ? "var(--color-text-primary)" : "var(--color-text-secondary)",
            fontWeight: active===i ? 500 : 400,
          }}>{s}</button>
        ))}
      </div>
    </nav>
  );
};

// ─── Hero ─────────────────────────────────────────────────────────────────────

const Hero = () => (
  <div style={{ padding:"3rem 0 2rem", textAlign:"center" }}>
    <div style={{ display:"inline-block", padding:"3px 12px", borderRadius:20,
      background:"#EEEDFE", color:"#534AB7", fontSize:11, fontWeight:500,
      letterSpacing:".05em", marginBottom:16 }}>
      EU-MERCOSUR · STANFORD CS109 · JUNE 2026
    </div>
    <h1 style={{ fontFamily:"var(--font-serif)", fontSize:28, fontWeight:500,
      lineHeight:1.3, color:"var(--color-text-primary)", margin:"0 auto 8px", maxWidth:540 }}>
      Probabilistic Routing for Cross-Border Payments
    </h1>
    <p style={{ fontSize:15, color:"var(--color-text-secondary)", margin:"0 0 28px" }}>
      A Google Maps for Money — built on MLE, Ornstein–Uhlenbeck dynamics, and mean-variance optimization
    </p>
    <div style={{ display:"flex", justifyContent:"center", gap:12, flexWrap:"wrap" }}>
      <Stat label="above commercial" value="+13.4%" accent="#1D9E75"/>
      <Stat label="one-sided z-statistic" value="Z = 69.7" accent="#534AB7"/>
      <Stat label="p-value" value="p < 10⁻¹⁰" accent="#534AB7"/>
      <Stat label="Pareto frontier paths" value="6 / 16" accent="#D85A30"/>
    </div>
  </div>
);

// ─── Problem ─────────────────────────────────────────────────────────────────

const ProblemSection = () => (
  <SectionWrap id="problem">
    <H2>The Problem</H2>
    <Sub>Why 10–15% of every cross-border transfer disappears</Sub>
    <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16 }}>
      <div style={{ background:"var(--color-background-secondary)", borderRadius:"var(--border-radius-lg)",
        padding:"16px", borderLeft:"3px solid #534AB7" }}>
        <div style={{ fontSize:12, fontWeight:500, color:"#534AB7", marginBottom:8 }}>PERSONAL MOTIVATION</div>
        <p style={{ fontSize:13, lineHeight:1.6, color:"var(--color-text-secondary)", margin:0 }}>
          Sending money between Paraguay, Brazil, and the US consistently loses 10–15% to opaque FX
          markups and transfer fees compounded at every hop of the correspondent banking chain.
          For a student wiring tuition funds, it's a nuisance. For a small business in the new
          EU-Mercosur corridor, it's a structural barrier.
        </p>
      </div>
      <div style={{ background:"var(--color-background-secondary)", borderRadius:"var(--border-radius-lg)",
        padding:"16px", borderLeft:"3px solid #1D9E75" }}>
        <div style={{ fontSize:12, fontWeight:500, color:"#1D9E75", marginBottom:8 }}>EU-MERCOSUR · MAY 2026</div>
        <p style={{ fontSize:13, lineHeight:1.6, color:"var(--color-text-secondary)", margin:"0 0 10px" }}>
          The EU-Mercosur Interim Trade Agreement entered provisional application on May 1, 2026 —
          one month before this submission. 780 million consumers, EUR ↔ BRL ↔ PYG corridors,
          no unified payment infrastructure.
        </p>
        <div style={{ fontFamily:"var(--font-mono)", fontSize:12, color:"#1D9E75" }}>
          €4B+/yr in tariff savings · 780M consumer zone
        </div>
      </div>
    </div>
    <div style={{ marginTop:12 }}>
      <FigBox name="network_all_nodes.png — institution-currency graph (31 nodes, 12 banks, 4 currencies)" />
    </div>
  </SectionWrap>
);

// ─── Phase 1 ─────────────────────────────────────────────────────────────────

const MLESection = () => (
  <SectionWrap id="phase1">
    <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:4 }}>
      <H2>Maximum Likelihood Distribution Fitting</H2>
      <Tag phase={1}/>
    </div>
    <Sub>Gaussian vs Log-Normal on corridor volatility — and why the Jacobian matters</Sub>
    <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16 }}>
      <div>
        <Eq label="Log-Normal log-likelihood (with Jacobian)" accent="#534AB7"
          text={"ℓ(μ,σ²) = −Σ log xᵢ  −  n/2 · log(2πσ²)  −  1/(2σ²) · Σ(log xᵢ − μ)²"} />
        <p style={{ fontSize:13, lineHeight:1.6, color:"var(--color-text-secondary)" }}>
          The leading term −Σ log xᵢ is the Jacobian of Y = log X. Without it, the two
          log-likelihoods are not on the same measure and cannot be compared.
          With it, the comparison is exact.
        </p>
      </div>
      <div>
        <div style={{ display:"flex", flexDirection:"column", gap:6 }}>
          {[
            { pair:"USD → BRL", delta:"+1818" }, { pair:"USD → PYG", delta:"+1829" },
            { pair:"USD → EUR", delta:"+1429" }, { pair:"EUR → BRL", delta:"+1508" },
            { pair:"BRL → PYG", delta:"+784" }, { pair:"EUR → PYG", delta:"+858" },
          ].map(row => (
            <div key={row.pair} style={{ display:"flex", justifyContent:"space-between",
              alignItems:"center", padding:"6px 10px",
              background:"var(--color-background-secondary)", borderRadius:"var(--border-radius-md)",
              fontSize:12 }}>
              <span style={{ fontFamily:"var(--font-mono)", color:"var(--color-text-primary)" }}>{row.pair}</span>
              <span style={{ fontWeight:500, color:"#534AB7" }}>Δℓ = {row.delta} ★ Log-Normal</span>
            </div>
          ))}
        </div>
        <div style={{ marginTop:8, padding:"8px 12px", background:"#EEEDFE",
          borderRadius:"var(--border-radius-md)", fontSize:12, color:"#534AB7", fontWeight:500 }}>
          Log-Normal wins all 12 corridors. Δℓ ∈ [+784, +1829].
        </div>
        <div style={{ marginTop:8 }}><FigBox name="mle_corridor_fits.png" /></div>
      </div>
    </div>
  </SectionWrap>
);

// ─── Phase 2 + Bridge ─────────────────────────────────────────────────────────

const OUSection = () => (
  <SectionWrap id="phase2">
    <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:4 }}>
      <H2>Ornstein–Uhlenbeck Log-Volatility Dynamics</H2>
      <Tag phase={2}/>
    </div>
    <Sub>Closed-form MLE via AR(1) — no numerical optimizer required</Sub>
    <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16, marginBottom:16 }}>
      <div>
        <Eq label="OU SDE → stationary distribution" accent="#1D9E75"
          text={"dYₜ = θ(μ − Yₜ) dt + σ dWₜ\n\nStationary:  Y∞ ~ N( μ,  σ²/2θ )"} />
        <Eq label="AR(1) closed-form OLS" accent="#1D9E75"
          text={"Yₜ₊₁ = â·Yₜ + b̂ + εₜ\n\nâ = S_XY / S_XX,   θ̂ = −log(â)/Δt"} />
      </div>
      <div>
        <p style={{ fontSize:13, lineHeight:1.6, color:"var(--color-text-secondary)" }}>
          Signed returns give â ≈ 0 (weak-form efficiency). Log-volatility yₜ = log|rₜ| clusters —
          the OU process captures this. The transition density becomes an AR(1) model solvable
          by OLS in closed form, with continuous-time parameters recovered analytically.
        </p>
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:6 }}>
          {[["θ range","1.42 – 2.29 /day"],["â range","0.10 – 0.24"],["Half-lives","0.3 – 0.5 days"],["Clipped","0 / 12 corridors"]].map(([l,v]) => (
            <div key={l} style={{ background:"var(--color-background-secondary)",
              borderRadius:"var(--border-radius-md)", padding:"8px 10px", fontSize:12 }}>
              <div style={{ color:"var(--color-text-secondary)" }}>{l}</div>
              <div style={{ fontWeight:500, fontFamily:"var(--font-mono)", color:"#1D9E75" }}>{v}</div>
            </div>
          ))}
        </div>
        <div style={{ marginTop:8 }}><FigBox name="ou_diagnostics.png" /></div>
      </div>
    </div>

    {/* The Bridge */}
    <div style={{ background:"linear-gradient(to right, #EEEDFE, #E1F5EE)",
      borderRadius:"var(--border-radius-lg)", padding:"18px 20px", border:"0.5px solid var(--color-border-secondary)" }}>
      <div style={{ fontSize:12, fontWeight:500, letterSpacing:".06em", color:"#534AB7", marginBottom:8 }}>
        THE BRIDGE: PHASE 1 ↔ PHASE 2
      </div>
      <div style={{ display:"grid", gridTemplateColumns:"1fr auto 1fr", gap:12, alignItems:"center" }}>
        <div style={{ background:"white", borderRadius:"var(--border-radius-md)", padding:"12px",
          border:"0.5px solid #AFA9EC" }}>
          <div style={{ fontSize:11, color:"#534AB7", fontWeight:500, marginBottom:4 }}>PHASE 1 — MLE MARGINAL</div>
          <div style={{ fontFamily:"var(--font-mono)", fontSize:13, color:"#3C3489" }}>
            |r| ~ LogNormal(μ̂, σ̂²)
          </div>
        </div>
        <div style={{ textAlign:"center", fontSize:20, color:"#534AB7" }}>≡</div>
        <div style={{ background:"white", borderRadius:"var(--border-radius-md)", padding:"12px",
          border:"0.5px solid #9FE1CB" }}>
          <div style={{ fontSize:11, color:"#1D9E75", fontWeight:500, marginBottom:4 }}>PHASE 2 — OU STATIONARY</div>
          <div style={{ fontFamily:"var(--font-mono)", fontSize:13, color:"#085041" }}>
            exp( N(μ, σ²/2θ) ) = LogNormal
          </div>
        </div>
      </div>
      <p style={{ margin:"10px 0 0", fontSize:12, color:"var(--color-text-secondary)", lineHeight:1.5 }}>
        Because Y∞ is Gaussian, exp(Y∞) is Log-Normal — exactly the distribution Phase 1 fit.
        Two phases, one coherent model: Phase 1 estimates the marginal, Phase 2 adds dynamics.
      </p>
    </div>
  </SectionWrap>
);

// ─── Phase 3 ─────────────────────────────────────────────────────────────────

const FrontierSection = () => {
  const [lambda, setLambda] = useState(0);
  const optimal = getOptimal(lambda);
  const regime = optimal.regime;
  const regimeColor = regime === "brl" ? "#1D9E75" : "#D85A30";
  const score = (optimal.E - (lambda/2) * optimal.sigma**2).toFixed(4);

  const L1 = 11, L2 = 331, LMAX = 500;

  const scatterData = [
    ...DOMINATED.map(d => ({ ...d, type:"dominated" })),
    ...FRONTIER.map(f => ({ sigma: f.sigma, E: f.E, type:"frontier", id: f.id,
      isOptimal: f.id === optimal.id })),
  ];

  const CustomDot = (props) => {
    const { cx, cy, payload } = props;
    if (!cx || !cy) return null;
    if (payload.type === "dominated") return <circle cx={cx} cy={cy} r={4} fill="#bdbdbd" opacity={0.6}/>;
    const isGreedy = payload.id === 6;
    const isOpt = payload.isOptimal;
    if (isOpt) return <path d={`M${cx},${cy-7} L${cx+6},${cy+5} L${cx-6},${cy+5} Z`} fill="#534AB7" stroke="white" strokeWidth={1.5}/>;
    if (isGreedy) return <circle cx={cx} cy={cy} r={5} fill="none" stroke="#D85A30" strokeWidth={2}/>;
    return <circle cx={cx} cy={cy} r={4.5} fill="#534AB7" opacity={0.8}/>;
  };

  const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null;
    const d = payload[0]?.payload;
    return (
      <div style={{ background:"var(--color-background-primary)", border:"0.5px solid var(--color-border-secondary)",
        borderRadius:"var(--border-radius-md)", padding:"8px 12px", fontSize:11 }}>
        <div>E[ρ] = {d.E?.toFixed(4)}</div>
        <div>σ[ρ] = {d.sigma?.toFixed(4)}</div>
        {d.type === "frontier" && FRONTIER.find(f=>f.id===d.id) &&
          <div style={{ color:"#534AB7", marginTop:4, fontWeight:500, maxWidth:160 }}>
            {FRONTIER.find(f=>f.id===d.id).label}
          </div>}
      </div>
    );
  };

  return (
    <SectionWrap id="phase3">
      <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:4 }}>
        <H2>Pareto Frontier and Mean-Variance Optimality</H2>
        <Tag phase={3}/>
      </div>
      <Sub>16 feasible paths evaluated — 6 on the Pareto frontier — two risk regimes</Sub>

      <div style={{ display:"grid", gridTemplateColumns:"1.1fr 1fr", gap:16, marginBottom:16 }}>
        <div>
          <div style={{ fontSize:12, color:"var(--color-text-secondary)", marginBottom:6 }}>
            Pareto frontier in (σ[ρ], E[ρ]) space · △ = current λ-optimal · ○ = greedy choice
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <ScatterChart margin={{ top:8, right:16, bottom:28, left:16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-tertiary)" />
              <XAxis dataKey="sigma" type="number" domain={[0.014, 0.06]} tick={{ fontSize:10 }}
                label={{ value:"Risk: σ[ρ]", position:"bottom", offset:10, fontSize:11 }} />
              <YAxis dataKey="E" type="number" domain={[0.955, 1.15]} tick={{ fontSize:10 }}
                label={{ value:"Return: E[ρ]", angle:-90, position:"insideLeft", offset:10, fontSize:11 }} />
              <ReferenceLine y={1} stroke="#888" strokeDasharray="4 4"
                label={{ value:"benchmark ρ=1", position:"left", fontSize:10, fill:"#888" }} />
              <Tooltip content={<CustomTooltip />} />
              <Scatter data={scatterData} shape={<CustomDot />} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <div style={{ display:"flex", flexDirection:"column", gap:10 }}>
          <Eq label="Mean-variance utility (Pratt–Arrow)" accent="#D85A30"
            text={"U_MV(X) = E[X] − (λ/2) · Var[X]"} />
          <Eq label="Critical risk-aversion crossover" accent="#D85A30"
            text={"λ*ᵢ→ⱼ = 2(Eᵢ − Eⱼ) / (σᵢ² − σⱼ²)"} />
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:6 }}>
            <div style={{ background:"#E1F5EE", borderRadius:"var(--border-radius-md)", padding:"10px 12px" }}>
              <div style={{ fontSize:11, color:"#1D9E75", fontWeight:500 }}>BRL-corridor</div>
              <div style={{ fontFamily:"var(--font-mono)", fontSize:12, color:"#085041" }}>E[ρ] ≈ 1.13</div>
              <div style={{ fontSize:11, color:"#0F6E56" }}>σ ∈ [0.034, 0.039]</div>
            </div>
            <div style={{ background:"#FAECE7", borderRadius:"var(--border-radius-md)", padding:"10px 12px" }}>
              <div style={{ fontSize:11, color:"#D85A30", fontWeight:500 }}>European-routed</div>
              <div style={{ fontFamily:"var(--font-mono)", fontSize:12, color:"#4A1B0C" }}>E[ρ] ∈ [0.97, 0.99]</div>
              <div style={{ fontSize:11, color:"#993C1D" }}>σ ∈ [0.018, 0.025]</div>
            </div>
          </div>
        </div>
      </div>

      {/* Lambda slider */}
      <div style={{ background:"var(--color-background-secondary)", borderRadius:"var(--border-radius-lg)",
        padding:"18px 20px", border:"0.5px solid var(--color-border-secondary)" }}>
        <div style={{ fontSize:12, fontWeight:500, color:"var(--color-text-primary)", marginBottom:12 }}>
          INTERACTIVE: drag λ to see which path is mean-variance optimal
        </div>
        <div style={{ position:"relative", marginBottom:8 }}>
          <input type="range" min={0} max={LMAX} step={1} value={lambda}
            onChange={e => setLambda(+e.target.value)} style={{ width:"100%" }}/>
          {/* threshold markers */}
          {[{v:L1,label:"λ*₁≈11"},{v:L2,label:"λ*₂≈331"}].map(({v,label}) => (
            <div key={v} style={{ position:"absolute", left:`${v/LMAX*100}%`, top:-18,
              transform:"translateX(-50%)", fontSize:10, color:"var(--color-text-tertiary)",
              whiteSpace:"nowrap", pointerEvents:"none" }}>
              ▼ {label}
            </div>
          ))}
        </div>
        <div style={{ display:"flex", justifyContent:"space-between", fontSize:11,
          color:"var(--color-text-tertiary)", marginBottom:12 }}>
          <span>λ = 0 (risk-neutral)</span>
          <span style={{ fontFamily:"var(--font-mono)", fontWeight:500, fontSize:13,
            color:"var(--color-text-primary)" }}>λ = {lambda}</span>
          <span>λ = 500 (risk-averse)</span>
        </div>
        <div style={{ display:"grid", gridTemplateColumns:"auto 1fr", gap:12, alignItems:"center" }}>
          <div style={{ width:12, height:12, borderRadius:"50%", background:regimeColor, flexShrink:0 }}/>
          <div>
            <div style={{ fontSize:13, fontWeight:500, color:regimeColor }}>{optimal.label}</div>
            <div style={{ fontFamily:"var(--font-mono)", fontSize:12, color:"var(--color-text-secondary)", marginTop:2 }}>
              E[ρ] = {optimal.E.toFixed(4)}   σ[ρ] = {optimal.sigma.toFixed(4)}
              &nbsp;&nbsp;|&nbsp;&nbsp;score = {optimal.E.toFixed(4)} − {lambda}/2 × {optimal.sigma.toFixed(4)}² = <strong style={{ color:regimeColor }}>{score}</strong>
            </div>
          </div>
        </div>
        <div style={{ display:"flex", gap:8, marginTop:12 }}>
          {[{lo:0,hi:11,label:"λ < 11: greedy path optimal",color:"#1D9E75"},
            {lo:11,hi:331,label:"11 ≤ λ < 331: lower-σ BRL route",color:"#534AB7"},
            {lo:331,hi:500,label:"λ ≥ 331: European low-variance",color:"#D85A30"}].map(r => (
            <div key={r.lo} style={{ flex:1, padding:"6px 8px", borderRadius:"var(--border-radius-md)",
              background: lambda >= r.lo && lambda < (r.hi === 500 ? 501 : r.hi) ? r.color+"22" : "var(--color-background-secondary)",
              border:`1px solid ${lambda >= r.lo && lambda < (r.hi === 500 ? 501 : r.hi) ? r.color : "var(--color-border-tertiary)"}`,
              fontSize:11, color: lambda >= r.lo && lambda < (r.hi === 500 ? 501 : r.hi) ? r.color : "var(--color-text-tertiary)",
              fontWeight: lambda >= r.lo && lambda < (r.hi === 500 ? 501 : r.hi) ? 500 : 400,
              transition:"all .2s", textAlign:"center" }}>
              {r.label}
            </div>
          ))}
        </div>
      </div>
    </SectionWrap>
  );
};

// ─── Phase 4 ─────────────────────────────────────────────────────────────────

const CLTSection = () => (
  <SectionWrap id="phase4">
    <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:4 }}>
      <H2>CLT-Based Statistical Inference</H2>
      <Tag phase={4}/>
    </div>
    <Sub>n = 500 Monte Carlo realizations — 95% confidence interval and one-sample z-test</Sub>
    <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16 }}>
      <div>
        <Eq label="CLT confidence interval" accent="#BA7517"
          text={"ρ̄ ~ N( E[ρ],  Var[ρ]/n )  →  ρ̄ ± 1.96 · σ̂_ρ/√n"} />
        <Eq label="One-sample Wald z-test  H₀: E[ρ] = 1" accent="#BA7517"
          text={"Z = (ρ̄ − 1) / (σ̂_ρ / √n)  ~  N(0,1)  under H₀"} />
        <div style={{ marginTop:8 }}><FigBox name="stats_distributions.png" /></div>
      </div>
      <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
        {[
          { label:"E[ρ]", value:"1.1336", accent:"#1D9E75" },
          { label:"95% Confidence Interval", value:"[1.1299,  1.1374]", accent:"#534AB7" },
          { label:"Standard Error", value:"SE = 0.00192", accent:"#534AB7" },
          { label:"Z-statistic", value:"Z = 69.69", accent:"#BA7517" },
          { label:"p-value (one-sided)", value:"< 10⁻¹⁰", accent:"#BA7517" },
        ].map(row => (
          <div key={row.label} style={{ display:"flex", justifyContent:"space-between", alignItems:"center",
            padding:"8px 12px", background:"var(--color-background-secondary)", borderRadius:"var(--border-radius-md)" }}>
            <span style={{ fontSize:12, color:"var(--color-text-secondary)" }}>{row.label}</span>
            <span style={{ fontFamily:"var(--font-mono)", fontSize:13, fontWeight:500, color:row.accent }}>{row.value}</span>
          </div>
        ))}
        <div style={{ padding:"10px 14px", background:"#E1F5EE", borderRadius:"var(--border-radius-md)",
          fontSize:13, color:"#0F6E56", fontWeight:500 }}>
          REJECT H₀ — overwhelming evidence the routed path beats commercial (13.4% improvement, p&lt;10⁻¹⁰)
        </div>
      </div>
    </div>
  </SectionWrap>
);

// ─── Results + Sensitivity ────────────────────────────────────────────────────

const ResultsSection = () => (
  <SectionWrap id="results">
    <H2>Sensitivity Analysis & Conclusion</H2>
    <Sub>Robustness to structural priors — and a four-phase summary</Sub>
    <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16, marginBottom:16 }}>
      <div>
        <div style={{ fontSize:12, fontWeight:500, color:"var(--color-text-secondary)", marginBottom:8 }}>
          ROBUSTNESS: ±20% TIER PERTURBATIONS · 10 TRIALS
        </div>
        {[
          { label:"Greedy on Pareto frontier", value:"7 / 10 trials", color:"#1D9E75" },
          { label:"E[ρ] range across trials", value:"[1.127, 1.135]", color:"#534AB7" },
          { label:"Minimum improvement", value:"> 12.7% above commercial", color:"#534AB7" },
          { label:"Bimodal structure persists", value:"10 / 10 trials", color:"#1D9E75" },
        ].map(r => (
          <div key={r.label} style={{ display:"flex", justifyContent:"space-between",
            padding:"7px 10px", borderBottom:"0.5px solid var(--color-border-tertiary)", fontSize:12 }}>
            <span style={{ color:"var(--color-text-secondary)" }}>{r.label}</span>
            <span style={{ fontFamily:"var(--font-mono)", fontWeight:500, color:r.color }}>{r.value}</span>
          </div>
        ))}
        <div style={{ marginTop:8 }}><FigBox name="sensitivity_analysis.png" /></div>
      </div>
      <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
        <div style={{ fontSize:12, fontWeight:500, color:"var(--color-text-secondary)", marginBottom:2 }}>
          SUMMARY: FOUR PHASES, ONE COHERENT MODEL
        </div>
        {[
          { phase:1, text:"Log-Normal MLE wins all 12 corridors (Δℓ ∈ [784, 1829])" },
          { phase:2, text:"OU stationary = exp(Gaussian) = Phase 1 Log-Normal — same model" },
          { phase:3, text:"Greedy is λ=0 Pareto optimal; λ*₁≈11, λ*₂≈331 derived analytically" },
          { phase:4, text:"E[ρ]=1.134, 95% CI [1.130, 1.137], Z=70, p<10⁻¹⁰" },
        ].map(r => (
          <div key={r.phase} style={{ display:"flex", gap:10, alignItems:"flex-start",
            padding:"10px 12px", background:"var(--color-background-secondary)",
            borderRadius:"var(--border-radius-md)" }}>
            <Tag phase={r.phase}/>
            <span style={{ fontSize:12, color:"var(--color-text-secondary)", lineHeight:1.5 }}>{r.text}</span>
          </div>
        ))}
        <div style={{ padding:"12px 14px", background:"#EEEDFE", borderRadius:"var(--border-radius-lg)",
          borderLeft:"3px solid #534AB7", marginTop:4 }}>
          <div style={{ fontSize:13, fontWeight:500, color:"#534AB7" }}>
            A mathematically grounded Google Maps for Money
          </div>
          <div style={{ fontSize:12, color:"#7F77DD", marginTop:4, lineHeight:1.5 }}>
            As EU-Mercosur trade scales across EUR ↔ BRL ↔ PYG corridors, this framework
            routes payments with provable statistical guarantees.
          </div>
        </div>
      </div>
    </div>
  </SectionWrap>
);

// ─── App ──────────────────────────────────────────────────────────────────────

export default function App() {
  const [active, setActive] = useState(0);

  useEffect(() => {
    const obs = new IntersectionObserver(entries => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          const idx = SECTION_IDS.indexOf(e.target.id);
          if (idx >= 0) setActive(idx);
        }
      });
    }, { threshold: 0.3 });
    SECTION_IDS.forEach(id => {
      const el = document.getElementById(id);
      if (el) obs.observe(el);
    });
    return () => obs.disconnect();
  }, []);

  return (
    <div style={{ maxWidth:720, margin:"0 auto", fontFamily:"var(--font-sans)" }}>
      <h2 className="sr-only">Probabilistic Routing for Cross-Border Payments — CS109 presentation</h2>
      <Nav active={active} />
      <Hero />
      <ProblemSection />
      <MLESection />
      <OUSection />
      <FrontierSection />
      <CLTSection />
      <ResultsSection />
      <div style={{ textAlign:"center", padding:"24px 0", fontSize:11,
        color:"var(--color-text-tertiary)" }}>
        Marco Paes · CS109: Probability for Computer Scientists · Stanford University · June 2026
      </div>
    </div>
  );
}
