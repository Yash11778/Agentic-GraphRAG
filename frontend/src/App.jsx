import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, LabelList, ReferenceLine,
  Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";

// Three series, three fixed slots from the validated categorical palette
// (blue / orange / aqua). Colour follows the pipeline, never its rank, so a
// pipeline keeps its hue in every chart and every table on the page.
const PIPELINES = [
  {
    key: "rag", label: "RAG", colour: "#2a78d6",
    blurb: "Vector search over 19,832 chunks, then generate. One retrieval step.",
  },
  {
    key: "graphrag", label: "GraphRAG", colour: "#eb6834",
    blurb: "Link the question's entities, traverse a fixed path, generate.",
  },
  {
    key: "agentic", label: "Agentic GraphRAG", colour: "#1baf7a",
    blurb: "Plan, choose a tool, evaluate the evidence, decide whether to continue.",
  },
];
// Pipeline display names, derived from PIPELINES so a rename happens once.
const PLABEL = Object.fromEntries(PIPELINES.map((p) => [p.key, p.label]));
const QTYPES = ["lookup", "temporal", "multi_hop", "aggregation", "superlative"];
const QLABEL = {
  lookup: "Lookup", temporal: "Temporal", multi_hop: "Multi-hop",
  aggregation: "Aggregation", superlative: "Superlative",
};
// Short forms for an axis that has five categories and a phone's width.
const QSHORT = {
  lookup: "Look", temporal: "Temp", multi_hop: "Multi",
  aggregation: "Aggr", superlative: "Super",
};
const ROUTE = {
  agentic: "full investigation", fast_path: "fast path",
  fast_path_escalated: "fast path, escalated",
};
const SECTIONS = [
  ["results", "Results"],
  ["cost", "Cost of agency"],
  ["hidden", "Held-out set"],
  ["explorer", "Question explorer"],
];

// Backend names as the result rows record them, in the words the page uses.
const BACKEND = { tigergraph: "TigerGraph", local: "the local backend" };
const backendName = (b) => BACKEND[b] ?? b ?? "an unrecorded backend";

const pct = (x) => `${Math.round((x ?? 0) * 100)}%`;
const num = (x) => Math.round(x ?? 0).toLocaleString();
const signed = (x) => `${x >= 0 ? "+" : "−"}${num(Math.abs(x))}`;
const signedPct = (x) => `${x >= 0 ? "+" : "−"}${Math.abs(Math.round(x))}%`;
// Median wall clock net of rate-limit sleeps, when the run recorded it; older
// result files only carry the mean, which a throttled run inflates.
const secs = (s) => {
  const v = s.median_active_s ?? s.median_wall_clock_s ?? s.avg_wall_clock_s;
  return v == null ? "—" : `${v.toFixed(1)}s`;
};

// A chart sized from its figure's measured width. The library's responsive
// wrapper waits for a resize notification, and until it arrives the chart is
// drawn at whatever width the page had before layout settled; measuring the
// figure directly after layout draws it right the first time, on any screen.
function Sized({ height, children }) {
  const ref = useRef(null);
  const [width, setWidth] = useState(0);
  useLayoutEffect(() => {
    const measure = () => { if (ref.current) setWidth(ref.current.clientWidth); };
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, []);
  return (
    <div ref={ref} style={{ width: "100%", height }}>
      {width > 0 && children(width)}
    </div>
  );
}

function useNarrow(width = 560) {
  const [narrow, setNarrow] = useState(
    () => typeof window !== "undefined" && window.innerWidth < width);
  useEffect(() => {
    const onResize = () => setNarrow(window.innerWidth < width);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [width]);
  return narrow;
}

export default function App() {
  const [report, setReport] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("report.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.statusText))))
      .then(setReport)
      .catch((e) => setError(String(e)));
  }, []);

  return (
    <>
      <TopBar />
      <main>
        <Hero report={report} />
        {error && (
          <p className="alert">
            report.json could not be loaded ({error}). Generate it with{" "}
            <code>python eval2/report.py</code>.
          </p>
        )}
        {report && (
          <>
            <Results report={report} />
            <Cost report={report} />
            <Hidden report={report} />
            <Explorer report={report} />
          </>
        )}
      </main>
      <SiteFooter report={report} />
    </>
  );
}

function TopBar() {
  return (
    <header className="topbar">
      <div className="bar">
        <div className="brand">
          <span className="mark" aria-hidden="true" />
          <div>
            <strong>Agentic GraphRAG</strong>
            <span>TigerGraph Hackathon &middot; Round 1</span>
          </div>
        </div>
        <nav>
          {SECTIONS.map(([id, label]) => (
            <a key={id} href={`#${id}`}>{label}</a>
          ))}
        </nav>
      </div>
    </header>
  );
}

function Hero({ report }) {
  const s = report?.summary;
  return (
    <section className="hero">
      <div className="hero-text">
        <h1>When does an agent earn its tokens?</h1>
        <p>
          One corpus of 2,951 Wikipedia articles and 100 questions, answered three
          ways. Same model, same temperature, same output budget for all three, so the
          only thing that differs between them is control flow.
        </p>
      </div>
      {s && (
        <dl className="hero-stats">
          <div>
            <dt>Best exact match</dt>
            <dd>{pct(s.agentic.exact)}</dd>
            <span>Agentic GraphRAG</span>
          </div>
          <div>
            <dt>Tokens per question</dt>
            <dd>{num(s.agentic.avg_total_tokens)}</dd>
            <span>fewer than either baseline</span>
          </div>
          <div>
            <dt>Held-out questions</dt>
            <dd>{report.hidden.answered}/{report.hidden.n}</dd>
            <span>
              answered, the rest declined
              {report.hidden.routes?.fast_path != null &&
                ` · ${report.hidden.routes.fast_path} took the fast path`}
            </span>
          </div>
        </dl>
      )}
    </section>
  );
}

/* ── Results ─────────────────────────────────────────────────────────────── */

function Results({ report }) {
  const { summary } = report;
  const narrow = useNarrow();
  const accuracy = QTYPES.map((q) => {
    const row = { qtype: narrow ? QSHORT[q] : QLABEL[q] };
    PIPELINES.forEach((p) => {
      row[p.key] = Math.round((summary[p.key]?.by_qtype?.[q]?.exact ?? 0) * 100);
    });
    return row;
  });
  const tokens = PIPELINES.map((p) => ({
    name: p.label,
    context: Math.round(summary[p.key]?.avg_context_tokens ?? 0),
    rest: Math.max(Math.round((summary[p.key]?.avg_total_tokens ?? 0)
      - (summary[p.key]?.avg_context_tokens ?? 0)), 0),
  }));
  const bestExact = Math.max(...PIPELINES.map((p) => summary[p.key].exact));

  return (
    <section id="results" className="band">
      <SectionHead
        eyebrow="Results"
        title="Three pipelines, one hundred questions"
        lede="Exact match against the gold answer is the headline number. Grounding
              compares the documents each pipeline cited against the gold document ids
              the organisers provide."
      />

      <div className="cards">
        {PIPELINES.map((p) => {
          const s = summary[p.key];
          return (
            <article key={p.key} className={`card${s.exact === bestExact ? " lead" : ""}`}>
              <header style={{ borderTopColor: p.colour }}>
                <h3>{p.label}</h3>
                <p>{p.blurb}</p>
              </header>
              <p className="score">{pct(s.exact)}<span>exact match</span></p>
              <dl>
                <div><dt>Tokens per question</dt><dd>{num(s.avg_total_tokens)}</dd></div>
                <div><dt>Of that, retrieved context</dt><dd>{num(s.avg_context_tokens)}</dd></div>
                <div><dt>Grounding F1</dt><dd>{s.avg_grounding_f1.toFixed(3)}</dd></div>
                <div><dt>Steps taken</dt><dd>{s.avg_steps.toFixed(1)}</dd></div>
                <div><dt>Seconds per question (median)</dt><dd>{secs(s)}</dd></div>
                <div><dt>Declined to answer</dt><dd>{pct(s.insufficient)}</dd></div>
              </dl>
            </article>
          );
        })}
      </div>

      <div className="split">
        <Figure
          title="Exact match by question type"
          caption="Aggregation and superlative questions need a complete set rather than
                   the most similar few documents. One spans 43 gold documents, which no
                   top-k window reaches."
        >
          <Sized height={300}>{(width) => (
            <BarChart width={width} height={300} data={accuracy}
                      margin={{ top: 20, right: 4, bottom: 4, left: -14 }}
                      barCategoryGap="24%" barGap={2}>
              <CartesianGrid stroke="#e7e5e0" vertical={false} />
              <XAxis dataKey="qtype" tickLine={false} axisLine={{ stroke: "#d8d5ce" }}
                     tick={{ fill: "#52514e", fontSize: 12 }} interval={0} />
              <YAxis domain={[0, 100]} unit="%" tickLine={false} axisLine={false}
                     tick={{ fill: "#8a8880", fontSize: 11 }} width={54} />
              <Tooltip cursor={{ fill: "#f2f0ec" }} content={<Hint suffix="%" />} />
              {PIPELINES.map((p) => (
                <Bar key={p.key} dataKey={p.key} name={p.label} fill={p.colour}
                     radius={[3, 3, 0, 0]} maxBarSize={34} isAnimationActive={false}
                     minPointSize={2}>
                  {/* Zeros are labelled too: a bar worth nothing and a series that
                      never ran look identical otherwise, and two question types
                      really are zero for both baselines. */}
                  <LabelList dataKey={p.key} content={<BarValue />} />
                </Bar>
              ))}
            </BarChart>
          )}</Sized>
          <Key items={PIPELINES.map((p) => [p.colour, p.label])} />
        </Figure>

        <Figure
          title="Where the tokens go"
          caption="Retrieved context is what the baselines spend their budget on. The
                   agent counts and compares inside the database, so documents never
                   enter the prompt."
        >
          <Sized height={300}>{(width) => (
            <BarChart width={width} height={300} data={tokens} layout="vertical"
                      margin={{ top: 8, right: 62, bottom: 4, left: 4 }}
                      barCategoryGap="30%">
              <CartesianGrid stroke="#e7e5e0" horizontal={false} />
              <XAxis type="number" tickLine={false} axisLine={{ stroke: "#d8d5ce" }}
                     tick={{ fill: "#8a8880", fontSize: 11 }} />
              <YAxis type="category" dataKey="name" width={140} tickLine={false}
                     axisLine={false} tick={{ fill: "#52514e", fontSize: 12 }} />
              <Tooltip cursor={{ fill: "#f2f0ec" }} content={<Hint suffix=" tokens" />} />
              <Bar dataKey="context" name="Retrieved context" stackId="t" fill="#eb6834"
                   maxBarSize={30} isAnimationActive={false} />
              <Bar dataKey="rest" name="Prompt, reasoning, answer" stackId="t"
                   fill="#2a78d6" maxBarSize={30} radius={[0, 3, 3, 0]}
                   isAnimationActive={false}>
                <LabelList position="right" offset={10} fill="#52514e" fontSize={11}
                           valueAccessor={(e) => num(e.payload.context + e.payload.rest)} />
              </Bar>
            </BarChart>
          )}</Sized>
          <Key items={[["#eb6834", "Retrieved context"],
                       ["#2a78d6", "Prompt, reasoning and answer"]]} />
        </Figure>
      </div>

      <Table
        caption="Exact match and token cost, by question type"
        columns={["Question type", "Questions", "RAG", "GraphRAG", "Agentic",
                  "Tokens RAG", "Tokens GraphRAG", "Tokens agentic"]}
        align={[0, 1, 1, 1, 1, 1, 1, 1]}
        rows={QTYPES.map((q) => {
          const [r, g, a] = PIPELINES.map((p) => summary[p.key].by_qtype[q]);
          const best = Math.max(r.exact, g.exact, a.exact);
          const mark = (s) => (s.exact === best ? <strong>{pct(s.exact)}</strong> : pct(s.exact));
          return [QLABEL[q], r.n, mark(r), mark(g), mark(a),
                  num(r.avg_total_tokens), num(g.avg_total_tokens), num(a.avg_total_tokens)];
        })}
      />
    </section>
  );
}

/* ── Cost of agency ──────────────────────────────────────────────────────── */

function Cost({ report }) {
  const [baseline, setBaseline] = useState("vs_graphrag");
  const source = report.cost_of_agency[baseline];
  const data = QTYPES.filter((q) => source[q]).map((q) => ({
    qtype: QLABEL[q],
    tokens: source[q].delta_tokens,
    accuracy: source[q].delta_exact * 100,
    verdict: source[q].verdict,
  }));

  return (
    <section id="cost" className="band alt">
      <SectionHead
        eyebrow="Cost of agency"
        title="Where the extra steps pay for themselves"
        lede="Each point is one question type: the accuracy the agent adds, against the
              tokens it spends to add it. Left of the vertical rule it is also cheaper
              than the baseline."
      />

      <div className="toggle">
        <span>Compared against</span>
        {[["vs_graphrag", "GraphRAG"], ["vs_rag", "RAG"]].map(([key, label]) => (
          <button key={key} className={baseline === key ? "on" : ""}
                  onClick={() => setBaseline(key)}>{label}</button>
        ))}
      </div>

      <div className="split">
        <Figure title="Accuracy gained against tokens spent"
                caption="Upper left is best: more accurate and cheaper. Upper right still
                         pays. Below the horizontal rule the agent is not earning its place.">
          <Sized height={340}>{(width) => (
            <ScatterChart width={width} height={340}
                          margin={{ top: 16, right: 28, bottom: 28, left: 4 }}>
              <CartesianGrid stroke="#eeece7" />
              <XAxis type="number" dataKey="tokens" tickLine={false}
                     axisLine={{ stroke: "#d8d5ce" }} tick={{ fill: "#8a8880", fontSize: 11 }}
                     label={{ value: "extra tokens per question", position: "insideBottom",
                              offset: -14, fill: "#52514e", fontSize: 12 }} />
              <YAxis type="number" dataKey="accuracy" unit="%" tickLine={false}
                     axisLine={false} tick={{ fill: "#8a8880", fontSize: 11 }} width={54}
                     label={{ value: "accuracy gained", angle: -90, position: "insideLeft",
                              offset: 16, fill: "#52514e", fontSize: 12 }} />
              <ZAxis range={[150, 150]} />
              <ReferenceLine x={0} stroke="#b8b4ab" />
              <ReferenceLine y={0} stroke="#b8b4ab" />
              <Tooltip cursor={false} content={<PointHint />} />
              <Scatter data={data} isAnimationActive={false}>
                {data.map((d) => (
                  <Cell key={d.qtype} fill={d.accuracy > 2 ? "#1baf7a" : "#eb6834"}
                        stroke="#fff" strokeWidth={2} />
                ))}
                <LabelList dataKey="qtype" content={<PointLabel data={data} />} />
              </Scatter>
            </ScatterChart>
          )}</Sized>
        </Figure>

        <div className="finding">
          <Table
            caption={`Agentic against ${baseline === "vs_rag" ? "RAG" : "GraphRAG"}`}
            columns={["Question type", "Accuracy", "Tokens", "Reading"]}
            align={[0, 1, 1, 0]}
            rows={data.map((d) => [
              d.qtype,
              <span className={d.accuracy >= 0 ? "up" : "down"}>{signedPct(d.accuracy)}</span>,
              <span className={d.tokens <= 0 ? "up" : "down"}>{signed(d.tokens)}</span>,
              d.verdict,
            ])}
          />
          <p>
            Agentic retrieval pays for itself when the answer requires a complete set,
            and is overkill when one document answers the question. The system acts on
            that while it runs: a question judged answerable from a single document takes
            a short path on a tighter budget, and escalates only if that path comes back
            empty.
          </p>
        </div>
      </div>
    </section>
  );
}

/* ── Held-out set ────────────────────────────────────────────────────────── */

function Hidden({ report }) {
  const h = report.hidden;
  if (!h || !h.n) return null;
  const routes = Object.entries(h.routes || {});
  const stops = Object.entries(h.stop_reasons || {});
  const byType = Object.entries(h.by_qtype || {});
  const byPipeline = Object.entries(h.by_pipeline || {});
  return (
    <section id="hidden" className="band">
      <SectionHead
        eyebrow="Held-out set"
        title={`${h.answered} of ${h.n} held-out questions answered`}
        lede={`The organisers' fifty questions without published answers, run once on
              the final build against ${backendName(h.backend)} through all three
              pipelines, as every row of the submission bundle records. Nothing here
              is scored, so the comparison below is cost and coverage rather than
              accuracy: both baselines decline eight of the fifty, the agent none,
              and it does it on a fraction of the context. Every trace is kept in full.`}
      />
      <div className="hidden-grid">
        <dl className="hero-stats">
          <div>
            <dt>Answered</dt>
            <dd>{h.answered}/{h.n}</dd>
            <span>the rest declined, never guessed</span>
          </div>
          <div>
            <dt>Tokens per question</dt>
            <dd>{num(h.total_tokens / h.n)}</dd>
            <span>{num(h.total_tokens)} in total</span>
          </div>
          {h.avg_steps != null && (
            <div>
              <dt>Steps per question</dt>
              <dd>{h.avg_steps.toFixed(1)}</dd>
              <span>planning step included</span>
            </div>
          )}
          {h.median_active_s != null && (
            <div>
              <dt>Seconds per question</dt>
              <dd>{h.median_active_s.toFixed(1)}s</dd>
              <span>median, net of the rate limit</span>
            </div>
          )}
        </dl>
        <div className="hidden-tables">
          {byPipeline.length > 0 && (
            <Table
              caption="All three pipelines, same 50 questions"
              columns={["Pipeline", "Answered", "Tokens", "Context", "Median s"]}
              align={[0, 1, 1, 1, 1]}
              rows={byPipeline.map(([name, v]) => [
                PLABEL[name] ?? name,
                `${v.answered}/${v.n}`,
                num(v.avg_tokens),
                num(v.avg_context_tokens),
                v.median_active_s.toFixed(1),
              ])}
            />
          )}
          {byType.length > 0 && (
            <Table
              caption="By question type"
              columns={["Question type", "Questions", "Answered", "Declined"]}
              align={[0, 1, 1, 1]}
              rows={byType.map(([q, v]) => [QLABEL[q] ?? q, v.n, v.answered, v.n - v.answered])}
            />
          )}
          <Table
            caption="Route taken"
            columns={["Route", "Questions"]}
            align={[0, 1]}
            rows={routes.map(([r, n]) => [ROUTE[r] ?? r.replace(/_/g, " "), n])}
          />
          <Table
            caption="Why the agent stopped"
            columns={["Stop reason", "Questions"]}
            align={[0, 1]}
            rows={stops.map(([r, n]) => [r.replace(/_/g, " "), n])}
          />
        </div>
      </div>
    </section>
  );
}

/* ── Question explorer ───────────────────────────────────────────────────── */

function Explorer({ report }) {
  const questions = report.questions;
  const [qid, setQid] = useState(
    questions.find((q) => q.qtype === "aggregation")?.qid ?? questions[0].qid);
  const question = useMemo(
    () => questions.find((q) => q.qid === qid) ?? questions[0], [qid, questions]);
  const agent = question.pipelines.agentic;

  return (
    <section id="explorer" className="band">
      <SectionHead
        eyebrow="Question explorer"
        title="Follow one investigation, step by step"
        lede="Every step as it was recorded while the agent ran: the tool it chose, what
              that cost, and why it stopped. Nothing here is reconstructed afterwards."
      />

      <label className="field">
        <span>Choose a question &mdash; {questions.length} available</span>
        <select value={qid} onChange={(e) => setQid(e.target.value)}>
          {questions.map((q) => (
            <option key={q.qid} value={q.qid}>
              {q.qid} &middot; {QLABEL[q.qtype]} &middot; {q.question.slice(0, 120)}
            </option>
          ))}
        </select>
      </label>

      <div className="asked">
        <p>{question.question}</p>
        <p className="gold">Gold answer: <strong>{question.gold}</strong></p>
      </div>

      <div className="cards answers">
        {PIPELINES.filter((p) => question.pipelines[p.key]).map((p) => {
          const r = question.pipelines[p.key];
          return (
            <article key={p.key} className={`card ${r.exact ? "right" : "wrong"}`}>
              <header style={{ borderTopColor: p.colour }}>
                <h3>{p.label}</h3>
                <p>{r.exact ? "Matched the gold answer" : "Did not match"}</p>
              </header>
              <p className="said">{r.answer || "—"}</p>
              <dl>
                <div><dt>Tokens</dt><dd>{num(r.total_tokens)}</dd></div>
                <div><dt>Documents cited</dt><dd>{r.citations.length}</dd></div>
                <div><dt>Steps</dt><dd>{r.trace.length || 1}</dd></div>
              </dl>
            </article>
          );
        })}
      </div>

      {agent && (
        <div className="investigation">
          <h3>How the agent got there</h3>
          <p className="run">
            Route {agent.route.replace(/_/g, " ")} &middot; stopped because{" "}
            {agent.stop_reason.replace(/_/g, " ")} &middot; {agent.citations.length}
            {" "}documents cited
          </p>
          <ol className="steps">
            {agent.trace.map((s, i) => (
              <li key={i}>
                <span className="no">{i + 1}</span>
                <div>
                  <div className="row">
                    <strong>{s.tool.replace(/_/g, " ")}</strong>
                    <em>{s.agent.replace(/_/g, " ")}</em>
                    <span className={`state ${s.status}`}>{s.status}</span>
                    {s.strategy_change && <span className="pivot">changed strategy</span>}
                  </div>
                  {s.args_digest && <p className="args">{s.args_digest}</p>}
                  <p className="meta">
                    {Math.round(s.latency_ms)} ms
                    {s.context_tokens > 0 && ` · ${num(s.context_tokens)} context tokens`}
                    {s.new_evidence_count > 0 && ` · +${s.new_evidence_count} documents`}
                    {s.note && ` · ${s.note}`}
                  </p>
                </div>
              </li>
            ))}
          </ol>
        </div>
      )}
    </section>
  );
}

/* ── Shared pieces ───────────────────────────────────────────────────────── */

function SectionHead({ eyebrow, title, lede }) {
  return (
    <div className="head">
      <p className="eyebrow">{eyebrow}</p>
      <h2>{title}</h2>
      <p className="lede">{lede}</p>
    </div>
  );
}

function Figure({ title, caption, children }) {
  return (
    <figure>
      <figcaption>
        <strong>{title}</strong>
        <span>{caption}</span>
      </figcaption>
      {children}
    </figure>
  );
}

function Key({ items }) {
  return (
    <ul className="key">
      {items.map(([colour, label]) => (
        <li key={label}><i style={{ background: colour }} />{label}</li>
      ))}
    </ul>
  );
}

function Table({ caption, columns, rows, align }) {
  return (
    <div className="table-wrap">
      <table>
        <caption>{caption}</caption>
        <thead>
          <tr>{columns.map((c, i) => (
            <th key={c} className={align?.[i] ? "right" : ""}>{c}</th>
          ))}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>{r.map((cell, j) => (
              <td key={j} className={align?.[j] ? "right num" : ""}>{cell}</td>
            ))}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function BarValue({ x, y, width, value }) {
  if (x == null || value == null) return null;
  return (
    <text x={x + width / 2} y={y - 6} textAnchor="middle"
          fill={value === 0 ? "#8a8880" : "#52514e"} fontSize={11}
          fontFamily="ui-sans-serif, system-ui, sans-serif">
      {value}
    </text>
  );
}

// Two question types can land almost on top of each other (multi-hop and temporal
// differ by four tokens and two points of accuracy), so a near neighbour's label
// is pushed down and sits on whichever side has room.
function PointLabel({ data, x, y, index }) {
  const point = data?.[index];
  if (!point || x == null) return null;
  const crowded = data.some((other, i) =>
    i < index && Math.abs(other.tokens - point.tokens) < 260
      && Math.abs(other.accuracy - point.accuracy) < 12);
  const left = point.tokens > 0;
  return (
    <text x={left ? x - 12 : x + 12} y={y + (crowded ? 26 : 4)}
          textAnchor={left ? "end" : "start"} fill="#52514e" fontSize={11}
          fontFamily="ui-sans-serif, system-ui, sans-serif">
      {point.qtype}
    </text>
  );
}

function Hint({ active, payload, label, suffix }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="hint">
      <p className="hint-head">{label}</p>
      {payload.map((row) => (
        <p key={row.name}>
          <i style={{ background: row.color }} />
          {row.name}<span>{num(row.value)}{suffix}</span>
        </p>
      ))}
    </div>
  );
}

function PointHint({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="hint">
      <p className="hint-head">{d.qtype}</p>
      <p>accuracy<span>{signedPct(d.accuracy)}</span></p>
      <p>tokens<span>{signed(d.tokens)}</span></p>
      <p>reading<span>{d.verdict}</span></p>
    </div>
  );
}

function SiteFooter({ report }) {
  const prov = report?.provenance;
  return (
    <footer className="site-footer">
      <div className="bar">
        <p>
          Figures are read from the committed result files: 300 public runs and 50
          held-out questions. Latency is the median per question, excluding time
          spent waiting on the provider&apos;s rate limit. Nothing on this page is
          computed live.
          {prov && ` The public runs read from ${backendName(prov.public.backend)} so
          that all three pipelines share one source; the held-out set ran against
          ${backendName(prov.hidden.backend)}. Both labels come from the rows.`}
        </p>
        <p className="stack">
          TigerGraph &middot; GSQL &middot; Groq gpt-oss-120b &middot; local ONNX embeddings
        </p>
      </div>
    </footer>
  );
}
