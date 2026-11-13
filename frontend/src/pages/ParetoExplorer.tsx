import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useSelectedSolution, useStore } from '../store';
import { pickLabels, representativePicks, type PickKind } from '../mock/solutions';
import type { Objectives, Solution } from '../types';
import { ObjectiveBars } from './Dashboard';
import DecisionPanel from '../components/DecisionPanel';

interface AxisMeta {
  short: string;
  name: string;
  unit: string;
  desc: string;
  fmt: (v: number) => string;
}

const AXES: Record<keyof Objectives, AxisMeta> = {
  f1: {
    short: 'f1 · Idle Gaps',
    name: 'Student & Faculty Internal Idle Gaps',
    unit: 'free periods',
    desc: 'Total internal free teaching periods for students + 0.5 × faculty free periods (mandatory lunch excluded). Lower is better.',
    fmt: (v) => v.toFixed(1),
  },
  f2: {
    short: 'f2 · Room Wastage',
    name: 'Room Seat Wastage Fraction',
    unit: '% empty seats',
    desc: 'Duration-weighted fraction of unused seats while classrooms are occupied. Lower is better (higher seat fit).',
    fmt: (v) => `${(v * 100).toFixed(1)}%`,
  },
  f3: {
    short: 'f3 · Workload Imbalance',
    name: 'Faculty Teaching Load Imbalance',
    unit: 'std dev',
    desc: 'Population standard deviation of (weekly teaching load / weekly allowance). Lower is better (fairer workload).',
    fmt: (v) => v.toFixed(3),
  },
};

export interface AlgoSpec {
  id: string;
  name: string;
  color: string;
  shape: 'circle' | 'triangle' | 'diamond' | 'square' | 'rhombus';
  desc: string;
}

const ALGORITHMS: AlgoSpec[] = [
  { id: 'memetic', name: 'Memetic NSGA-II', color: '#2563eb', shape: 'circle', desc: 'Feasible mutation-only genetic algorithm with faculty branch-and-bound' },
  { id: 'moead', name: 'MOEA/D (Tchebycheff)', color: '#059669', shape: 'triangle', desc: 'Decomposition into weighted subproblems with neighborhood information sharing' },
  { id: 'amosa', name: 'AMOSA (Sim. Annealing)', color: '#7c3aed', shape: 'diamond', desc: 'Archived simulated annealing with Pareto dominance energy and cooling' },
  { id: 'nsga2', name: 'Standard NSGA-II', color: '#ea580c', shape: 'square', desc: 'Classical genetic algorithm with categorical crossover (Deb et al., 2002)' },
  { id: 'greedy', name: 'Repeated Greedy', color: '#d97706', shape: 'rhombus', desc: 'Multi-start DSatur constructive heuristic baseline' },
];

const pickColor: Record<PickKind, string> = {
  gaps: '#c8553d',
  rooms: '#d9a441',
  balance: '#7a8b6f',
  knee: '#5b6fb5',
};

function ScatterPlot({
  xk,
  yk,
  all,
  picks,
  selectedId,
  showAll,
  onPick,
  enabledAlgos,
  fronts,
}: {
  xk: keyof Objectives;
  yk: keyof Objectives;
  all: Solution[];
  picks: Record<PickKind, Solution>;
  selectedId: string;
  showAll: boolean;
  onPick: (id: string) => void;
  enabledAlgos: Record<string, boolean>;
  fronts: {
    memetic: Objectives[];
    moead: Objectives[];
    amosa: Objectives[];
    nsga2: Objectives[];
    greedy: Objectives[];
  };
}) {
  const W = 380;
  const H = 300;
  const m = { l: 56, r: 18, t: 24, b: 54 };

  const pts = showAll ? all : all.filter((s) => s.onFront);

  // Collect all visible coordinates to compute robust bounding box
  const allX: number[] = pts.map((s) => s.objectives[xk]);
  const allY: number[] = pts.map((s) => s.objectives[yk]);

  if (enabledAlgos.moead) fronts.moead.forEach((o) => { allX.push(o[xk]); allY.push(o[yk]); });
  if (enabledAlgos.amosa) fronts.amosa.forEach((o) => { allX.push(o[xk]); allY.push(o[yk]); });
  if (enabledAlgos.nsga2) fronts.nsga2.forEach((o) => { allX.push(o[xk]); allY.push(o[yk]); });
  if (enabledAlgos.greedy) fronts.greedy.forEach((o) => { allX.push(o[xk]); allY.push(o[yk]); });

  const pad = (lo: number, hi: number) => {
    const d = hi - lo || Math.abs(hi) || 1;
    return [Math.max(0, lo - d * 0.1), hi + d * 0.1];
  };

  const [x0, x1] = pad(allX.length ? Math.min(...allX) : 0, allX.length ? Math.max(...allX) : 1);
  const [y0, y1] = pad(allY.length ? Math.min(...allY) : 0, allY.length ? Math.max(...allY) : 1);

  const sx = (v: number) => m.l + ((v - x0) / Math.max(x1 - x0, 1e-6)) * (W - m.l - m.r);
  const sy = (v: number) => H - m.b - ((v - y0) / Math.max(y1 - y0, 1e-6)) * (H - m.t - m.b);

  const pickOf = (s: Solution) => (Object.keys(picks) as PickKind[]).find((k) => picks[k].id === s.id);

  return (
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      <svg className="scatter" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${AXES[xk].short} vs ${AXES[yk].short}`}>
        {/* Background Canvas */}
        <rect x={m.l} y={m.t} width={W - m.l - m.r} height={H - m.t - m.b} fill="#fbf8f2" rx="6" />

        {/* Grid lines */}
        {[0.25, 0.5, 0.75].map((t) => (
          <g key={t} stroke="#e9e0d2" strokeDasharray="3 3">
            <line x1={m.l} x2={W - m.r} y1={m.t + t * (H - m.t - m.b)} y2={m.t + t * (H - m.t - m.b)} />
            <line y1={m.t} y2={H - m.b} x1={m.l + t * (W - m.l - m.r)} x2={m.l + t * (W - m.l - m.r)} />
          </g>
        ))}

        {/* Ticks and Axis Values */}
        <text x={m.l} y={H - m.b + 15} fontSize="10" fill="#8a8173">{AXES[xk].fmt(x0)}</text>
        <text x={W - m.r} y={H - m.b + 15} fontSize="10" fill="#8a8173" textAnchor="end">{AXES[xk].fmt(x1)}</text>
        <text x={m.l - 6} y={H - m.b} fontSize="10" fill="#8a8173" textAnchor="end">{AXES[yk].fmt(y0)}</text>
        <text x={m.l - 6} y={m.t + 8} fontSize="10" fill="#8a8173" textAnchor="end">{AXES[yk].fmt(y1)}</text>

        {/* Horizontal (X-Axis) Title */}
        <text x={(m.l + W - m.r) / 2} y={H - 12} fontSize="11" fill="#3f3b35" textAnchor="middle" fontWeight="700">
          X: {AXES[xk].short} ({AXES[xk].unit}) ← Better
        </text>

        {/* Vertical (Y-Axis) Title */}
        <text transform={`translate(14 ${(m.t + H - m.b) / 2}) rotate(-90)`} fontSize="11" fill="#3f3b35" textAnchor="middle" fontWeight="700">
          Y: {AXES[yk].short} ({AXES[yk].unit}) ← Better
        </text>

        {/* 1. Greedy Points (Amber Rhombus) */}
        {enabledAlgos.greedy && fronts.greedy.map((o, i) => (
          <rect
            key={`grd-${i}`}
            x={sx(o[xk]) - 4.5}
            y={sy(o[yk]) - 4.5}
            width="9"
            height="9"
            fill="#d97706"
            fillOpacity="0.85"
            stroke="#92400e"
            strokeWidth="1.2"
            transform={`rotate(45 ${sx(o[xk])} ${sy(o[yk])})`}
          >
            <title>Repeated Greedy: f1={o.f1.toFixed(1)}, f2={(o.f2 * 100).toFixed(1)}%, f3={o.f3.toFixed(3)}</title>
          </rect>
        ))}

        {/* 2. Standard NSGA-II Points (Orange Square) */}
        {enabledAlgos.nsga2 && fronts.nsga2.map((o, i) => (
          <rect
            key={`nsga2-${i}`}
            x={sx(o[xk]) - 4}
            y={sy(o[yk]) - 4}
            width="8"
            height="8"
            fill="#ea580c"
            fillOpacity="0.85"
            stroke="#c2410c"
            strokeWidth="1.2"
          >
            <title>Standard NSGA-II: f1={o.f1.toFixed(1)}, f2={(o.f2 * 100).toFixed(1)}%, f3={o.f3.toFixed(3)}</title>
          </rect>
        ))}

        {/* 3. AMOSA Points (Purple Diamond) */}
        {enabledAlgos.amosa && fronts.amosa.map((o, i) => {
          const cx = sx(o[xk]);
          const cy = sy(o[yk]);
          const sz = 6;
          return (
            <polygon
              key={`amosa-${i}`}
              points={`${cx},${cy - sz} ${cx + sz},${cy} ${cx},${cy + sz} ${cx - sz},${cy}`}
              fill="#7c3aed"
              fillOpacity="0.85"
              stroke="#5b21b6"
              strokeWidth="1.2"
            >
              <title>AMOSA: f1={o.f1.toFixed(1)}, f2={(o.f2 * 100).toFixed(1)}%, f3={o.f3.toFixed(3)}</title>
            </polygon>
          );
        })}

        {/* 4. MOEA/D Points (Emerald Green Triangle) */}
        {enabledAlgos.moead && fronts.moead.map((o, i) => {
          const cx = sx(o[xk]);
          const cy = sy(o[yk]);
          const sz = 6.5;
          return (
            <polygon
              key={`moead-${i}`}
              points={`${cx},${cy - sz} ${cx + sz},${cy + sz * 0.8} ${cx - sz},${cy + sz * 0.8}`}
              fill="#059669"
              fillOpacity="0.85"
              stroke="#047857"
              strokeWidth="1.2"
            >
              <title>MOEA/D (Tchebycheff): f1={o.f1.toFixed(1)}, f2={(o.f2 * 100).toFixed(1)}%, f3={o.f3.toFixed(3)}</title>
            </polygon>
          );
        })}

        {/* 5. Memetic NSGA-II Points (Royal Blue Circles / Pick Badges) */}
        {enabledAlgos.memetic && pts.map((s) => {
          const pk = pickOf(s);
          const sel = s.id === selectedId;
          const fill = pk ? pickColor[pk] : s.onFront ? '#2563eb' : '#94a3b8';
          const r = pk ? 7 : s.onFront ? 5 : 3.5;
          return (
            <g key={s.id} onClick={() => onPick(s.id)} style={{ cursor: 'pointer' }}>
              {sel && (
                <circle cx={sx(s.objectives[xk])} cy={sy(s.objectives[yk])} r={r + 4} fill="none" stroke="#1d4ed8" strokeWidth="2.5" />
              )}
              <circle
                cx={sx(s.objectives[xk])}
                cy={sy(s.objectives[yk])}
                r={r}
                fill={fill}
                stroke="#fff"
                strokeWidth="1"
                opacity={s.onFront ? 1 : 0.65}
              >
                <title>
                  Memetic NSGA-II ({s.id}) — {pk ? pickLabels[pk].title : s.onFront ? 'Pareto optimal' : 'Dominated'}:
                  f1={s.objectives.f1.toFixed(1)}, f2={(s.objectives.f2 * 100).toFixed(1)}%, f3={s.objectives.f3.toFixed(3)}
                </title>
              </circle>
            </g>
          );
        })}
      </svg>

      {/* Explicit Axis Explanation Banner */}
      <div style={{ padding: '8px 12px', background: '#f5efe4', borderRadius: 6, fontSize: 11, marginTop: 4, lineHeight: 1.4 }}>
        <div><b>X-Axis:</b> {AXES[xk].name} · {AXES[xk].desc}</div>
        <div style={{ marginTop: 2 }}><b>Y-Axis:</b> {AXES[yk].name} · {AXES[yk].desc}</div>
      </div>
    </div>
  );
}

export default function ParetoExplorer() {
  const { solutions, selectedId, select, stats } = useStore();
  const selected = useSelectedSolution();
  const nav = useNavigate();
  const [showAll, setShowAll] = useState(true);

  // Toggle state for all 5 algorithms
  const [enabledAlgos, setEnabledAlgos] = useState<Record<string, boolean>>({
    memetic: true,
    moead: true,
    amosa: true,
    nsga2: true,
    greedy: true,
  });

  const toggleAlgo = (id: string) => {
    setEnabledAlgos((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const front = useMemo(() => solutions.filter((s) => s.onFront).sort((a, b) => a.objectives.f1 - b.objectives.f1), [solutions]);
  const picks = useMemo(() => representativePicks(front), [front]);

  // Fronts measured in the last run. An algorithm that was not run has no points; nothing is estimated.
  const fronts = useMemo(
    () => ({
      memetic: front.map((s) => s.objectives),
      moead: stats?.moeadFront ?? [],
      amosa: stats?.amosaFront ?? [],
      nsga2: stats?.nsga2Front ?? [],
      greedy: stats?.baselineFront ?? [],
    }),
    [front, stats],
  );

  // One row per algorithm. `status` says whether it was run in the last optimiser run and what happened.
  const comparisonRows = useMemo(() => {
    const names: Record<string, string> = {
      memetic: 'Memetic NSGA-II (final method)',
      moead: 'MOEA/D (Tchebycheff decomposition)',
      amosa: 'AMOSA (simulated annealing)',
      nsga2: 'Standard NSGA-II',
      greedy: 'Repeated greedy (DSatur baseline)',
    };
    return ALGORITHMS.map((algo) => {
      const row = stats?.algorithmComparison?.find((a) => a.id === algo.id);
      const pts = fronts[algo.id as keyof typeof fronts];
      const greedyRan = algo.id === 'greedy' && stats?.baselineMethod != null;
      let status: 'ok' | 'no_feasible' | 'failed' | 'not_run' | 'demo' = 'ok';
      if (!stats) status = algo.id === 'memetic' ? 'demo' : 'not_run';
      else if (algo.id !== 'memetic' && !row && !greedyRan) status = 'not_run';
      else if (row?.status === 'failed') status = 'failed';
      else if (!pts.length) status = 'no_feasible';
      const measured = status === 'ok' || status === 'no_feasible';
      const hv = !measured ? null : algo.id === 'memetic' ? stats!.hypervolume : row?.hv ?? (greedyRan ? stats!.baselineHypervolume : null);
      const time = !measured ? null : algo.id === 'memetic' ? stats!.runtime : row?.runtime ?? (greedyRan ? stats!.baselineRuntime : null);
      return { id: algo.id, name: names[algo.id], color: algo.color, pts, hv, time, status, error: row?.error };
    });
  }, [fronts, stats]);
  const ranCount = comparisonRows.filter((r) => r.status === 'ok' || r.status === 'no_feasible').length;

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Multi-Objective Pareto Trade-Off Explorer</h1>
          <p>
            {stats ? (
              <>
                Trade-offs found by the last optimiser run ({ranCount} of {ALGORITHMS.length} algorithms run). Every point is an independently
                checked, clash-free timetable.
              </>
            ) : (
              <>
                <b>Offline demo data.</b> These sample timetables are not optimiser output. Run the optimiser on the dashboard to see measured
                results.
              </>
            )}
          </p>
        </div>
        <div className="spacer" />
        <label className="check" style={{ fontSize: 13 }}>
          <input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} /> Show dominated candidate solutions
        </label>
      </div>

      {/* Algorithm Legend & Filter Toggle Bar */}
      <div className="card" style={{ padding: '12px 16px', marginBottom: 18, background: '#faf6ee', border: '1px solid #e3d9c6' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
          <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--ink)' }}>Toggle Algorithms:</span>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16 }}>
            {ALGORITHMS.map((algo) => (
              <label
                key={algo.id}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  cursor: 'pointer',
                  fontSize: 12,
                  fontWeight: enabledAlgos[algo.id] ? 600 : 400,
                  color: enabledAlgos[algo.id] ? algo.color : '#94a3b8',
                }}
              >
                <input
                  type="checkbox"
                  checked={enabledAlgos[algo.id] ?? true}
                  onChange={() => toggleAlgo(algo.id)}
                  style={{ accentColor: algo.color }}
                />
                <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: algo.shape === 'circle' ? 99 : 2, background: algo.color }} />
                <span>
                  {algo.name}
                  {comparisonRows.find((r) => r.id === algo.id)?.status === 'not_run' && ' (not run)'}
                </span>
              </label>
            ))}
          </div>
        </div>
      </div>

      {/* Representative Picks Legend */}
      <div className="row" style={{ marginBottom: 14, gap: 16 }}>
        <span style={{ fontSize: 11, fontWeight: 600, color: '#6b655c' }}>Special Picks (Memetic Front):</span>
        {(Object.keys(pickLabels) as PickKind[]).map((k) => (
          <span key={k} className="row tiny" style={{ gap: 6, color: 'var(--ink-2)' }}>
            <i style={{ width: 10, height: 10, borderRadius: 99, background: pickColor[k], display: 'inline-block' }} /> {pickLabels[k].title}
          </span>
        ))}
      </div>

      {/* 2D Projection Scatter Plots with Axis Labels */}
      <div className="grid grid-3" style={{ marginBottom: 24, gap: 18 }}>
        {([['f1', 'f2'], ['f1', 'f3'], ['f2', 'f3']] as [keyof Objectives, keyof Objectives][]).map(([x, y]) => (
          <div key={x + y} className="card" style={{ padding: 14 }}>
            <h3 style={{ margin: '0 0 8px', fontSize: 13, color: 'var(--ink)' }}>
              {AXES[x].short} vs. {AXES[y].short}
            </h3>
            <ScatterPlot
              xk={x}
              yk={y}
              all={solutions}
              picks={picks}
              selectedId={selectedId}
              showAll={showAll}
              onPick={select}
              enabledAlgos={enabledAlgos}
              fronts={fronts}
            />
          </div>
        ))}
      </div>

      {/* Full 5-Algorithm Benchmark Comparison Table */}
      <div className="card table-wrap" style={{ marginBottom: 24 }}>
        <h2 style={{ marginBottom: 6 }}>Algorithm comparison (last run)</h2>
        <p className="tiny" style={{ marginTop: 0, color: '#6b655c' }}>
          Measured values only, all under the same evaluation budget on the same instance. Lower is better for f1–f3; higher is better for
          hypervolume (HV). Algorithms that were not part of the run show “Not run”: turn on the comparison options on the dashboard to include
          them.
        </p>
        <table className="table">
          <thead>
            <tr>
              <th>Algorithm</th>
              <th>Status</th>
              <th>Hypervolume (HV)</th>
              <th>Pareto Front Size</th>
              <th>Best Idle Gaps (f1)</th>
              <th>Best Room Wastage (f2)</th>
              <th>Best Imbalance (f3)</th>
              <th>Compute Time</th>
            </tr>
          </thead>
          <tbody>
            {comparisonRows.map((row) => {
              const bestVal = (k: keyof Objectives) =>
                row.pts.length ? Math.min(...row.pts.map((p) => p[k])) : null;
              const f1b = bestVal('f1');
              const f2b = bestVal('f2');
              const f3b = bestVal('f3');
              return (
                <tr key={row.id}>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{ width: 10, height: 10, borderRadius: 2, background: row.color, display: 'inline-block' }} />
                      <b>{row.name}</b>
                    </div>
                  </td>
                  <td>
                    {row.status === 'ok' && <span className="chip" style={{ background: '#dcfce7', color: '#15803d' }}>Feasible</span>}
                    {row.status === 'no_feasible' && <span className="chip" style={{ background: '#fee2e2', color: '#b91c1c' }}>No feasible timetable</span>}
                    {row.status === 'failed' && <span className="chip" style={{ background: '#fee2e2', color: '#b91c1c' }} title={row.error}>Run failed</span>}
                    {row.status === 'not_run' && <span className="chip">Not run</span>}
                    {row.status === 'demo' && <span className="chip">Demo data</span>}
                  </td>
                  <td><b>{row.hv != null ? row.hv.toFixed(4) : 'No data'}</b></td>
                  <td>{row.status === 'not_run' || row.status === 'failed' ? '—' : row.pts.length}</td>
                  <td>{f1b != null ? AXES.f1.fmt(f1b) : '—'}</td>
                  <td>{f2b != null ? AXES.f2.fmt(f2b) : '—'}</td>
                  <td>{f3b != null ? AXES.f3.fmt(f3b) : '—'}</td>
                  <td>{row.time != null ? `${row.time.toFixed(2)}s` : '—'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* TOPSIS / AHP decision support on the memetic front */}
      <DecisionPanel front={front} selectedId={selectedId} onSelect={select} />

      {/* Selected Timetable Inspector */}
      <div className="grid split">
        <div className="card table-wrap">
          <h2 style={{ marginBottom: 10 }}>Discovered Non-Dominated Timetables (Memetic Front)</h2>
          <table className="table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Idle Gaps (f1)</th>
                <th>Room Waste (f2)</th>
                <th>Imbalance (f3)</th>
                <th>Designation</th>
              </tr>
            </thead>
            <tbody>
              {front.map((s) => {
                const pk = (Object.keys(picks) as PickKind[]).filter((k) => picks[k].id === s.id);
                return (
                  <tr key={s.id} className={s.id === selectedId ? 'sel' : ''} onClick={() => select(s.id)}>
                    <td><b>{s.id}</b></td>
                    <td>{s.objectives.f1.toFixed(1)}</td>
                    <td>{(s.objectives.f2 * 100).toFixed(1)}%</td>
                    <td>{s.objectives.f3.toFixed(3)}</td>
                    <td>
                      {pk.map((k) => (
                        <span key={k} className="chip" style={{ background: pickColor[k], color: '#fff', marginRight: 4 }}>
                          {pickLabels[k].title}
                        </span>
                      ))}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="card tape pick">
          <h2>Selected: {selected.id}</h2>
          <span className="tiny">{selected.onFront ? 'Pareto-optimal trade-off solution' : 'Dominated alternative'}</span>
          <ObjectiveBars s={selected} front={front} />
          <button className="btn accent" style={{ marginTop: 14 }} onClick={() => nav('/designer')}>
            Open in Timetable Designer →
          </button>
        </div>
      </div>
    </>
  );
}
