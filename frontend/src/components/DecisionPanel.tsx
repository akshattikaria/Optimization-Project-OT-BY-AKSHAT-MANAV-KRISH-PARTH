import { useEffect, useMemo, useState } from 'react';
import { fetchDecision } from '../api';
import type { DecisionResult, Solution } from '../types';

const OBJECTIVES = [
  { key: 'gaps', label: 'Fewer idle gaps (f1)', short: 'Gaps' },
  { key: 'room', label: 'Better room fit (f2)', short: 'Room fit' },
  { key: 'workload', label: 'Fairer workload (f3)', short: 'Workload' },
] as const;

const PROFILES: { id: string; label: string; weights: [number, number, number] }[] = [
  { id: 'balanced', label: 'Balanced', weights: [1, 1, 1] },
  { id: 'student_first', label: 'Students first', weights: [3, 1, 1] },
  { id: 'admin_room_fit', label: 'Room fit first', weights: [1, 3, 1] },
  { id: 'faculty_fairness', label: 'Faculty fairness first', weights: [1, 1, 3] },
];

// Saaty's scale for one pairwise judgement "A compared with B".
const SAATY: { value: number; label: string }[] = [
  { value: 9, label: 'A extremely more important (9)' },
  { value: 7, label: 'A very strongly more (7)' },
  { value: 5, label: 'A strongly more (5)' },
  { value: 3, label: 'A moderately more (3)' },
  { value: 1, label: 'Equally important (1)' },
  { value: 1 / 3, label: 'B moderately more (1/3)' },
  { value: 1 / 5, label: 'B strongly more (1/5)' },
  { value: 1 / 7, label: 'B very strongly more (1/7)' },
  { value: 1 / 9, label: 'B extremely more (1/9)' },
];

const PAIRS: [string, string][] = [
  ['Idle gaps', 'Room fit'],
  ['Idle gaps', 'Workload balance'],
  ['Room fit', 'Workload balance'],
];

const pct = (v: number) => `${(v * 100).toFixed(0)}%`;

/** TOPSIS recommendation from the Pareto front, with AHP-derived weights and a weight-sensitivity analysis.
 *  Every number shown comes from the server's `timetable.mcdm` module; nothing is computed or estimated here. */
export default function DecisionPanel({ front, selectedId, onSelect }: { front: Solution[]; selectedId: string; onSelect: (id: string) => void }) {
  const [mode, setMode] = useState<'weights' | 'ahp'>('weights');
  const [weights, setWeights] = useState<[number, number, number]>([1, 1, 1]);
  const [judgements, setJudgements] = useState<[number, number, number]>([1, 1, 1]);
  const [result, setResult] = useState<DecisionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const objectives = useMemo(() => front.map((s) => [s.objectives.f1, s.objectives.f2, s.objectives.f3]), [front]);
  const weightSum = weights[0] + weights[1] + weights[2];

  useEffect(() => {
    if (!objectives.length || (mode === 'weights' && weightSum <= 0)) {
      setResult(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    const timer = setTimeout(() => {
      fetchDecision(objectives, mode === 'weights' ? weights : null, mode === 'ahp' ? judgements : null)
        .then((r) => {
          if (!cancelled) {
            setResult(r);
            setError(null);
          }
        })
        .catch((e: Error) => {
          if (!cancelled) {
            setResult(null);
            setError(e.message);
          }
        })
        .finally(() => !cancelled && setLoading(false));
    }, 250);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [objectives, mode, weights, judgements, weightSum]);

  const idOf = (index: number) => front[index]?.id ?? `#${index + 1}`;
  const recommended = result && result.recommended != null ? front[result.recommended] : null;
  const sens = result?.sensitivity ?? null;

  return (
    <div className="card" style={{ marginBottom: 24 }}>
      <h2 style={{ marginBottom: 4 }}>Recommended timetable (TOPSIS decision support)</h2>
      <p className="tiny" style={{ marginTop: 0 }}>
        The front offers {front.length} non-dominated timetable{front.length === 1 ? '' : 's'}. TOPSIS ranks them by closeness to the best value of
        every objective and distance from the worst, using the importance you give each objective.
      </p>

      <div className="seg" style={{ margin: '10px 0' }}>
        <button className={mode === 'weights' ? 'on' : ''} onClick={() => setMode('weights')}>Set weights directly</button>
        <button className={mode === 'ahp' ? 'on' : ''} onClick={() => setMode('ahp')}>Derive weights by pairwise comparison (AHP)</button>
      </div>

      {mode === 'weights' ? (
        <div>
          <div className="seg" style={{ marginBottom: 10 }}>
            {PROFILES.map((p) => (
              <button key={p.id} className={p.weights.every((w, i) => w === weights[i]) ? 'on' : ''} onClick={() => setWeights(p.weights)}>
                {p.label}
              </button>
            ))}
          </div>
          <div className="grid grid-3" style={{ gap: 14 }}>
            {OBJECTIVES.map((o, i) => (
              <label key={o.key} style={{ fontSize: 12 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontWeight: 600 }}>
                  <span>{o.label}</span>
                  <span>{weightSum > 0 ? pct(weights[i] / weightSum) : '—'}</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={10}
                  step={0.5}
                  value={weights[i]}
                  style={{ width: '100%' }}
                  onChange={(e) => {
                    const next = [...weights] as [number, number, number];
                    next[i] = Number(e.target.value);
                    setWeights(next);
                  }}
                />
              </label>
            ))}
          </div>
          {weightSum <= 0 && <p className="tiny" style={{ color: '#b91c1c' }}>Give at least one objective a weight above zero.</p>}
        </div>
      ) : (
        <div>
          <div className="grid grid-3" style={{ gap: 14 }}>
            {PAIRS.map(([a, b], i) => (
              <label key={a + b} style={{ fontSize: 12 }}>
                <div style={{ fontWeight: 600, marginBottom: 4 }}>A: {a} · B: {b}</div>
                <select
                  value={judgements[i]}
                  style={{ width: '100%' }}
                  onChange={(e) => {
                    const next = [...judgements] as [number, number, number];
                    next[i] = Number(e.target.value);
                    setJudgements(next);
                  }}
                >
                  {SAATY.map((s) => (
                    <option key={s.label} value={s.value}>{s.label}</option>
                  ))}
                </select>
              </label>
            ))}
          </div>
          {result?.ahp && (
            <p className="tiny" style={{ marginBottom: 0 }}>
              AHP weights: {OBJECTIVES.map((o, i) => `${o.short} ${pct(result.ahp!.weights[i])}`).join(' · ')}. Consistency ratio{' '}
              <b style={{ color: result.ahp.consistent ? '#15803d' : '#b91c1c' }}>{result.ahp.consistencyRatio.toFixed(3)}</b>{' '}
              {result.ahp.consistent
                ? '(at or below 0.10: the three judgements agree with each other).'
                : '(above 0.10: the three judgements contradict each other; revise them before relying on these weights).'}
            </p>
          )}
        </div>
      )}

      {error && (
        <p className="tiny" style={{ color: '#b91c1c', marginTop: 12 }}>
          No recommendation: {error}. The ranking is computed by the optimiser server, so nothing is shown without it.
        </p>
      )}
      {!error && !result && <p className="tiny" style={{ marginTop: 12 }}>{loading ? 'Ranking the front…' : 'No data.'}</p>}

      {result && recommended && (
        <div className="grid split" style={{ marginTop: 16, gap: 18 }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
              <span className="chip accent">Recommended</span>
              <b style={{ fontSize: 18 }}>{recommended.id}</b>
              <span className="tiny">closeness score {result.ranking[0].score.toFixed(3)} (1 = best on every objective)</span>
              <button className="btn sm" disabled={recommended.id === selectedId} onClick={() => onSelect(recommended.id)}>
                {recommended.id === selectedId ? 'Selected' : 'Select this timetable'}
              </button>
            </div>
            <table className="table" style={{ marginTop: 10 }}>
              <thead>
                <tr><th>Rank</th><th>ID</th><th>Score</th><th>Idle gaps (f1)</th><th>Room waste (f2)</th><th>Imbalance (f3)</th></tr>
              </thead>
              <tbody>
                {result.ranking.slice(0, 5).map((r) => (
                  <tr key={r.index} className={idOf(r.index) === selectedId ? 'sel' : ''} onClick={() => onSelect(idOf(r.index))}>
                    <td>{r.rank}</td>
                    <td><b>{idOf(r.index)}</b></td>
                    <td>{r.score.toFixed(3)}</td>
                    <td>{r.objectives[0].toFixed(1)}</td>
                    <td>{(r.objectives[1] * 100).toFixed(1)}%</td>
                    <td>{r.objectives[2].toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {result.ranking.length > 5 && <p className="tiny">Top 5 of {result.ranking.length} ranked timetables.</p>}
          </div>

          {sens && (
            <div>
              <h3 style={{ margin: '0 0 6px' }}>How sensitive is this choice to the weights?</h3>
              {sens.distinct_choices === 1 ? (
                <p className="tiny" style={{ marginTop: 0 }}>
                  {idOf(sens.weight_space[0].index)} is the top choice for all {sens.grid_points} weight combinations tested, so the weights do not
                  change the recommendation.
                </p>
              ) : (
                <>
                  <p className="tiny" style={{ marginTop: 0 }}>
                    Share of {sens.grid_points} evenly spread weight combinations in which each timetable ranks first:
                  </p>
                  {sens.weight_space.slice(0, 6).map((w) => (
                    <div key={w.index} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, marginBottom: 4 }}>
                      <b style={{ width: 44 }}>{idOf(w.index)}</b>
                      <div style={{ flex: 1, background: '#eee6d8', borderRadius: 4, height: 10 }}>
                        <div style={{ width: `${w.share * 100}%`, height: 10, borderRadius: 4, background: w.index === result.recommended ? '#2563eb' : '#94a3b8' }} />
                      </div>
                      <span style={{ width: 40, textAlign: 'right' }}>{(w.share * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                  {sens.weight_space.length > 6 && <p className="tiny">{sens.weight_space.length - 6} more timetables each win a smaller share.</p>}
                </>
              )}
              <p className="tiny" style={{ marginBottom: 4 }}>
                {sens.stability == null
                  ? 'Changing any single weight from 0% to 100% (the other two keeping their ratio) never changes the recommendation.'
                  : `The recommendation changes if a single weight moves by about ${(sens.stability * 100).toFixed(0)} percentage points.`}
              </p>
              {sens.sweeps.map((sw) => (
                <div key={sw.objective} style={{ fontSize: 12, marginBottom: 6 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>Weight on {OBJECTIVES.find((o) => o.key === sw.objective)?.short.toLowerCase()} from 0% to 100% (now {pct(sw.base_weight)})</span>
                  </div>
                  <div style={{ display: 'flex', height: 18, borderRadius: 4, overflow: 'hidden', border: '1px solid #e3d9c6' }}>
                    {sw.segments.map((seg, k) => {
                      const start = k === 0 ? 0 : (sw.segments[k - 1].to + seg.from) / 2;
                      const end = k === sw.segments.length - 1 ? 1 : (seg.to + sw.segments[k + 1].from) / 2;
                      return (
                        <div
                          key={k}
                          title={`${idOf(seg.index)} ranks first while this weight is between ${pct(start)} and ${pct(end)}`}
                          style={{
                            width: `${(end - start) * 100}%`,
                            background: seg.index === result.recommended ? '#bfdbfe' : k % 2 ? '#f1e9da' : '#e7dcc8',
                            fontSize: 10,
                            lineHeight: '18px',
                            textAlign: 'center',
                            overflow: 'hidden',
                            whiteSpace: 'nowrap',
                          }}
                        >
                          {idOf(seg.index)}
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
