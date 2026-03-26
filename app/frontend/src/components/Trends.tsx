import { useState, useMemo } from 'react';
import { TrendingUp, BookOpen, Layers, Lightbulb } from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine,
  BarChart, Bar, Cell,
} from 'recharts';
import { motion } from 'framer-motion';
import dashboardData from '../data/dashboard_data.json';
import type { DashboardData, LearningCurvePoint, ProcedureType } from '../types';

const data = dashboardData as DashboardData;

const PHYSICIAN_COLORS: Record<string, string> = {
  'Dr. A': '#3b82f6',
  'Dr. B': '#ef4444',
  'Dr. C': '#22c55e',
};

const OUTLIER_THRESHOLD = data.metadata.threshold;

const fadeIn = {
  initial: { opacity: 0, y: 24 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.5 },
};

function InsightCard({ children }: { children: React.ReactNode }) {
  return (
    <div className="bg-blue-50 border border-blue-200 rounded-xl p-5 flex gap-3 items-start">
      <Lightbulb size={20} className="text-[#1e40af] shrink-0 mt-0.5" />
      <p className="text-sm text-gray-700 leading-relaxed">{children}</p>
    </div>
  );
}

function SectionHeader({ icon, title }: { icon: React.ReactNode; title: string }) {
  return (
    <div className="flex items-center gap-2 mb-4">
      {icon}
      <h2 className="text-lg font-semibold text-gray-900">{title}</h2>
    </div>
  );
}

// -- Learning Curve --

function computeMovingAverage(points: { seq: number; duration: number }[], window: number) {
  return points.map((d, i) => {
    const start = Math.max(0, i - Math.floor(window / 2));
    const end = Math.min(points.length, i + Math.ceil(window / 2));
    const slice = points.slice(start, end);
    const avg = slice.reduce((s, v) => s + v.duration, 0) / slice.length;
    return { seq: d.seq, ma: Math.round(avg * 10) / 10 };
  });
}

function LearningCurveSection() {
  const rawData: LearningCurvePoint[] = data.trends.learningCurve;

  const physicians = useMemo(() => [...new Set(rawData.map(d => d.physician))], [rawData]);
  const [visible, setVisible] = useState<Record<string, boolean>>(
    () => Object.fromEntries(physicians.map(p => [p, true]))
  );
  const [showMA, setShowMA] = useState(true);

  const { chartData, maData } = useMemo(() => {
    const byPhys: Record<string, { seq: number; duration: number }[]> = {};
    const seqCounters: Record<string, number> = {};

    for (const p of physicians) {
      byPhys[p] = [];
      seqCounters[p] = 0;
    }

    for (const d of rawData) {
      seqCounters[d.physician]++;
      byPhys[d.physician].push({ seq: seqCounters[d.physician], duration: d.duration });
    }

    const maxLen = Math.max(...Object.values(byPhys).map(a => a.length));
    const merged: Record<string, number | undefined>[] = [];
    for (let i = 0; i < maxLen; i++) {
      const row: Record<string, number | undefined> = { seq: i + 1 };
      for (const p of physicians) {
        if (byPhys[p][i]) row[p] = byPhys[p][i].duration;
      }
      merged.push(row);
    }

    const maByPhys: Record<string, { seq: number; ma: number }[]> = {};
    for (const p of physicians) {
      maByPhys[p] = computeMovingAverage(byPhys[p], 7);
    }

    const maMerged: Record<string, number | undefined>[] = [];
    for (let i = 0; i < maxLen; i++) {
      const row: Record<string, number | undefined> = { seq: i + 1 };
      for (const p of physicians) {
        if (maByPhys[p][i]) row[`${p}_ma`] = maByPhys[p][i].ma;
      }
      maMerged.push(row);
    }

    return { chartData: merged, maData: maMerged };
  }, [rawData, physicians]);

  const combined = useMemo(() => {
    return chartData.map((row, i) => ({ ...row, ...(maData[i] || {}) }));
  }, [chartData, maData]);

  const togglePhysician = (phys: string) => {
    setVisible(prev => ({ ...prev, [phys]: !prev[phys] }));
  };

  return (
    <motion.div {...fadeIn} className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <SectionHeader
        icon={<BookOpen size={20} className="text-[#1e40af]" />}
        title="Learning Curve Analysis"
      />

      <div className="flex flex-col lg:flex-row gap-6">
        <div className="flex-1 min-w-0">
          {/* Legend toggles */}
          <div className="flex flex-wrap items-center gap-4 mb-4">
            {physicians.map(p => (
              <button
                key={p}
                onClick={() => togglePhysician(p)}
                aria-pressed={visible[p]}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium border transition-all cursor-pointer ${
                  visible[p]
                    ? 'border-transparent text-white'
                    : 'border-gray-300 text-gray-400 bg-white'
                }`}
                style={visible[p] ? { backgroundColor: PHYSICIAN_COLORS[p] } : {}}
              >
                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: PHYSICIAN_COLORS[p] }} />
                {p}
              </button>
            ))}
            <label className="flex items-center gap-1.5 text-xs text-gray-500 ml-auto cursor-pointer">
              <input
                type="checkbox"
                checked={showMA}
                onChange={() => setShowMA(!showMA)}
                className="rounded"
              />
              Moving average
            </label>
          </div>

          <ResponsiveContainer width="100%" height={340}>
            <LineChart data={combined} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis
                dataKey="seq"
                label={{ value: 'Case sequence (per physician)', position: 'insideBottom', offset: -2, style: { fontSize: 11, fill: '#9ca3af' } }}
                tick={{ fontSize: 11 }}
              />
              <YAxis
                label={{ value: 'Duration (min)', angle: -90, position: 'insideLeft', style: { fontSize: 11, fill: '#9ca3af' } }}
                tick={{ fontSize: 11 }}
              />
              <Tooltip
                contentStyle={{ borderRadius: 8, fontSize: 12, border: '1px solid #e5e7eb' }}
                labelFormatter={(v) => `Case #${v}`}
              />
              <ReferenceLine y={OUTLIER_THRESHOLD} stroke="#f59e0b" strokeDasharray="6 3" label={{ value: 'Outlier threshold', position: 'right', style: { fontSize: 10, fill: '#f59e0b' } }} />
              {physicians.map(p => visible[p] && (
                <Line
                  key={p}
                  type="monotone"
                  dataKey={p}
                  stroke={PHYSICIAN_COLORS[p]}
                  strokeWidth={1.5}
                  dot={{ r: 2.5, fill: PHYSICIAN_COLORS[p] }}
                  activeDot={{ r: 5 }}
                  connectNulls
                  isAnimationActive={false}
                />
              ))}
              {showMA && physicians.map(p => visible[p] && (
                <Line
                  key={`${p}_ma`}
                  type="monotone"
                  dataKey={`${p}_ma`}
                  stroke={PHYSICIAN_COLORS[p]}
                  strokeWidth={2.5}
                  strokeDasharray="6 3"
                  dot={false}
                  connectNulls
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="lg:w-72 shrink-0 flex flex-col gap-4">
          <InsightCard>
            Both physicians show improvement over time, with Dr A's learning curve flattening
            earlier. Dashed lines show 7-case moving averages to highlight the trend beneath
            case-to-case variability.
          </InsightCard>
          <div className="text-xs text-gray-400">
            Toggle physicians on/off using the colored buttons above the chart. The moving
            average overlay smooths noise to reveal the underlying learning trajectory.
          </div>
        </div>
      </div>
    </motion.div>
  );
}

// -- Case Complexity --

function ComplexitySection() {
  const procTypes: ProcedureType[] = data.complexity.procedureTypes;

  const chartData = useMemo(() =>
    procTypes.map(p => ({
      name: p.type.replace(/ \(.*\)/, ''),
      fullName: p.type,
      normal: p.totalCases - p.outlierCases,
      outlier: p.outlierCases,
      total: p.totalCases,
      outlierRate: p.outlierRate,
      meanDuration: p.meanDuration,
    })),
    [procTypes]
  );

  // Find the highest outlier rate procedure for highlight card
  const highestOutlierProc = useMemo(() =>
    procTypes.reduce((best, p) => p.outlierRate > best.outlierRate ? p : best, procTypes[0]),
    [procTypes]
  );

  // Find base/standard procedure for comparison
  const standardProc = useMemo(() =>
    procTypes.find(p => p.type.toLowerCase().includes('standard') || p.type.toLowerCase().includes('pfa only'))
      ?? procTypes.reduce((best, p) => p.totalCases > best.totalCases ? p : best, procTypes[0]),
    [procTypes]
  );

  return (
    <motion.div {...fadeIn} transition={{ duration: 0.5, delay: 0.3 }} className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <SectionHeader
        icon={<Layers size={20} className="text-[#1e40af]" />}
        title="Case Complexity & Procedure Type Impact"
      />

      <div className="flex flex-col lg:flex-row gap-6">
        <div className="flex-1 min-w-0">
          <ResponsiveContainer width="100%" height={340}>
            <BarChart data={chartData} margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis
                dataKey="name"
                tick={{ fontSize: 11 }}
                interval={0}
              />
              <YAxis
                tick={{ fontSize: 11 }}
                label={{ value: 'Number of cases', angle: -90, position: 'insideLeft', style: { fontSize: 11, fill: '#9ca3af' } }}
              />
              <Tooltip
                contentStyle={{ borderRadius: 8, fontSize: 12, border: '1px solid #e5e7eb' }}
                content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const d = payload[0]?.payload;
                  return (
                    <div className="bg-white border border-gray-200 rounded-lg p-3 shadow-lg text-xs">
                      <p className="font-semibold text-gray-800 mb-1">{d.fullName}</p>
                      <p className="text-gray-500">Total: {d.total} cases</p>
                      <p className="text-gray-500">Outliers: {d.outlier} ({d.outlierRate}%)</p>
                      <p className="text-gray-500">Mean duration: {d.meanDuration} min</p>
                    </div>
                  );
                }}
              />
              <Bar dataKey="normal" stackId="a" fill="#3b82f6" name="Normal" radius={[0, 0, 0, 0]} />
              <Bar dataKey="outlier" stackId="a" fill="#ef4444" name="Outlier" radius={[4, 4, 0, 0]}>
                {chartData.map((entry, i) => (
                  <Cell key={i} fill={entry.outlierRate >= 50 ? '#dc2626' : '#ef4444'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          {/* Rate labels under bars */}
          <div className="flex justify-around mt-1 px-8">
            {chartData.map(d => (
              <span
                key={d.name}
                className={`text-xs font-semibold ${d.outlierRate >= 50 ? 'text-red-600' : 'text-gray-400'}`}
              >
                {d.outlierRate}%
              </span>
            ))}
          </div>
        </div>

        <div className="lg:w-72 shrink-0 flex flex-col gap-4">
          {/* Highest outlier rate highlight */}
          <div className="bg-red-50 border border-red-200 rounded-xl p-5">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-3 h-3 rounded-full bg-red-500" />
              <span className="text-sm font-bold text-red-700">
                {highestOutlierProc.type.replace(/ \(.*\)/, '')}: {highestOutlierProc.outlierRate}% Outlier Rate
              </span>
            </div>
            <p className="text-xs text-red-600 leading-relaxed">
              {highestOutlierProc.type.replace(/ \(.*\)/, '')} has the highest outlier rate at {highestOutlierProc.outlierRate}%,
              with {highestOutlierProc.outlierCases} of {highestOutlierProc.totalCases} cases exceeding the duration threshold.
            </p>
          </div>

          <InsightCard>
            {standardProc.type.replace(/ \(.*\)/, '')} cases have only a {standardProc.outlierRate}% outlier rate.
            Additional ablation targets are the strongest complexity drivers for prolonged procedures.
          </InsightCard>

          {/* Legend */}
          <div className="flex items-center gap-4 text-xs text-gray-500">
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded bg-[#3b82f6]" /> Normal
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded bg-[#ef4444]" /> Outlier
            </span>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

// -- Main page --

export default function Trends() {
  return (
    <div>
      <div className="flex items-center gap-3 mb-6">
        <TrendingUp size={28} className="text-[#1e40af]" />
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Trends & Complexity</h1>
          <p className="text-sm text-gray-400">Learning curves and case complexity analysis</p>
        </div>
      </div>

      <div className="space-y-6">
        <LearningCurveSection />
        <ComplexitySection />
      </div>
    </div>
  );
}
