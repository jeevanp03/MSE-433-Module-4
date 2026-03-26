import { useMemo } from 'react';
import {
  GitCompareArrows,
  Lightbulb,
  Timer,
  Target,
  TrendingDown,
} from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, Cell, ReferenceLine, Legend,
} from 'recharts';
import { motion } from 'framer-motion';
import dashboardData from '../data/dashboard_data.json';
import type { DashboardData, RepoCaseData } from '../types';

const data = dashboardData as DashboardData;
const repo = data.repositioning;
const gs = repo.globalStats;

const PHYSICIAN_COLORS: Record<string, string> = {
  'Dr. A': '#3b82f6',
  'Dr. B': '#ef4444',
  'Dr. C': '#22c55e',
};

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

function StatCard({ label, value, sub, color = 'blue' }: { label: string; value: string; sub?: string; color?: string }) {
  const colorMap: Record<string, string> = {
    blue: 'bg-blue-50 border-blue-200 text-blue-700',
    red: 'bg-red-50 border-red-200 text-red-700',
    green: 'bg-green-50 border-green-200 text-green-700',
    amber: 'bg-amber-50 border-amber-200 text-amber-700',
  };
  return (
    <div className={`rounded-xl border p-4 ${colorMap[color] ?? colorMap.blue}`}>
      <p className="text-xs font-medium opacity-70 mb-1">{label}</p>
      <p className="text-2xl font-bold">{value}</p>
      {sub && <p className="text-xs opacity-60 mt-1">{sub}</p>}
    </div>
  );
}

// -- Key Metrics Overview --

function MetricsOverview() {
  return (
    <motion.div {...fadeIn}>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <StatCard
          label="Avg Repositioning Time"
          value={`${gs.globalRepoMean} min`}
          sub={`${gs.repoPctOfAbl}% of ABL DURATION`}
          color="amber"
        />
        <StatCard
          label="Correlation with PT IN-OUT"
          value={`r = ${gs.repoCorrelation}`}
          sub={`R² = ${gs.repoR2Pct}% variance explained`}
          color="red"
        />
        <StatCard
          label="Best-in-Class Rate"
          value={`${gs.bestInClassRate} min/site`}
          sub={gs.bestInClassPhys}
          color="green"
        />
        <StatCard
          label="Program Savings Potential"
          value={`${gs.totalProgramSavingsMin} min`}
          sub={`${gs.avgSavingsPerCaseMin} min/case avg`}
          color="blue"
        />
      </div>
    </motion.div>
  );
}

// -- Ablation Phase Breakdown --

function AblationBreakdown() {
  const physicians = Object.keys(repo.perPhysician);

  const chartData = useMemo(() =>
    physicians.map(p => {
      const ps = repo.perPhysician[p];
      return {
        name: p,
        pulseOn: ps.ablTimeMean,
        repositioning: ps.mean,
        total: ps.ablDurationMean,
        repoPct: ps.repoPctOfAbl,
      };
    }),
    [physicians]
  );

  return (
    <motion.div {...fadeIn} transition={{ delay: 0.1 }} className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <div className="flex items-center gap-2 mb-4">
        <Timer size={20} className="text-[#1e40af]" />
        <h2 className="text-lg font-semibold text-gray-900">Ablation Phase Breakdown</h2>
      </div>

      <div className="flex flex-col lg:flex-row gap-6">
        <div className="flex-1 min-w-0">
          <ResponsiveContainer width="100%" height={340}>
            <BarChart data={chartData} margin={{ top: 20, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="name" tick={{ fontSize: 12 }} />
              <YAxis
                tick={{ fontSize: 11 }}
                label={{ value: 'Duration (min)', angle: -90, position: 'insideLeft', style: { fontSize: 11, fill: '#9ca3af' } }}
              />
              <Tooltip
                contentStyle={{ borderRadius: 8, fontSize: 12, border: '1px solid #e5e7eb' }}
                content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const d = payload[0]?.payload;
                  return (
                    <div className="bg-white border border-gray-200 rounded-lg p-3 shadow-lg text-xs">
                      <p className="font-semibold text-gray-800 mb-1">{d.name}</p>
                      <p className="text-green-600">Pulse-On: {d.pulseOn} min</p>
                      <p className="text-amber-600">Repositioning: {d.repositioning} min ({d.repoPct}%)</p>
                      <p className="text-gray-500 font-semibold">Total: {d.total} min</p>
                    </div>
                  );
                }}
              />
              <Bar dataKey="pulseOn" stackId="a" fill="#22c55e" name="Pulse-On (ABL TIME)" radius={[0, 0, 0, 0]} />
              <Bar dataKey="repositioning" stackId="a" fill="#f59e0b" name="Repositioning" radius={[4, 4, 0, 0]} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="lg:w-72 shrink-0 flex flex-col gap-4">
          <InsightCard>
            On average, <strong>{gs.repoPctOfAbl}%</strong> of ablation duration is spent repositioning the
            catheter between ablation sites, not delivering energy. Pulse-on time is nearly constant
            across physicians ({gs.globalAblTimeMean} min avg), meaning the variability comes entirely from
            repositioning technique.
          </InsightCard>
          <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-xs text-amber-700">
            <p className="font-semibold mb-1">Why repositioning takes so long</p>
            <p className="leading-relaxed">
              Each catheter move requires real-time contact quality verification, sheath angle adjustment,
              and gap checking on the 3D map. This is inherently dynamic and technique-dependent &mdash;
              not a simple routing problem.
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

// -- Per-Site Efficiency Comparison --

function EfficiencyComparison() {
  const physicians = Object.keys(repo.perPhysician);

  const chartData = useMemo(() =>
    physicians.map(p => ({
      name: p,
      rate: repo.perPhysician[p].repoPerSiteMean,
      std: repo.perPhysician[p].repoPerSiteStd,
      fill: PHYSICIAN_COLORS[p] ?? '#6b7280',
    })),
    [physicians]
  );

  const best = gs.bestInClassPhys;
  const bestRate = gs.bestInClassRate;

  return (
    <motion.div {...fadeIn} transition={{ delay: 0.2 }} className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <div className="flex items-center gap-2 mb-4">
        <Target size={20} className="text-[#1e40af]" />
        <h2 className="text-lg font-semibold text-gray-900">Per-Site Repositioning Efficiency</h2>
      </div>

      <div className="flex flex-col lg:flex-row gap-6">
        <div className="flex-1 min-w-0">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData} margin={{ top: 20, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="name" tick={{ fontSize: 12 }} />
              <YAxis
                tick={{ fontSize: 11 }}
                label={{ value: 'min / ablation site', angle: -90, position: 'insideLeft', style: { fontSize: 11, fill: '#9ca3af' } }}
                domain={[0, 'auto']}
              />
              <Tooltip
                contentStyle={{ borderRadius: 8, fontSize: 12, border: '1px solid #e5e7eb' }}
                content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const d = payload[0]?.payload;
                  return (
                    <div className="bg-white border border-gray-200 rounded-lg p-3 shadow-lg text-xs">
                      <p className="font-semibold text-gray-800">{d.name}</p>
                      <p className="text-gray-500">{d.rate.toFixed(2)} &plusmn; {d.std.toFixed(2)} min/site</p>
                    </div>
                  );
                }}
              />
              <ReferenceLine
                y={gs.clinicalFloorPerSite}
                stroke="#16a34a"
                strokeDasharray="6 3"
                label={{ value: `Clinical floor (${gs.clinicalFloorPerSite} min/site)`, position: 'right', style: { fontSize: 10, fill: '#16a34a' } }}
              />
              <Bar dataKey="rate" name="Rate">
                {chartData.map((entry, i) => (
                  <Cell key={i} fill={entry.fill} opacity={0.8} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="lg:w-72 shrink-0 flex flex-col gap-4">
          {/* Efficiency gap highlight */}
          {physicians.filter(p => p !== best).map(p => {
            const gap = repo.perPhysician[p].repoPerSiteMean - bestRate;
            const gapPct = ((gap / bestRate) * 100).toFixed(0);
            if (gap <= 0.05) return null;
            return (
              <div key={p} className="bg-red-50 border border-red-200 rounded-xl p-4">
                <div className="flex items-center gap-2 mb-1">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: PHYSICIAN_COLORS[p] }} />
                  <span className="text-sm font-bold text-red-700">{p}: +{gapPct}% slower</span>
                </div>
                <p className="text-xs text-red-600 leading-relaxed">
                  {p} averages {repo.perPhysician[p].repoPerSiteMean} min/site vs {best}'s {bestRate} min/site.
                  This {gap.toFixed(2)} min/site gap compounds across {repo.perPhysician[p].n} cases.
                </p>
              </div>
            );
          })}
          <InsightCard>
            The efficiency gap is <strong>technique-driven</strong>, not complexity-driven.
            #ABL vs repositioning time correlation is only r={gs.sitesVsRepoCorrelation} &mdash;
            more sites don't proportionally increase repositioning time. The difference is in
            catheter handling, contact verification speed, and workflow discipline.
          </InsightCard>
        </div>
      </div>
    </motion.div>
  );
}

// -- Scatter: Repo time vs PT IN-OUT --

function RepoScatter() {
  const cases: RepoCaseData[] = repo.perCase;
  const physicians = [...new Set(cases.map(c => c.physician))];

  const scatterData = useMemo(() =>
    cases.map(c => ({
      x: c.repoTime,
      y: c.ptInOut,
      physician: c.physician,
      caseNum: c.caseNum,
      outlier: c.outlierClass === 1,
      fill: PHYSICIAN_COLORS[c.physician] ?? '#6b7280',
    })),
    [cases]
  );

  return (
    <motion.div {...fadeIn} transition={{ delay: 0.3 }} className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <div className="flex items-center gap-2 mb-4">
        <TrendingDown size={20} className="text-[#1e40af]" />
        <h2 className="text-lg font-semibold text-gray-900">Repositioning Time vs Total Duration</h2>
      </div>

      <div className="flex flex-col lg:flex-row gap-6">
        <div className="flex-1 min-w-0">
          <ResponsiveContainer width="100%" height={380}>
            <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis
                type="number"
                dataKey="x"
                name="Repositioning"
                tick={{ fontSize: 11 }}
                label={{ value: 'Repositioning Time (min)', position: 'insideBottom', offset: -8, style: { fontSize: 11, fill: '#9ca3af' } }}
              />
              <YAxis
                type="number"
                dataKey="y"
                name="PT IN-OUT"
                tick={{ fontSize: 11 }}
                label={{ value: 'PT IN-OUT (min)', angle: -90, position: 'insideLeft', style: { fontSize: 11, fill: '#9ca3af' } }}
              />
              <Tooltip
                contentStyle={{ borderRadius: 8, fontSize: 12, border: '1px solid #e5e7eb' }}
                content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const d = payload[0]?.payload;
                  return (
                    <div className="bg-white border border-gray-200 rounded-lg p-3 shadow-lg text-xs">
                      <p className="font-semibold text-gray-800">Case #{d.caseNum} ({d.physician})</p>
                      <p className="text-gray-500">Repositioning: {d.x} min</p>
                      <p className="text-gray-500">PT IN-OUT: {d.y} min</p>
                      {d.outlier && <p className="text-red-600 font-semibold mt-1">Outlier case</p>}
                    </div>
                  );
                }}
              />
              <ReferenceLine y={data.metadata.threshold} stroke="#f59e0b" strokeDasharray="6 3" />
              {physicians.map(p => (
                <Scatter
                  key={p}
                  name={p}
                  data={scatterData.filter(d => d.physician === p)}
                  fill={PHYSICIAN_COLORS[p]}
                  opacity={0.7}
                />
              ))}
              <Legend wrapperStyle={{ fontSize: 11 }} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        <div className="lg:w-72 shrink-0 flex flex-col gap-4">
          <div className="bg-gray-50 border border-gray-200 rounded-xl p-4">
            <h3 className="text-sm font-semibold text-gray-800 mb-2">Correlation Summary</h3>
            <table className="text-xs text-gray-600 w-full">
              <tbody>
                <tr className="border-b border-gray-100">
                  <td className="py-1.5">Repositioning time</td>
                  <td className="py-1.5 text-right font-bold text-red-600">r = {gs.repoCorrelation}</td>
                </tr>
                <tr className="border-b border-gray-100">
                  <td className="py-1.5">ABL DURATION (total)</td>
                  <td className="py-1.5 text-right font-semibold">r = {gs.ablDurationCorrelation}</td>
                </tr>
                <tr>
                  <td className="py-1.5">ABL TIME (pulse-on)</td>
                  <td className="py-1.5 text-right text-gray-400">r = {gs.ablTimeCorrelation}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <InsightCard>
            Repositioning time (r={gs.repoCorrelation}) is a <strong>stronger predictor</strong> of
            total duration than ABL DURATION itself (r={gs.ablDurationCorrelation}). Pulse-on time
            (r={gs.ablTimeCorrelation}) has almost no relationship with total duration.
          </InsightCard>
        </div>
      </div>
    </motion.div>
  );
}

// -- Savings Projections --

function SavingsProjections() {
  const physicians = Object.keys(repo.savingsProjections);

  const chartData = useMemo(() =>
    physicians.map(p => {
      const sav = repo.savingsProjections[p];
      return {
        name: p,
        current: sav.currentMeanRepo,
        target: sav.projectedRepoAtBest,
        savings: sav.savingsVsBestMin,
        floor: sav.projectedRepoAtFloor,
        fill: PHYSICIAN_COLORS[p] ?? '#6b7280',
      };
    }),
    [physicians]
  );

  return (
    <motion.div {...fadeIn} transition={{ delay: 0.4 }} className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <div className="flex items-center gap-2 mb-4">
        <TrendingDown size={20} className="text-[#1e40af]" />
        <h2 className="text-lg font-semibold text-gray-900">Time Savings Projections</h2>
      </div>

      <div className="flex flex-col lg:flex-row gap-6">
        <div className="flex-1 min-w-0">
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={chartData} margin={{ top: 20, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="name" tick={{ fontSize: 12 }} />
              <YAxis
                tick={{ fontSize: 11 }}
                label={{ value: 'Repositioning Time (min)', angle: -90, position: 'insideLeft', style: { fontSize: 11, fill: '#9ca3af' } }}
              />
              <Tooltip
                contentStyle={{ borderRadius: 8, fontSize: 12, border: '1px solid #e5e7eb' }}
                content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const d = payload[0]?.payload;
                  return (
                    <div className="bg-white border border-gray-200 rounded-lg p-3 shadow-lg text-xs">
                      <p className="font-semibold text-gray-800 mb-1">{d.name}</p>
                      <p className="text-gray-500">Current: {d.current} min</p>
                      <p className="text-green-600">Target ({gs.bestInClassPhys}'s rate): {d.target} min</p>
                      {d.savings > 0 && <p className="text-red-600 font-bold">Potential savings: {d.savings} min/case</p>}
                      <p className="text-gray-400 mt-1">Clinical floor: {d.floor} min</p>
                    </div>
                  );
                }}
              />
              <ReferenceLine
                y={gs.clinicalFloorTotal}
                stroke="#16a34a"
                strokeDasharray="6 3"
                label={{ value: `Clinical floor (${gs.clinicalFloorTotal} min)`, position: 'right', style: { fontSize: 10, fill: '#16a34a' } }}
              />
              <Bar dataKey="current" name="Current" opacity={0.8}>
                {chartData.map((entry, i) => (
                  <Cell key={i} fill={entry.fill} />
                ))}
              </Bar>
              <Bar dataKey="target" name={`Target (${gs.bestInClassPhys}'s rate)`} opacity={0.3}>
                {chartData.map((entry, i) => (
                  <Cell key={i} fill={entry.fill} />
                ))}
              </Bar>
              <Legend wrapperStyle={{ fontSize: 11 }} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="lg:w-72 shrink-0 flex flex-col gap-4">
          <div className="bg-green-50 border border-green-200 rounded-xl p-4">
            <p className="text-sm font-bold text-green-700 mb-2">Total Program Impact</p>
            <p className="text-2xl font-bold text-green-600">{gs.totalProgramSavingsMin} min</p>
            <p className="text-xs text-green-600 mt-1">
              saved across {data.metadata.totalCases} cases if all physicians match {gs.bestInClassPhys}'s
              repositioning rate ({gs.bestInClassRate} min/site)
            </p>
          </div>

          {physicians.filter(p => repo.savingsProjections[p].savingsVsBestMin > 0.5).map(p => {
            const sav = repo.savingsProjections[p];
            return (
              <div key={p} className="bg-gray-50 border border-gray-200 rounded-xl p-4">
                <p className="text-sm font-semibold text-gray-800 mb-1">{p}</p>
                <p className="text-xs text-gray-600 leading-relaxed">
                  Current: {sav.currentRate} min/site &rarr; Target: {sav.bestInClassTarget} min/site<br />
                  <strong className="text-red-600">{sav.savingsVsBestMin} min saved per case</strong> ({sav.currentMeanRepo} &rarr; {sav.projectedRepoAtBest} min)
                </p>
              </div>
            );
          })}

          <div className="text-xs text-gray-400 leading-relaxed">
            Clinical floor ({gs.clinicalFloorTotal} min) represents the minimum safe repositioning time
            based on tissue contact verification requirements. The target is {gs.bestInClassPhys}'s
            demonstrated rate, which is achievable through technique coaching.
          </div>
        </div>
      </div>
    </motion.div>
  );
}

// -- Outlier vs Normal Comparison --

function OutlierComparison() {
  const physicians = Object.keys(repo.perPhysician);

  const chartData = useMemo(() =>
    physicians.flatMap(p => {
      const ps = repo.perPhysician[p];
      const items = [];
      if (ps.normalMean != null) {
        items.push({ name: `${p}\nNormal`, value: ps.normalMean, fill: PHYSICIAN_COLORS[p] ?? '#6b7280', type: 'Normal' });
      }
      if (ps.outlierMean != null) {
        items.push({ name: `${p}\nOutlier`, value: ps.outlierMean, fill: '#ef4444', type: 'Outlier' });
      }
      return items;
    }),
    [physicians]
  );

  return (
    <motion.div {...fadeIn} transition={{ delay: 0.5 }} className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <div className="flex items-center gap-2 mb-4">
        <GitCompareArrows size={20} className="text-[#1e40af]" />
        <h2 className="text-lg font-semibold text-gray-900">Outlier vs Normal: Repositioning Time</h2>
      </div>

      <div className="flex flex-col lg:flex-row gap-6">
        <div className="flex-1 min-w-0">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData} margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="name" tick={{ fontSize: 10 }} interval={0} />
              <YAxis
                tick={{ fontSize: 11 }}
                label={{ value: 'Repositioning Time (min)', angle: -90, position: 'insideLeft', style: { fontSize: 11, fill: '#9ca3af' } }}
              />
              <Tooltip
                contentStyle={{ borderRadius: 8, fontSize: 12, border: '1px solid #e5e7eb' }}
                content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const d = payload[0]?.payload;
                  return (
                    <div className="bg-white border border-gray-200 rounded-lg p-3 shadow-lg text-xs">
                      <p className="font-semibold text-gray-800">{d.name.replace('\n', ' - ')}</p>
                      <p className="text-gray-500">Avg Repositioning: {d.value} min</p>
                    </div>
                  );
                }}
              />
              <Bar dataKey="value" name="Avg Repositioning Time">
                {chartData.map((entry, i) => (
                  <Cell key={i} fill={entry.fill} opacity={entry.type === 'Outlier' ? 0.9 : 0.7} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="lg:w-72 shrink-0 flex flex-col gap-4">
          <div className="bg-red-50 border border-red-200 rounded-xl p-4">
            <p className="text-sm font-bold text-red-700 mb-1">Outlier Gap</p>
            <p className="text-xs text-red-600 leading-relaxed">
              Outlier cases average <strong>{gs.outlierMean} min</strong> repositioning vs <strong>{gs.normalMean} min</strong> for
              normal cases &mdash; a <strong>+{gs.diffMin} min</strong> difference ({((gs.diffMin / gs.normalMean) * 100).toFixed(0)}% higher).
            </p>
          </div>
          <InsightCard>
            The outlier-normal repositioning gap ({gs.diffMin} min) alone accounts for a significant portion of the
            total duration difference. Reducing repositioning time is the single highest-leverage intervention.
          </InsightCard>
        </div>
      </div>
    </motion.div>
  );
}

// -- Main Page --

export default function RepositioningAnalysis() {
  return (
    <div>
      <div className="flex items-center gap-3 mb-6">
        <GitCompareArrows size={28} className="text-[#1e40af]" />
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Catheter Repositioning Analysis</h1>
          <p className="text-sm text-gray-400">
            The single biggest source of wasted time inside the ablation phase
          </p>
        </div>
      </div>

      <MetricsOverview />

      <div className="space-y-6">
        <AblationBreakdown />
        <EfficiencyComparison />
        <RepoScatter />
        <SavingsProjections />
        <OutlierComparison />
      </div>
    </div>
  );
}
