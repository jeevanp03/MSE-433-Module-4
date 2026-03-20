import { useMemo } from 'react';
import { Users, Stethoscope, Clock, AlertTriangle, BarChart3 } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, Radar, Legend, Cell,
} from 'recharts';
import { motion } from 'framer-motion';
import dashboardData from '../data/dashboard_data.json';
import type { DashboardData, PhysicianSeverity } from '../types';

const data = dashboardData as DashboardData;

const PHYSICIAN_COLORS: Record<string, string> = {
  'Dr. A': '#3b82f6',
  'Dr. B': '#ef4444',
  'Dr. C': '#22c55e',
};

// Compute once at module level since data is a static import
const physKeys = Object.keys(data.physicians);

const fade = (delay = 0) => ({
  initial: { opacity: 0, y: 20 } as const,
  animate: { opacity: 1, y: 0 } as const,
  transition: { delay, duration: 0.4 },
});

function StatCell({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div className="bg-gray-50 rounded-lg p-2.5 text-center">
      <p className="text-xs text-gray-500">{label}</p>
      <p className="text-lg font-bold text-gray-900">{value}</p>
      <p className="text-xs text-gray-400">{sub}</p>
    </div>
  );
}

export default function PhysicianComparison() {
  const { physicians, cases, metadata } = data;

  const radarData = useMemo(() => {
    const timingPhases = [
      'PT PREP/INTUBATION',
      'ACCESSS',
      'TSP',
      'PRE-MAP',
      'ABL DURATION',
      'POST CARE/EXTUBATION',
    ];
    const shortLabels: Record<string, string> = {
      'PT PREP/INTUBATION': 'PREP/INTUB',
      'POST CARE/EXTUBATION': 'POST CARE',
    };
    return timingPhases.map((phase) => {
      const entry: Record<string, string | number> = {
        phase: shortLabels[phase] ?? phase,
      };
      for (const phys of physKeys) {
        const physCases = cases.filter((c) => c.physician === phys);
        const avg =
          physCases.reduce(
            (sum, c) => sum + (c.features[phase] ?? 0),
            0,
          ) / physCases.length;
        entry[phys] = Math.round(avg * 10) / 10;
      }
      return entry;
    });
  }, [cases]);

  const durationData = physKeys.map((p) => ({
    physician: p,
    mean: physicians[p].meanDuration,
    median: physicians[p].medianDuration,
    color: PHYSICIAN_COLORS[p],
  }));

  const outlierRateData = physKeys.map((p) => ({
    physician: p,
    rate:
      Math.round(
        (physicians[p].outlierCount / physicians[p].caseCount) * 1000,
      ) / 10,
    count: physicians[p].outlierCount,
    total: physicians[p].caseCount,
    color: PHYSICIAN_COLORS[p],
  }));

  return (
    <div>
      <div className="flex items-center gap-3 mb-6">
        <Users size={28} className="text-[#1e40af]" />
        <h1 className="text-2xl font-bold text-gray-900">Physician Comparison</h1>
      </div>

      {/* Physician stat cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        {physKeys.map((phys, i) => {
          const stats = physicians[phys];
          const color = PHYSICIAN_COLORS[phys];
          return (
            <motion.div
              key={phys}
              {...fade(i * 0.1)}
              className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
            >
              <div className="flex items-center gap-3 mb-4">
                <div
                  className="w-10 h-10 rounded-full flex items-center justify-center text-white font-bold text-sm"
                  style={{ backgroundColor: color }}
                >
                  {phys.slice(-1)}
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900">{phys}</h3>
                  <span className="text-xs text-gray-400">
                    {stats.caseCount} cases
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <StatCell
                  label="Outliers"
                  value={`${stats.outlierCount}`}
                  sub={`${((stats.outlierCount / stats.caseCount) * 100).toFixed(1)}%`}
                />
                <StatCell label="Mean" value={`${stats.meanDuration}`} sub="min" />
                <StatCell label="Median" value={`${stats.medianDuration}`} sub="min" />
                <StatCell
                  label="IQR Threshold"
                  value={`${stats.iqrThreshold}`}
                  sub="min"
                />
              </div>

              <div className="mt-4 pt-3 border-t border-gray-100 text-xs text-gray-500">
                Q1: {stats.Q1} | Q3: {stats.Q3} | IQR: {stats.IQR}
              </div>
            </motion.div>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* Mean/Median Duration Comparison */}
        <motion.div
          {...fade(0.2)}
          className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
        >
          <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <Clock size={18} className="text-gray-500" />
            Duration Comparison
          </h2>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart
              data={durationData}
              margin={{ top: 5, right: 20, bottom: 20, left: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="physician" fontSize={12} />
              <YAxis
                fontSize={11}
                label={{
                  value: 'Minutes',
                  angle: -90,
                  position: 'insideLeft',
                  fontSize: 12,
                }}
              />
              <Tooltip
                formatter={(value, name) => [
                  `${value} min`,
                  name === 'mean' ? 'Mean' : 'Median',
                ]}
              />
              <Legend />
              <Bar dataKey="mean" name="Mean" radius={[4, 4, 0, 0]}>
                {durationData.map((d, i) => (
                  <Cell key={i} fill={d.color} />
                ))}
              </Bar>
              <Bar dataKey="median" name="Median" radius={[4, 4, 0, 0]}>
                {durationData.map((d, i) => (
                  <Cell key={i} fill={d.color} opacity={0.5} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Outlier Rate Comparison */}
        <motion.div
          {...fade(0.3)}
          className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
        >
          <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <AlertTriangle size={18} className="text-gray-500" />
            Outlier Rate by Physician
          </h2>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart
              data={outlierRateData}
              margin={{ top: 5, right: 20, bottom: 20, left: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="physician" fontSize={12} />
              <YAxis
                fontSize={11}
                label={{
                  value: 'Outlier Rate (%)',
                  angle: -90,
                  position: 'insideLeft',
                  fontSize: 12,
                }}
              />
              <Tooltip
                formatter={(value, _name, entry) => {
                  const p = (entry as { payload: { count: number; total: number } }).payload;
                  return [`${value}% (${p.count}/${p.total})`, 'Outlier Rate'];
                }}
              />
              <Bar
                dataKey="rate"
                name="Outlier Rate (%)"
                radius={[4, 4, 0, 0]}
              >
                {outlierRateData.map((d, i) => (
                  <Cell key={i} fill={d.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      {/* Radar chart */}
      <motion.div
        {...fade(0.4)}
        className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-8"
      >
        <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <BarChart3 size={18} className="text-gray-500" />
          Timing Phase Profiles
        </h2>
        <p className="text-sm text-gray-500 mb-4">
          Average minutes per timing phase for each physician. Larger area
          indicates longer phases.
        </p>
        <ResponsiveContainer width="100%" height={380}>
          <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="75%">
            <PolarGrid stroke="#e5e7eb" />
            <PolarAngleAxis dataKey="phase" fontSize={11} />
            <PolarRadiusAxis fontSize={10} angle={30} />
            {physKeys.map((phys) => (
              <Radar
                key={phys}
                name={phys}
                dataKey={phys}
                stroke={PHYSICIAN_COLORS[phys]}
                fill={PHYSICIAN_COLORS[phys]}
                fillOpacity={0.15}
                strokeWidth={2}
              />
            ))}
            <Legend />
            <Tooltip />
          </RadarChart>
        </ResponsiveContainer>
      </motion.div>

      {/* Top SHAP Drivers per Physician */}
      <motion.div
        {...fade(0.5)}
        className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-8"
      >
        <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <Stethoscope size={18} className="text-gray-500" />
          Top SHAP Drivers per Physician
        </h2>
        <p className="text-sm text-gray-500 mb-6">
          What drives outliers differently for each physician (per-physician
          LightGBM+SHAP models).
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {physKeys.map((phys) => {
            const stats = physicians[phys];
            const drivers = Object.entries(stats.topDrivers);
            const color = PHYSICIAN_COLORS[phys];

            if (drivers.length === 0) {
              return (
                <div
                  key={phys}
                  className="border border-gray-100 rounded-lg p-4"
                >
                  <div className="flex items-center gap-2 mb-3">
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: color }}
                    />
                    <span className="font-semibold text-gray-900">
                      {phys}
                    </span>
                  </div>
                  <p className="text-sm text-gray-400 italic">
                    Too few cases ({stats.caseCount}) to fit a per-physician
                    model.
                  </p>
                </div>
              );
            }

            const maxVal = Math.max(...drivers.map(([, v]) => v));
            return (
              <div
                key={phys}
                className="border border-gray-100 rounded-lg p-4"
              >
                <div className="flex items-center gap-2 mb-3">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: color }}
                  />
                  <span className="font-semibold text-gray-900">
                    {phys}
                  </span>
                </div>
                <div className="space-y-2">
                  {drivers.map(([feature, value]) => (
                    <div key={feature}>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-gray-600 truncate mr-2">
                          {feature}
                        </span>
                        <span className="text-gray-900 font-medium">
                          {value.toFixed(3)}
                        </span>
                      </div>
                      <div className="w-full bg-gray-100 rounded-full h-2">
                        <div
                          className="h-2 rounded-full transition-all duration-500"
                          style={{
                            width: `${(value / maxVal) * 100}%`,
                            backgroundColor: color,
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </motion.div>

      {/* Severity profile table */}
      <motion.div
        {...fade(0.6)}
        className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
      >
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Physician Severity Profiles
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-2 px-3 text-gray-500 font-medium">
                  Metric
                </th>
                {physKeys.map((phys) => (
                  <th
                    key={phys}
                    className="text-center py-2 px-3 font-medium"
                    style={{ color: PHYSICIAN_COLORS[phys] }}
                  >
                    {phys}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {[
                { label: 'Cases', key: 'n_cases' },
                { label: 'Mean Duration (min)', key: 'pt_in_out.mean' },
                { label: 'Std Dev (min)', key: 'pt_in_out.std' },
                { label: 'Avg Ablation Sites', key: 'abl_sites.mean' },
                { label: 'Avg Applications', key: 'applications' },
                { label: 'Avg ABL Duration', key: 'abl_duration_mean' },
                { label: 'Avg TSP', key: 'tsp_mean' },
                { label: 'Avg PRE-MAP', key: 'pre_map_mean' },
                {
                  label: '% Additional Procedures',
                  key: 'pct_additional_procedures',
                },
              ].map((row) => (
                <tr key={row.key}>
                  <td className="py-2 px-3 text-gray-600">{row.label}</td>
                  {physKeys.map((phys) => {
                    const severity: PhysicianSeverity =
                      metadata.physicianSeverity[phys];
                    let val: number | undefined;
                    if (row.key.includes('.')) {
                      const [obj, field] = row.key.split('.');
                      val = (
                        severity as unknown as Record<
                          string,
                          Record<string, number>
                        >
                      )[obj]?.[field];
                    } else {
                      val = (severity as unknown as Record<string, number>)[
                        row.key
                      ];
                    }
                    return (
                      <td
                        key={phys}
                        className="py-2 px-3 text-center text-gray-900"
                      >
                        {typeof val === 'number' ? val.toFixed(1) : '-'}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>
    </div>
  );
}
