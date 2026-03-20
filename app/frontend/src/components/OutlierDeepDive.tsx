import { useState, useMemo } from 'react';
import { AlertTriangle, ChevronDown, ChevronUp, Filter } from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import dashboardData from '../data/dashboard_data.json';
import type { DashboardData, Case } from '../types';

const data = dashboardData as DashboardData;
const features = data.metadata.featuresUsed;
const THRESHOLD = data.metadata.threshold;

const TIMING_PHASES = [
  'PT PREP/INTUBATION',
  'ACCESSS',
  'TSP',
  'PRE-MAP',
  'ABL DURATION',
  'POST CARE/EXTUBATION',
] as const;

const PROCEDURE_FLAGS = ['NOTE_CTI', 'NOTE_BOX', 'NOTE_PST', 'NOTE_SVC'] as const;

type SortKey = 'caseNum' | 'physician' | 'ptInOut' | 'outlierClass';
type SortDir = 'asc' | 'desc';

// Build a lookup map: caseNum -> SHAP values
// shapValues has 143 entries for 145 cases; cases with null features are excluded
const shapByCaseNum: Map<number, number[]> = (() => {
  const map = new Map<number, number[]>();
  // Cases with any null feature value were excluded from the model
  const casesWithFeatures = data.cases.filter(
    c => Object.values(c.features).every(v => v !== null && v !== undefined)
  );
  // shapValues[i] corresponds to casesWithFeatures[i]
  casesWithFeatures.forEach((c, i) => {
    if (i < data.shapValues.length) {
      map.set(c.caseNum, data.shapValues[i]);
    }
  });
  return map;
})();

export default function OutlierDeepDive() {
  const cases = data.cases;

  const [sortKey, setSortKey] = useState<SortKey>('ptInOut');
  const [sortDir, setSortDir] = useState<SortDir>('desc');
  const [physicianFilter, setPhysicianFilter] = useState<string>('all');
  const [outlierOnly, setOutlierOnly] = useState(false);
  const [procedureFilters, setProcedureFilters] = useState<Record<string, boolean>>({
    NOTE_CTI: false,
    NOTE_BOX: false,
    NOTE_PST: false,
    NOTE_SVC: false,
  });
  const [expandedRow, setExpandedRow] = useState<number | null>(null);

  const physicians = useMemo(
    () => [...new Set(cases.map((c) => c.physician))].sort(),
    [cases]
  );

  const globalOutliers = useMemo(
    () => cases.filter((c) => c.outlierClass === 1),
    [cases]
  );
  const outliersByPhys = useMemo(() => {
    const counts: Record<string, number> = {};
    globalOutliers.forEach((c) => {
      counts[c.physician] = (counts[c.physician] || 0) + 1;
    });
    return counts;
  }, [globalOutliers]);
  const topPhys = useMemo(() => {
    const entries = Object.entries(outliersByPhys).sort((a, b) => b[1] - a[1]);
    return entries.length > 0 ? entries[0] : null;
  }, [outliersByPhys]);

  const filteredCases = useMemo(() => {
    let result = cases.slice();

    if (physicianFilter !== 'all') {
      result = result.filter((c) => c.physician === physicianFilter);
    }
    if (outlierOnly) {
      result = result.filter(
        (c) => c.outlierClass === 1 || c.physOutlierClass === 1
      );
    }

    const activeProcedures = Object.entries(procedureFilters)
      .filter(([, v]) => v)
      .map(([k]) => k);
    if (activeProcedures.length > 0) {
      result = result.filter((c) =>
        activeProcedures.every((p) => c.features[p] === 1)
      );
    }

    result.sort((a, b) => {
      let cmp = 0;
      switch (sortKey) {
        case 'caseNum':
          cmp = a.caseNum - b.caseNum;
          break;
        case 'physician':
          cmp = a.physician.localeCompare(b.physician);
          break;
        case 'ptInOut':
          cmp = a.ptInOut - b.ptInOut;
          break;
        case 'outlierClass':
          cmp = a.outlierClass - b.outlierClass;
          break;
      }
      return sortDir === 'asc' ? cmp : -cmp;
    });

    return result;
  }, [cases, physicianFilter, outlierOnly, procedureFilters, sortKey, sortDir]);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortDir(key === 'ptInOut' ? 'desc' : 'asc');
    }
  };

  const SortIcon = ({ columnKey }: { columnKey: SortKey }) => {
    if (sortKey !== columnKey)
      return <ChevronDown size={14} className="text-gray-300" />;
    return sortDir === 'asc' ? (
      <ChevronUp size={14} className="text-[#1e40af]" />
    ) : (
      <ChevronDown size={14} className="text-[#1e40af]" />
    );
  };

  const getWaterfallData = (caseNum: number) => {
    const caseShap = shapByCaseNum.get(caseNum);
    if (!caseShap) return null;

    const caseInfo = cases.find((c) => c.caseNum === caseNum);
    if (!caseInfo) return null;

    const items = features.map((f, i) => ({
      feature: f,
      shapValue: caseShap[i],
      featureValue: caseInfo.features[f],
    }));

    items.sort((a, b) => Math.abs(b.shapValue) - Math.abs(a.shapValue));
    return items;
  };

  return (
    <div>
      <div className="flex items-center gap-3 mb-6">
        <AlertTriangle size={28} className="text-[#1e40af]" />
        <h1 className="text-2xl font-bold text-gray-900">Outlier Deep Dive</h1>
      </div>

      {/* Summary card */}
      <div className="bg-gradient-to-r from-red-50 to-orange-50 border border-red-100 rounded-xl p-5 mb-6">
        <div className="flex flex-wrap items-center gap-6">
          <div>
            <p className="text-sm text-gray-500">Global Outliers</p>
            <p className="text-2xl font-bold text-red-700">
              {globalOutliers.length}
            </p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Threshold (90th pctl)</p>
            <p className="text-2xl font-bold text-gray-800">
              {THRESHOLD.toFixed(1)} min
            </p>
          </div>
          {topPhys && (
            <div>
              <p className="text-sm text-gray-500">Dominant Physician</p>
              <p className="text-2xl font-bold text-orange-700">
                {topPhys[0]}: {topPhys[1]}/{globalOutliers.length}
              </p>
            </div>
          )}
          <div>
            <p className="text-sm text-gray-500">Total Cases</p>
            <p className="text-2xl font-bold text-gray-800">
              {cases.length}
            </p>
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-4 mb-4">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2 text-sm text-gray-600">
            <Filter size={16} />
            <span className="font-medium">Filters:</span>
          </div>

          <select
            value={physicianFilter}
            onChange={(e) => setPhysicianFilter(e.target.value)}
            className="border border-gray-200 rounded-lg px-3 py-1.5 text-sm bg-white"
            aria-label="Filter by physician"
          >
            <option value="all">All Physicians</option>
            {physicians.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>

          <label className="flex items-center gap-2 text-sm text-gray-600 cursor-pointer">
            <input
              type="checkbox"
              checked={outlierOnly}
              onChange={(e) => setOutlierOnly(e.target.checked)}
              className="rounded accent-[#1e40af]"
            />
            Outliers only
          </label>

          <div className="flex items-center gap-3 border-l border-gray-200 pl-4">
            {PROCEDURE_FLAGS.map((flag) => (
              <label
                key={flag}
                className="flex items-center gap-1.5 text-xs text-gray-600 cursor-pointer"
              >
                <input
                  type="checkbox"
                  checked={procedureFilters[flag]}
                  onChange={(e) =>
                    setProcedureFilters((prev) => ({
                      ...prev,
                      [flag]: e.target.checked,
                    }))
                  }
                  className="rounded accent-[#0d9488]"
                />
                {flag.replace('NOTE_', '')}
              </label>
            ))}
          </div>

          <span className="text-xs text-gray-400 ml-auto">
            {filteredCases.length} cases shown
          </span>
        </div>
      </div>

      {/* Table */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-50 border-b border-gray-200">
                {(
                  [
                    ['caseNum', 'Case #'],
                    ['physician', 'Physician'],
                    ['ptInOut', 'PT IN-OUT (min)'],
                    ['outlierClass', 'Status'],
                  ] as [SortKey, string][]
                ).map(([key, label]) => (
                  <th
                    key={key}
                    className="px-4 py-3 text-left font-medium text-gray-600 cursor-pointer hover:bg-gray-100 transition-colors"
                    onClick={() => handleSort(key)}
                    aria-label={`Sort by ${label}`}
                  >
                    <div className="flex items-center gap-1">
                      {label}
                      <SortIcon columnKey={key} />
                    </div>
                  </th>
                ))}
                {TIMING_PHASES.map((phase) => (
                  <th
                    key={phase}
                    className="px-3 py-3 text-left font-medium text-gray-500 text-xs whitespace-nowrap"
                  >
                    {phase}
                  </th>
                ))}
                <th className="px-3 py-3" />
              </tr>
            </thead>
            <tbody>
              {filteredCases.map((c) => {
                const isExpanded = expandedRow === c.caseNum;
                const isGlobalOutlier = c.outlierClass === 1;
                const isPhysOutlier = c.physOutlierClass === 1;
                const rowBg = isGlobalOutlier
                  ? 'bg-red-50/70'
                  : isPhysOutlier
                    ? 'bg-amber-50/50'
                    : '';

                return (
                  <motion.tr
                    key={c.caseNum}
                    layout
                    className={`border-b border-gray-100 hover:bg-gray-50/50 transition-colors cursor-pointer ${rowBg}`}
                    onClick={() =>
                      setExpandedRow(isExpanded ? null : c.caseNum)
                    }
                  >
                    <td className="px-4 py-3 font-mono text-gray-800">
                      {c.caseNum}
                    </td>
                    <td className="px-4 py-3 text-gray-700">{c.physician}</td>
                    <td className="px-4 py-3 font-mono font-semibold text-gray-800">
                      {c.ptInOut.toFixed(0)}
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex gap-1.5">
                        {isGlobalOutlier && (
                          <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-700">
                            Global
                          </span>
                        )}
                        {isPhysOutlier && (
                          <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-amber-100 text-amber-700">
                            Physician
                          </span>
                        )}
                        {!isGlobalOutlier && !isPhysOutlier && (
                          <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-green-50 text-green-600">
                            Normal
                          </span>
                        )}
                      </div>
                    </td>
                    {TIMING_PHASES.map((phase) => (
                      <td
                        key={phase}
                        className="px-3 py-3 font-mono text-xs text-gray-600"
                      >
                        {c.features[phase]?.toFixed(1) ?? '-'}
                      </td>
                    ))}
                    <td className="px-3 py-3">
                      {isExpanded ? (
                        <ChevronUp size={16} className="text-gray-400" />
                      ) : (
                        <ChevronDown size={16} className="text-gray-400" />
                      )}
                    </td>
                  </motion.tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Expanded SHAP waterfall */}
        <AnimatePresence>
          {expandedRow !== null && (
            <WaterfallPanel
              caseNum={expandedRow}
              getWaterfallData={getWaterfallData}
              cases={cases}
            />
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

function WaterfallPanel({
  caseNum,
  getWaterfallData,
  cases,
}: {
  caseNum: number;
  getWaterfallData: (caseNum: number) => { feature: string; shapValue: number; featureValue: number }[] | null;
  cases: Case[];
}) {
  const waterData = getWaterfallData(caseNum);
  const caseInfo = cases.find((c) => c.caseNum === caseNum);

  if (!waterData || !caseInfo) {
    return (
      <motion.div
        initial={{ opacity: 0, height: 0 }}
        animate={{ opacity: 1, height: 'auto' }}
        exit={{ opacity: 0, height: 0 }}
        className="border-t border-gray-200 p-6 bg-gray-50"
      >
        <p className="text-sm text-gray-500">
          No SHAP data available for this case.
        </p>
      </motion.div>
    );
  }

  const totalShap = waterData.reduce((s, d) => s + d.shapValue, 0);

  return (
    <motion.div
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: 'auto' }}
      exit={{ opacity: 0, height: 0 }}
      transition={{ duration: 0.25 }}
      className="overflow-hidden"
    >
      <div className="border-t border-gray-200 p-6 bg-gray-50/80">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold text-gray-800">
            Case #{caseNum} -- SHAP Contributions
          </h3>
          <div className="flex gap-4 text-xs text-gray-500">
            <span>
              PT IN-OUT:{' '}
              <span className="font-semibold text-gray-700">
                {caseInfo.ptInOut.toFixed(0)} min
              </span>
            </span>
            <span>
              Sum SHAP:{' '}
              <span
                className={`font-semibold ${
                  totalShap > 0 ? 'text-red-600' : 'text-blue-600'
                }`}
              >
                {totalShap > 0 ? '+' : ''}
                {totalShap.toFixed(3)}
              </span>
            </span>
            <span>
              Prediction:{' '}
              <span
                className={`font-semibold ${
                  caseInfo.outlierClass === 1
                    ? 'text-red-600'
                    : 'text-green-600'
                }`}
              >
                {caseInfo.outlierLabel}
              </span>
            </span>
          </div>
        </div>

        <ResponsiveContainer width="100%" height={Math.max(waterData.length * 32, 200)}>
          <BarChart
            data={waterData}
            layout="vertical"
            margin={{ left: 160, right: 80, top: 5, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" horizontal={false} />
            <XAxis
              type="number"
              tick={{ fontSize: 11 }}
              label={{
                value: 'SHAP Value',
                position: 'insideBottom',
                offset: -5,
                style: { fontSize: 11, fill: '#6b7280' },
              }}
            />
            <YAxis
              type="category"
              dataKey="feature"
              tick={{ fontSize: 11 }}
              width={150}
            />
            <ReferenceLine x={0} stroke="#9ca3af" strokeWidth={1} />
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const item = payload[0].payload as {
                  feature: string;
                  shapValue: number;
                  featureValue: number;
                };
                return (
                  <div className="bg-white border border-gray-200 rounded-lg shadow-lg p-3">
                    <p className="font-semibold text-gray-800 text-sm">
                      {item.feature}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      Feature value:{' '}
                      <span className="font-mono">{item.featureValue.toFixed(2)}</span>
                    </p>
                    <p className="text-xs mt-1">
                      SHAP:{' '}
                      <span
                        className={`font-mono font-semibold ${
                          item.shapValue > 0 ? 'text-red-600' : 'text-blue-600'
                        }`}
                      >
                        {item.shapValue > 0 ? '+' : ''}
                        {item.shapValue.toFixed(4)}
                      </span>
                    </p>
                  </div>
                );
              }}
            />
            <Bar dataKey="shapValue" radius={[0, 4, 4, 0]}>
              {waterData.map((entry) => (
                <Cell
                  key={entry.feature}
                  fill={entry.shapValue > 0 ? '#dc2626' : '#2563eb'}
                  fillOpacity={0.75}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>

        <div className="flex items-center gap-6 mt-3 text-xs text-gray-500">
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm bg-red-600 inline-block" />
            Pushes toward outlier (+)
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm bg-blue-600 inline-block" />
            Pushes toward normal (-)
          </div>
        </div>
      </div>
    </motion.div>
  );
}
