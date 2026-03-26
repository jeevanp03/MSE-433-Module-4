import { useState, useMemo } from 'react';
import { BrainCircuit, Info } from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import dashboardData from '../data/dashboard_data.json';
import type { DashboardData, FeatureImportance, FeatureStats } from '../types';

const data = dashboardData as DashboardData;

const CLINICAL_TOOLTIPS: Record<string, string> = {
  'PT PREP/INTUBATION': 'Pt-In to Access. Time from patient entering the lab to start of vascular access. Includes positioning, monitoring hookup, anesthesia induction, intubation, and sterile draping.',
  ACCESSS: 'Vascular access: femoral vein puncture and sheath insertion.',
  TSP: 'Transseptal puncture: time from starting the transseptal crossing to confirming left atrial access.',
  'PRE-MAP': 'Electroanatomic mapping of the left atrium before ablation begins.',
  'ABL DURATION': 'Abl Start to End. Total elapsed time from first to last ablation delivery, including catheter repositioning between sites.',
  'ABL TIME': 'Cumulative active "pulse on" energy delivery time. Shorter than ABL DURATION because it excludes repositioning.',
  '#ABL': 'Number of individual ablation sites targeted.',
  'POST CARE/EXTUBATION': 'Cath-Out to Pt-Out. Hemostasis, extubation, and brief post-procedure monitoring.',
  PHYSICIAN_ENC: 'Anonymized operating electrophysiologist (Dr. A, Dr. B, or Dr. C).',
  NOTE_CTI: 'Additional ablation: cavotricuspid isthmus (CTI).',
  NOTE_BOX: 'Additional ablation: box isolation (posterior wall).',
  NOTE_PST: 'Additional ablation: posterior wall (PST BOX).',
  NOTE_SVC: 'Additional ablation: superior vena cava isolation.',
};

// Build views dynamically from physicians where modelFitted === true
const fittedPhysicians = Object.entries(data.physicians)
  .filter(([, p]) => p.modelFitted)
  .map(([name]) => name);

type ViewMode = 'global' | string;

export default function ShapExplorer() {
  const [view, setView] = useState<ViewMode>('global');
  const [selectedFeature, setSelectedFeature] = useState<string | null>(null);

  const viewData = useMemo((): FeatureImportance[] => {
    if (view === 'global') {
      return [...data.globalModel.featureImportance]
        .sort((a, b) => b.shapMean - a.shapMean);
    }
    const physData = data.physicians[view];
    if (!physData?.modelFitted || !physData.topDrivers) return [];
    return Object.entries(physData.topDrivers)
      .map(([feature, shapMean]) => ({ feature, shapMean, importance: 0 }))
      .sort((a, b) => b.shapMean - a.shapMean);
  }, [view]);

  // Compute mean SHAP direction per feature from shapValues matrix
  const featureMeanShap = useMemo(() => {
    const features = data.metadata.featuresUsed;
    const means: Record<string, number> = {};
    features.forEach((f, i) => {
      const vals = data.shapValues.map((row) => row[i]);
      means[f] = vals.reduce((s, v) => s + v, 0) / vals.length;
    });
    return means;
  }, []);

  const featureStats: Record<string, FeatureStats> = data.featureStats;

  const views: ViewMode[] = ['global', ...fittedPhysicians];

  return (
    <div>
      <div className="flex items-center gap-3 mb-6">
        <BrainCircuit size={28} className="text-[#1e40af]" />
        <h1 className="text-2xl font-bold text-gray-900">SHAP Explorer</h1>
      </div>

      <p className="text-gray-500 text-sm mb-6">
        Feature importance ranked by mean |SHAP value|. Higher values indicate
        stronger influence on outlier classification. Click a bar for details.
      </p>

      {/* View toggle */}
      <div className="flex gap-2 mb-6" role="group" aria-label="Select SHAP model view">
        {views.map((v) => (
          <button
            key={v}
            onClick={() => {
              setView(v);
              setSelectedFeature(null);
            }}
            aria-pressed={view === v}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors cursor-pointer ${
              view === v
                ? 'bg-[#1e40af] text-white'
                : 'bg-white border border-gray-200 text-gray-600 hover:bg-gray-50'
            }`}
          >
            {v === 'global' ? 'Global Model' : v}
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={view}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.25 }}
        >
          {viewData.length === 0 ? (
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8">
              <div className="text-center mb-4">
                <p className="text-gray-500 font-medium">
                  No SHAP model available for {view}
                </p>
              </div>
              <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 max-w-lg mx-auto">
                <p className="text-amber-800 text-sm">
                  <strong>Why?</strong> A per-physician SHAP model requires at least 2 outlier cases to train a LightGBM classifier.
                  {view === 'Dr. C' && (
                    <span> Dr. C has only 15 total cases with 1 outlier (Case 90, flagged as TROUBLESHOOT — a technical issue, not a procedural bottleneck). With a single outlier, there is insufficient signal to build a meaningful model. Dr. C&apos;s cases are included in the global model.</span>
                  )}
                </p>
              </div>
            </div>
          ) : (
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-800 mb-4">
                {view === 'global'
                  ? 'Global Feature Importance'
                  : `${view} - Top Outlier Drivers`}
              </h2>

              <ResponsiveContainer width="100%" height={Math.max(viewData.length * 44, 200)}>
                <BarChart
                  data={viewData}
                  layout="vertical"
                  margin={{ left: 160, right: 40, top: 10, bottom: 10 }}
                >
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                  <XAxis
                    type="number"
                    tick={{ fontSize: 12 }}
                    label={{
                      value: 'Mean |SHAP Value|',
                      position: 'insideBottom',
                      offset: -5,
                      style: { fontSize: 12, fill: '#6b7280' },
                    }}
                  />
                  <YAxis
                    type="category"
                    dataKey="feature"
                    tick={{ fontSize: 12 }}
                    width={150}
                  />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (!active || !payload?.length) return null;
                      const item = payload[0].payload as FeatureImportance;
                      return (
                        <div className="bg-white border border-gray-200 rounded-lg shadow-lg p-3 max-w-xs">
                          <p className="font-semibold text-gray-800 text-sm">
                            {item.feature}
                          </p>
                          <p className="text-xs text-gray-500 mt-1">
                            {CLINICAL_TOOLTIPS[item.feature] || ''}
                          </p>
                          <p className="text-sm mt-2 text-gray-700">
                            Mean |SHAP|:{' '}
                            <span className="font-mono font-semibold">
                              {item.shapMean.toFixed(4)}
                            </span>
                          </p>
                          {featureMeanShap[item.feature] !== undefined && (
                            <p className="text-xs mt-1 text-gray-500">
                              Avg direction:{' '}
                              <span
                                className={
                                  featureMeanShap[item.feature] > 0
                                    ? 'text-red-600'
                                    : 'text-blue-600'
                                }
                              >
                                {featureMeanShap[item.feature] > 0
                                  ? 'pushes toward outlier'
                                  : 'pushes toward normal'}
                              </span>
                            </p>
                          )}
                        </div>
                      );
                    }}
                  />
                  <Bar
                    dataKey="shapMean"
                    radius={[0, 4, 4, 0]}
                    cursor="pointer"
                    onClick={(_data: unknown, index: number) => {
                      const entry = viewData[index];
                      setSelectedFeature(
                        selectedFeature === entry.feature
                          ? null
                          : entry.feature
                      );
                    }}
                  >
                    {viewData.map((entry) => {
                      const isSelected = selectedFeature === entry.feature;
                      const meanDir = featureMeanShap[entry.feature] ?? 0;
                      const baseColor =
                        meanDir > 0 ? '#dc2626' : '#2563eb';
                      return (
                        <Cell
                          key={entry.feature}
                          fill={baseColor}
                          fillOpacity={isSelected ? 1 : 0.7}
                          stroke={isSelected ? '#111827' : 'none'}
                          strokeWidth={isSelected ? 2 : 0}
                        />
                      );
                    })}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>

              <div className="flex items-center gap-6 mt-4 text-xs text-gray-500">
                <div className="flex items-center gap-1.5">
                  <span className="w-3 h-3 rounded-sm bg-red-600 inline-block" />
                  Pushes toward outlier
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-3 h-3 rounded-sm bg-blue-600 inline-block" />
                  Pushes toward normal
                </div>
              </div>
            </div>
          )}

          {/* Feature detail panel */}
          <AnimatePresence>
            {selectedFeature && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden"
              >
                <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mt-4">
                  <div className="flex items-start gap-2 mb-3">
                    <Info size={18} className="text-[#0d9488] mt-0.5" />
                    <div>
                      <h3 className="font-semibold text-gray-800">
                        {selectedFeature}
                      </h3>
                      <p className="text-sm text-gray-500 mt-0.5">
                        {CLINICAL_TOOLTIPS[selectedFeature] || 'No description available'}
                      </p>
                    </div>
                  </div>

                  {featureStats[selectedFeature] && (
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-4">
                      {[
                        { label: 'Mean', value: featureStats[selectedFeature].mean },
                        { label: 'Median', value: featureStats[selectedFeature].median },
                        { label: 'Std Dev', value: featureStats[selectedFeature].std },
                        { label: 'Range', value: `${featureStats[selectedFeature].min} - ${featureStats[selectedFeature].max}` },
                      ].map((stat) => (
                        <div
                          key={stat.label}
                          className="bg-gray-50 rounded-lg p-3 text-center"
                        >
                          <p className="text-xs text-gray-500">{stat.label}</p>
                          <p className="text-sm font-semibold text-gray-800 mt-1 font-mono">
                            {typeof stat.value === 'number'
                              ? stat.value.toFixed(2)
                              : stat.value}
                          </p>
                        </div>
                      ))}
                    </div>
                  )}

                  {featureMeanShap[selectedFeature] !== undefined && (
                    <p className="text-sm text-gray-600 mt-4">
                      Average SHAP direction:{' '}
                      <span
                        className={`font-semibold ${
                          featureMeanShap[selectedFeature] > 0
                            ? 'text-red-600'
                            : 'text-blue-600'
                        }`}
                      >
                        {featureMeanShap[selectedFeature] > 0 ? '+' : ''}
                        {featureMeanShap[selectedFeature].toFixed(4)}
                      </span>{' '}
                      ({featureMeanShap[selectedFeature] > 0
                        ? 'higher values tend to push cases toward outlier'
                        : 'higher values tend to push cases toward normal'})
                    </p>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
