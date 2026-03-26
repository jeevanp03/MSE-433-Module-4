import { useState, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import {
  SlidersHorizontal,
  Zap,
  Target,
  AlertTriangle,
  Shield,
  RotateCcw,
  Info,
} from 'lucide-react';
import whatIfData from '../data/whatif_data.json';
import dashboardData from '../data/dashboard_data.json';
import type { WhatIfData, WhatIfPreset, DashboardData, FeatureStats, ResponseSurface } from '../types';

// ---------------------------------------------------------------------------
// Constants & Data
// ---------------------------------------------------------------------------

const wiData = whatIfData as WhatIfData;
const dbData = dashboardData as DashboardData;

const ALL_FEATURES: string[] = wiData.allFeatures;
const MEDIANS: Record<string, number> = wiData.medians;
const PRESETS: Record<string, WhatIfPreset> = wiData.presets;
const RESPONSE_SURFACES: ResponseSurface[] = wiData.responseSurface;
const FEATURE_STATS: Record<string, FeatureStats> = dbData.featureStats;

// Compute base log-odds from actual outlier rate
const outlierRate = dbData.metadata.outlierCount / dbData.metadata.totalCases;
const BASE_LOG_ODDS = Math.log(outlierRate / (1 - outlierRate));

const TIMING_FEATURES = [
  'PT PREP/INTUBATION',
  'ACCESSS',
  'TSP',
  'PRE-MAP',
  'ABL DURATION',
  'ABL TIME',
  '#ABL',
  'POST CARE/EXTUBATION',
];

const BINARY_FEATURES = ['NOTE_CTI', 'NOTE_BOX', 'NOTE_PST', 'NOTE_SVC'];
const PHYSICIAN_FEATURE = 'PHYSICIAN_ENC';

const PHYSICIAN_NAMES: Record<number, string> = { 0: 'Dr A', 1: 'Dr B', 2: 'Dr C' };

const FEATURE_LABELS: Record<string, string> = {
  'PT PREP/INTUBATION': 'Prep / Intubation',
  ACCESSS: 'Access',
  TSP: 'Transseptal Puncture',
  'PRE-MAP': 'Pre-Mapping',
  'ABL DURATION': 'Ablation Duration',
  'ABL TIME': 'Ablation Time',
  '#ABL': '# Ablations',
  'POST CARE/EXTUBATION': 'Post Care / Extubation',
  PHYSICIAN_ENC: 'Physician',
  NOTE_CTI: 'CTI',
  NOTE_BOX: 'BOX',
  NOTE_PST: 'PST',
  NOTE_SVC: 'SVC',
};

const PRESET_META: { key: string; label: string; icon: React.ReactNode; color: string; bgColor: string; borderColor: string }[] = [
  { key: 'typical', label: 'Typical Case', icon: <Target size={16} />, color: 'text-emerald-700', bgColor: 'bg-emerald-50', borderColor: 'border-emerald-200' },
  { key: 'complex', label: 'Complex Case', icon: <AlertTriangle size={16} />, color: 'text-amber-700', bgColor: 'bg-amber-50', borderColor: 'border-amber-200' },
  { key: 'worstCase', label: 'Dr B Worst Case', icon: <Zap size={16} />, color: 'text-red-700', bgColor: 'bg-red-50', borderColor: 'border-red-200' },
  { key: 'optimized', label: 'Optimized Case', icon: <Shield size={16} />, color: 'text-blue-700', bgColor: 'bg-blue-50', borderColor: 'border-blue-200' },
];

// ---------------------------------------------------------------------------
// Interpolation helpers
// ---------------------------------------------------------------------------

function interpolateOnSurface(
  surface: ResponseSurface,
  value: number
): { probability: number; contributions: Record<string, number> } {
  const pts = surface.values;
  if (pts.length === 0) return { probability: 0, contributions: {} };

  if (value <= pts[0].featureValue) return { probability: pts[0].probability, contributions: { ...pts[0].shapContributions } };
  if (value >= pts[pts.length - 1].featureValue)
    return { probability: pts[pts.length - 1].probability, contributions: { ...pts[pts.length - 1].shapContributions } };

  for (let i = 0; i < pts.length - 1; i++) {
    if (value >= pts[i].featureValue && value <= pts[i + 1].featureValue) {
      const range = pts[i + 1].featureValue - pts[i].featureValue;
      const t = range === 0 ? 0 : (value - pts[i].featureValue) / range;
      const prob = pts[i].probability + t * (pts[i + 1].probability - pts[i].probability);
      const contributions: Record<string, number> = {};
      for (const key of Object.keys(pts[i].shapContributions)) {
        contributions[key] =
          pts[i].shapContributions[key] + t * (pts[i + 1].shapContributions[key] - pts[i].shapContributions[key]);
      }
      return { probability: prob, contributions };
    }
  }
  return { probability: pts[pts.length - 1].probability, contributions: { ...pts[pts.length - 1].shapContributions } };
}

function computePrediction(featureValues: Record<string, number>) {
  const surfaceMap = new Map<string, ResponseSurface>();
  for (const s of RESPONSE_SURFACES) surfaceMap.set(s.featureName, s);

  const allContributions: Record<string, number[]> = {};
  ALL_FEATURES.forEach((f) => (allContributions[f] = []));

  for (const s of RESPONSE_SURFACES) {
    const val = featureValues[s.featureName] ?? MEDIANS[s.featureName] ?? 0;
    const result = interpolateOnSurface(s, val);

    for (const [feat, contrib] of Object.entries(result.contributions)) {
      allContributions[feat]?.push(contrib);
    }
  }

  const shapContributions: Record<string, number> = {};
  for (const feat of ALL_FEATURES) {
    if (surfaceMap.has(feat)) {
      const val = featureValues[feat] ?? MEDIANS[feat] ?? 0;
      const result = interpolateOnSurface(surfaceMap.get(feat)!, val);
      shapContributions[feat] = result.contributions[feat] ?? 0;
    } else {
      const vals = allContributions[feat];
      shapContributions[feat] = vals && vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
    }
  }

  const totalShap = Object.values(shapContributions).reduce((a, b) => a + b, 0);
  const logOdds = BASE_LOG_ODDS + totalShap;
  const probability = 1 / (1 + Math.exp(-logOdds));

  return { probability: Math.max(0, Math.min(1, probability)), shapContributions };
}

// ---------------------------------------------------------------------------
// Gauge Component (SVG)
// ---------------------------------------------------------------------------

function OutlierGauge({ probability, isAnimating }: { probability: number; isAnimating?: boolean }) {
  const pct = probability * 100;
  const radius = 90;
  const strokeWidth = 18;
  const circumference = Math.PI * radius;
  const dashOffset = circumference * (1 - probability);

  const getColor = (p: number) => {
    if (p < 0.3) return '#10b981';
    if (p < 0.6) return '#f59e0b';
    return '#ef4444';
  };

  const color = getColor(probability);
  const bgColor =
    probability < 0.3
      ? 'from-emerald-50 to-white'
      : probability < 0.6
        ? 'from-amber-50 to-white'
        : 'from-red-50 to-white';

  const riskLabel = probability < 0.3 ? 'Low Risk' : probability < 0.6 ? 'Medium Risk' : 'High Risk';

  return (
    <div className={`relative flex flex-col items-center bg-gradient-to-b ${bgColor} rounded-2xl p-6`}>
      <p className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-2">Outlier Probability</p>
      <svg width="220" height="130" viewBox="0 0 220 130" role="img" aria-label={`Outlier probability gauge showing ${pct.toFixed(1)}%, ${riskLabel}`}>
        <defs>
          <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#10b981" />
            <stop offset="50%" stopColor="#f59e0b" />
            <stop offset="100%" stopColor="#ef4444" />
          </linearGradient>
          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Background arc */}
        <path
          d={`M ${110 - radius} 115 A ${radius} ${radius} 0 0 1 ${110 + radius} 115`}
          fill="none"
          stroke="#e5e7eb"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
        />

        {/* Filled arc */}
        <motion.path
          d={`M ${110 - radius} 115 A ${radius} ${radius} 0 0 1 ${110 + radius} 115`}
          fill="none"
          stroke="url(#gaugeGrad)"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={false}
          animate={{ strokeDashoffset: dashOffset }}
          transition={{ type: 'spring', stiffness: 60, damping: 15 }}
          filter="url(#glow)"
        />

        {/* Needle */}
        <motion.g
          initial={false}
          animate={{ rotate: -180 + probability * 180 }}
          transition={{ type: 'spring', stiffness: 60, damping: 15 }}
          style={{ transformOrigin: '110px 115px' }}
        >
          <line x1="110" y1="115" x2="110" y2={115 - radius + strokeWidth + 4} stroke={color} strokeWidth="3" strokeLinecap="round" />
          <circle cx="110" cy="115" r="6" fill={color} />
        </motion.g>

        {/* Scale labels */}
        <text x="15" y="128" textAnchor="middle" className="text-[10px]" fill="#9ca3af">
          0%
        </text>
        <text x="110" y="18" textAnchor="middle" className="text-[10px]" fill="#9ca3af">
          50%
        </text>
        <text x="205" y="128" textAnchor="middle" className="text-[10px]" fill="#9ca3af">
          100%
        </text>
      </svg>

      <motion.div
        className="absolute bottom-14 text-center"
        initial={false}
        animate={{ scale: isAnimating ? [1, 1.1, 1] : 1 }}
        transition={{ duration: 0.3 }}
      >
        <motion.span
          className="text-4xl font-bold tabular-nums"
          style={{ color }}
          initial={false}
          animate={{ color }}
          transition={{ duration: 0.3 }}
        >
          {pct.toFixed(1)}%
        </motion.span>
      </motion.div>

      <motion.span
        className="text-sm font-semibold mt-1 px-3 py-0.5 rounded-full"
        style={{
          color,
          backgroundColor: probability < 0.3 ? '#ecfdf5' : probability < 0.6 ? '#fffbeb' : '#fef2f2',
        }}
        initial={false}
        animate={{ color }}
      >
        {riskLabel}
      </motion.span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Feature Slider Component
// ---------------------------------------------------------------------------

function FeatureSlider({
  feature,
  value,
  onChange,
  stats,
}: {
  feature: string;
  value: number;
  onChange: (v: number) => void;
  stats: FeatureStats;
}) {
  const label = FEATURE_LABELS[feature] || feature;
  const pct = stats.max - stats.min > 0 ? ((value - stats.min) / (stats.max - stats.min)) * 100 : 50;
  const medianPct = stats.max - stats.min > 0 ? ((stats.median - stats.min) / (stats.max - stats.min)) * 100 : 50;
  const isAboveMedian = value > stats.median * 1.1;

  return (
    <div className="group">
      <div className="flex items-center justify-between mb-1.5">
        <label className="text-xs font-medium text-gray-600 group-hover:text-gray-900 transition-colors">{label}</label>
        <motion.span
          className={`text-xs font-bold tabular-nums px-1.5 py-0.5 rounded ${
            isAboveMedian ? 'bg-amber-100 text-amber-700' : 'bg-gray-100 text-gray-600'
          }`}
          key={value}
          initial={{ scale: 1.2 }}
          animate={{ scale: 1 }}
          transition={{ duration: 0.15 }}
        >
          {Number.isInteger(value) ? value : value.toFixed(1)}
        </motion.span>
      </div>
      <div className="relative">
        <input
          type="range"
          min={stats.min}
          max={stats.max}
          step={stats.max - stats.min > 50 ? 1 : 0.5}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          aria-label={`${label} slider`}
          className="w-full h-2 rounded-full appearance-none cursor-pointer
            [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4
            [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-[#1e40af] [&::-webkit-slider-thumb]:shadow-md
            [&::-webkit-slider-thumb]:hover:bg-[#1e3a8a] [&::-webkit-slider-thumb]:transition-colors
            [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-white"
          style={{
            background: `linear-gradient(to right, #1e40af ${pct}%, #e5e7eb ${pct}%)`,
          }}
        />
        {/* Median marker */}
        <div
          className="absolute top-1/2 -translate-y-1/2 w-0.5 h-4 bg-teal-500 pointer-events-none opacity-60"
          style={{ left: `${medianPct}%` }}
          title={`Median: ${stats.median}`}
        />
      </div>
      <div className="flex justify-between text-[10px] text-gray-400 mt-0.5">
        <span>{stats.min}</span>
        <span className="text-teal-600">med: {stats.median}</span>
        <span>{stats.max}</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// SHAP Contribution Bar Chart
// ---------------------------------------------------------------------------

function ShapBarChart({ contributions }: { contributions: Record<string, number> }) {
  const chartData = Object.entries(contributions)
    .map(([feature, value]) => ({
      feature: FEATURE_LABELS[feature] || feature,
      value: parseFloat(value.toFixed(4)),
      rawFeature: feature,
    }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 10);

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4">
      <div className="flex items-center gap-2 mb-3">
        <Info size={14} className="text-gray-400" />
        <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Feature Contributions (SHAP)</h3>
      </div>
      <div className="flex items-center gap-4 mb-2 text-[10px] text-gray-400">
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-2 rounded-sm bg-[#1e40af]" /> Reduces risk
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-2 rounded-sm bg-[#ef4444]" /> Increases risk
        </span>
      </div>
      <ResponsiveContainer width="100%" height={Math.max(200, chartData.length * 32)}>
        <BarChart data={chartData} layout="vertical" margin={{ top: 0, right: 20, bottom: 0, left: 110 }}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f3f4f6" />
          <XAxis type="number" tick={{ fontSize: 10, fill: '#9ca3af' }} axisLine={false} tickLine={false} />
          <YAxis
            dataKey="feature"
            type="category"
            tick={{ fontSize: 11, fill: '#4b5563' }}
            axisLine={false}
            tickLine={false}
            width={105}
          />
          <Tooltip
            contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #e5e7eb' }}
            formatter={(val) => [Number(val).toFixed(4), 'SHAP']}
          />
          <ReferenceLine x={0} stroke="#d1d5db" />
          <Bar dataKey="value" radius={[4, 4, 4, 4]} maxBarSize={20}>
            {chartData.map((entry, idx) => (
              <Cell key={idx} fill={entry.value >= 0 ? '#ef4444' : '#1e40af'} fillOpacity={0.85} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function WhatIfSimulator() {
  const [featureValues, setFeatureValues] = useState<Record<string, number>>(() => ({ ...MEDIANS }));
  const [isAnimating, setIsAnimating] = useState(false);
  const [activePreset, setActivePreset] = useState<string | null>('typical');

  const updateFeature = useCallback((feature: string, value: number) => {
    setFeatureValues((prev) => ({ ...prev, [feature]: value }));
    setActivePreset(null);
    setIsAnimating(true);
    setTimeout(() => setIsAnimating(false), 300);
  }, []);

  const loadPreset = useCallback((key: string) => {
    const preset = PRESETS[key];
    if (!preset) return;
    setFeatureValues({ ...preset.featureValues });
    setActivePreset(key);
    setIsAnimating(true);
    setTimeout(() => setIsAnimating(false), 500);
  }, []);

  const resetToMedian = useCallback(() => {
    setFeatureValues({ ...MEDIANS });
    setActivePreset('typical');
    setIsAnimating(true);
    setTimeout(() => setIsAnimating(false), 300);
  }, []);

  const prediction = useMemo(() => computePrediction(featureValues), [featureValues]);

  const presetProbabilities = useMemo(() => {
    const probs: Record<string, number> = {};
    for (const [key, preset] of Object.entries(PRESETS)) {
      probs[key] = preset.outlierProbability;
    }
    return probs;
  }, []);

  const physician = Math.round(featureValues[PHYSICIAN_FEATURE] ?? 1);

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-[#1e40af]/10 rounded-lg">
            <SlidersHorizontal size={24} className="text-[#1e40af]" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">What-If Simulator</h1>
            <p className="text-sm text-gray-500">Adjust parameters to explore outlier risk factors in real-time</p>
          </div>
        </div>
        <button
          onClick={resetToMedian}
          className="flex items-center gap-2 px-3 py-2 text-sm text-gray-600 bg-white border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors cursor-pointer"
          aria-label="Reset all features to median values"
        >
          <RotateCcw size={14} />
          Reset
        </button>
      </div>

      {/* Preset Scenario Buttons */}
      <div className="grid grid-cols-4 gap-3 mb-6">
        {PRESET_META.map((pm) => (
          <motion.button
            key={pm.key}
            onClick={() => loadPreset(pm.key)}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            aria-label={`Load ${pm.label} preset`}
            aria-pressed={activePreset === pm.key}
            className={`relative flex flex-col items-center gap-1.5 px-3 py-3 rounded-xl border-2 transition-all cursor-pointer ${
              activePreset === pm.key
                ? `${pm.bgColor} ${pm.borderColor} shadow-md`
                : 'bg-white border-gray-200 hover:border-gray-300'
            }`}
          >
            <span className={`${activePreset === pm.key ? pm.color : 'text-gray-400'}`}>{pm.icon}</span>
            <span className={`text-xs font-semibold ${activePreset === pm.key ? pm.color : 'text-gray-600'}`}>
              {pm.label}
            </span>
            <span
              className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                presetProbabilities[pm.key] < 0.3
                  ? 'bg-emerald-100 text-emerald-700'
                  : presetProbabilities[pm.key] < 0.6
                    ? 'bg-amber-100 text-amber-700'
                    : 'bg-red-100 text-red-700'
              }`}
            >
              {(presetProbabilities[pm.key] * 100).toFixed(1)}%
            </span>
          </motion.button>
        ))}
      </div>

      {/* Main Layout: Sliders (left) | Gauge + Chart (right) */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Left Panel - Sliders */}
        <div className="lg:col-span-3 space-y-5">
          {/* Physician Selector */}
          <motion.div
            className="bg-white rounded-xl border border-gray-200 p-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.05 }}
          >
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Physician & Scheduling</h3>
            <div className="space-y-4">
              <div>
                <label className="text-xs font-medium text-gray-600 mb-2 block">Physician</label>
                <div className="flex gap-2" role="group" aria-label="Select physician">
                  {[0, 1, 2].map((id) => (
                    <motion.button
                      key={id}
                      onClick={() => updateFeature(PHYSICIAN_FEATURE, id)}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      aria-pressed={physician === id}
                      className={`flex-1 py-2.5 px-3 rounded-lg text-sm font-semibold transition-all cursor-pointer ${
                        physician === id
                          ? 'bg-[#1e40af] text-white shadow-lg shadow-blue-500/25'
                          : 'bg-gray-100 text-gray-500 hover:bg-gray-200'
                      }`}
                    >
                      {PHYSICIAN_NAMES[id]}
                    </motion.button>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>

          {/* Timing Phases */}
          <motion.div
            className="bg-white rounded-xl border border-gray-200 p-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Timing Phases (minutes)</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-4">
              {TIMING_FEATURES.filter(feat => FEATURE_STATS[feat]).map((feat) => (
                <FeatureSlider
                  key={feat}
                  feature={feat}
                  value={featureValues[feat] ?? MEDIANS[feat] ?? 0}
                  onChange={(v) => updateFeature(feat, v)}
                  stats={FEATURE_STATS[feat]}
                />
              ))}
            </div>
          </motion.div>

          {/* Procedure Type Checkboxes */}
          <motion.div
            className="bg-white rounded-xl border border-gray-200 p-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 }}
          >
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Procedure Type</h3>
            <div className="flex flex-wrap gap-3" role="group" aria-label="Procedure type toggles">
              {BINARY_FEATURES.map((feat) => {
                const isOn = (featureValues[feat] ?? 0) >= 0.5;
                const featLabel = FEATURE_LABELS[feat] || feat;
                return (
                  <motion.button
                    key={feat}
                    onClick={() => updateFeature(feat, isOn ? 0 : 1)}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    aria-pressed={isOn}
                    aria-label={`Toggle ${featLabel} procedure`}
                    className={`px-4 py-2 rounded-lg text-sm font-semibold border-2 transition-all cursor-pointer ${
                      isOn
                        ? 'bg-teal-50 border-teal-300 text-teal-700'
                        : 'bg-white border-gray-200 text-gray-400 hover:border-gray-300'
                    }`}
                  >
                    <span className="mr-1.5">{isOn ? '\u2713' : '\u25CB'}</span>
                    {featLabel}
                  </motion.button>
                );
              })}
            </div>
          </motion.div>
        </div>

        {/* Right Panel - Gauge + SHAP Chart */}
        <div className="lg:col-span-2 space-y-5">
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
          >
            <AnimatePresence mode="wait">
              <motion.div
                key="gauge"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.3 }}
              >
                <OutlierGauge probability={prediction.probability} isAnimating={isAnimating} />
              </motion.div>
            </AnimatePresence>
          </motion.div>

          {/* Risk Summary Card */}
          <motion.div
            className="bg-white rounded-xl border border-gray-200 p-4"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.15 }}
          >
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Configuration Summary</h3>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="bg-gray-50 rounded-lg p-2">
                <span className="text-gray-400 block">Physician</span>
                <span className="font-bold text-gray-700">{PHYSICIAN_NAMES[physician] ?? `ID ${physician}`}</span>
              </div>
              <div className="bg-gray-50 rounded-lg p-2">
                <span className="text-gray-400 block">Procedures</span>
                <span className="font-bold text-gray-700">
                  {BINARY_FEATURES.filter((f) => (featureValues[f] ?? 0) >= 0.5)
                    .map((f) => FEATURE_LABELS[f])
                    .join(', ') || 'Standard'}
                </span>
              </div>
              <div className="bg-gray-50 rounded-lg p-2">
                <span className="text-gray-400 block">Total SHAP</span>
                <span
                  className={`font-bold ${
                    Object.values(prediction.shapContributions).reduce((a, b) => a + b, 0) > 0
                      ? 'text-red-600'
                      : 'text-emerald-600'
                  }`}
                >
                  {Object.values(prediction.shapContributions)
                    .reduce((a, b) => a + b, 0)
                    .toFixed(3)}
                </span>
              </div>
            </div>
          </motion.div>

          {/* SHAP Chart */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
          >
            <ShapBarChart contributions={prediction.shapContributions} />
          </motion.div>
        </div>
      </div>
    </div>
  );
}
