import { useState, useMemo, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
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
import {
  ArrowRightLeft,
  Users,
  TrendingDown,
  Shuffle,
  Search,
  ChevronDown,
  ChevronUp,
  AlertTriangle,
  Award,
  Info,
  Check,
  X,
  ArrowRight,
  UserCheck,
  Trash2,
} from 'lucide-react';
import reassignmentRaw from '../data/reassignment_data.json';
import dashboardData from '../data/dashboard_data.json';
import type { DashboardData } from '../types';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Prediction {
  probability: number;
  shapTop5: Record<string, number>;
}

interface CaseReassignment {
  caseNum: number;
  originalPhysician: string;
  originalPhysicianEnc: number;
  ptInOut: number;
  outlierClass: number;
  features: Record<string, number>;
  predictions: Record<string, Prediction>;
  optimalPhysician: number;
  optimalProbability: number;
}

interface BatchCase {
  caseNum: number;
  beforeProb: number;
  afterProb: number;
  resolved: boolean;
}

interface BatchScenario {
  sourcePhysician: string;
  targetPhysician: string;
  affectedCases: number;
  outliersResolved: number;
  outliersRemaining: number;
  cases: BatchCase[];
}

interface OptimalAssignment {
  currentOutliers: number;
  optimizedOutliers: number;
  reductionPercent: number;
  changes: {
    caseNum: number;
    fromPhysician: string;
    toPhysician: string;
    beforeProb: number;
    afterProb: number;
  }[];
}

interface ReassignmentData {
  caseReassignments: CaseReassignment[];
  batchScenarios: BatchScenario[];
  optimalAssignment: OptimalAssignment;
  physicianWorkload: {
    current: Record<string, number>;
    optimal: Record<string, number>;
  };
}

// ---------------------------------------------------------------------------
// Data & Constants
// ---------------------------------------------------------------------------

const rData = reassignmentRaw as unknown as ReassignmentData;
const dbData = dashboardData as DashboardData;

const PHYSICIAN_COLORS: Record<string, string> = {
  'Dr. A': '#3b82f6',
  'Dr. B': '#ef4444',
  'Dr. C': '#22c55e',
};

const PHYSICIAN_ENC_MAP: Record<number, string> = { 0: 'Dr. A', 1: 'Dr. B', 2: 'Dr. C' };
const PHYSICIAN_IDS = [0, 1, 2] as const;

const TIMING_PHASES = [
  'PT PREP/INTUBATION',
  'ACCESSS',
  'TSP',
  'PRE-MAP',
  'ABL DURATION',
  'ABL TIME',
  'POST CARE/EXTUBATION',
] as const;

const TIMING_LABELS: Record<string, string> = {
  'PT PREP/INTUBATION': 'Prep',
  ACCESSS: 'Access',
  TSP: 'TSP',
  'PRE-MAP': 'Pre-Map',
  'ABL DURATION': 'Abl Dur',
  'ABL TIME': 'Abl Time',
  'POST CARE/EXTUBATION': 'Post Care',
};

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
  CASE_ORDER_IN_DAY: 'Case Order in Day',
};

const THRESHOLD = dbData.metadata.threshold;

// ---------------------------------------------------------------------------
// Animated Number
// ---------------------------------------------------------------------------

function AnimatedNumber({ value, suffix = '', decimals = 0 }: { value: number; suffix?: string; decimals?: number }) {
  const [display, setDisplay] = useState(0);
  useEffect(() => {
    const duration = 1200;
    const start = performance.now();
    let rafId: number;
    const animate = (now: number) => {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplay(value * eased);
      if (progress < 1) rafId = requestAnimationFrame(animate);
    };
    rafId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafId);
  }, [value]);
  return (
    <span>
      {display.toFixed(decimals)}
      {suffix}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Mini Gauge (adapted from WhatIfSimulator)
// ---------------------------------------------------------------------------

function MiniGauge({ probability, size = 120 }: { probability: number; size?: number }) {
  const pct = probability * 100;
  const radius = size * 0.41;
  const strokeWidth = size * 0.09;
  const cx = size / 2;
  const cy = size * 0.58;
  const circumference = Math.PI * radius;
  const dashOffset = circumference * (1 - probability);

  const getColor = (p: number) => {
    if (p < 0.3) return '#10b981';
    if (p < 0.6) return '#f59e0b';
    return '#ef4444';
  };

  const color = getColor(probability);

  return (
    <div className="flex flex-col items-center">
      <svg width={size} height={size * 0.6} viewBox={`0 0 ${size} ${size * 0.6}`}>
        <defs>
          <linearGradient id={`gauge-${size}-${Math.round(probability * 1000)}`} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#10b981" />
            <stop offset="50%" stopColor="#f59e0b" />
            <stop offset="100%" stopColor="#ef4444" />
          </linearGradient>
        </defs>
        <path
          d={`M ${cx - radius} ${cy} A ${radius} ${radius} 0 0 1 ${cx + radius} ${cy}`}
          fill="none"
          stroke="#e5e7eb"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
        />
        <motion.path
          d={`M ${cx - radius} ${cy} A ${radius} ${radius} 0 0 1 ${cx + radius} ${cy}`}
          fill="none"
          stroke={`url(#gauge-${size}-${Math.round(probability * 1000)})`}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={false}
          animate={{ strokeDashoffset: dashOffset }}
          transition={{ type: 'spring', stiffness: 60, damping: 15 }}
        />
        <motion.g
          initial={false}
          animate={{ rotate: -180 + probability * 180 }}
          transition={{ type: 'spring', stiffness: 60, damping: 15 }}
          style={{ transformOrigin: `${cx}px ${cy}px` }}
        >
          <line
            x1={cx}
            y1={cy}
            x2={cx}
            y2={cy - radius + strokeWidth + 2}
            stroke={color}
            strokeWidth="2"
            strokeLinecap="round"
          />
          <circle cx={cx} cy={cy} r={3} fill={color} />
        </motion.g>
      </svg>
      <span className="text-lg font-bold tabular-nums" style={{ color }}>
        {pct.toFixed(1)}%
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Impact Dashboard Cards
// ---------------------------------------------------------------------------

function ImpactDashboard({ manualReassignments }: { manualReassignments: Map<number, number> }) {
  const { optimalAssignment } = rData;

  // Compute manual reassignment stats
  const manualStats = useMemo(() => {
    let resolved = 0;
    manualReassignments.forEach((newPhysId, caseNum) => {
      const c = rData.caseReassignments.find((cr) => cr.caseNum === caseNum);
      if (!c || c.outlierClass !== 1) return;
      const newProb = c.predictions[String(newPhysId)]?.probability ?? 1;
      if (newProb < 0.5) resolved++;
    });
    return { total: manualReassignments.size, resolved };
  }, [manualReassignments]);

  // Find most impactful single reassignment
  const bestSingle = useMemo(() => {
    let best = { caseNum: 0, from: '', to: '', reduction: 0 };
    for (const c of rData.caseReassignments) {
      if (c.outlierClass !== 1) continue;
      const origProb = c.predictions[String(c.originalPhysicianEnc)]?.probability ?? 1;
      for (const id of PHYSICIAN_IDS) {
        if (id === c.originalPhysicianEnc) continue;
        const newProb = c.predictions[String(id)]?.probability ?? 1;
        const reduction = origProb - newProb;
        if (reduction > best.reduction) {
          best = {
            caseNum: c.caseNum,
            from: c.originalPhysician,
            to: PHYSICIAN_ENC_MAP[id],
            reduction,
          };
        }
      }
    }
    return best;
  }, []);

  const cards = [
    {
      title: 'Current Outliers',
      value: optimalAssignment.currentOutliers,
      suffix: '',
      color: 'bg-red-50 border-red-200',
      iconBg: 'bg-red-100',
      iconColor: 'text-red-600',
      icon: <AlertTriangle size={20} />,
      sub: `above ${THRESHOLD.toFixed(0)} min threshold`,
    },
    {
      title: 'Optimized Outliers',
      value: optimalAssignment.optimizedOutliers,
      suffix: '',
      color: 'bg-emerald-50 border-emerald-200',
      iconBg: 'bg-emerald-100',
      iconColor: 'text-emerald-600',
      icon: <TrendingDown size={20} />,
      sub: 'with optimal assignments',
    },
    {
      title: 'Potential Reduction',
      value: optimalAssignment.reductionPercent,
      suffix: '%',
      color: 'bg-blue-50 border-blue-200',
      iconBg: 'bg-blue-100',
      iconColor: 'text-blue-600',
      icon: <ArrowRightLeft size={20} />,
      sub: 'outliers eliminated',
    },
    {
      title: 'Best Single Reassignment',
      value: (bestSingle.reduction * 100),
      suffix: '% drop',
      color: 'bg-amber-50 border-amber-200',
      iconBg: 'bg-amber-100',
      iconColor: 'text-amber-600',
      icon: <Award size={20} />,
      sub: `Case ${bestSingle.caseNum}: ${bestSingle.from} → ${bestSingle.to}`,
      decimals: 1,
    },
    ...(manualStats.total > 0
      ? [
          {
            title: 'Your Reassignments',
            value: manualStats.total,
            suffix: '',
            color: 'bg-purple-50 border-purple-200',
            iconBg: 'bg-purple-100',
            iconColor: 'text-purple-600',
            icon: <UserCheck size={20} />,
            sub: `${manualStats.resolved} outlier${manualStats.resolved !== 1 ? 's' : ''} resolved`,
            decimals: 0,
          },
        ]
      : []),
  ];

  return (
    <div className={`grid grid-cols-1 sm:grid-cols-2 ${cards.length > 4 ? 'lg:grid-cols-5' : 'lg:grid-cols-4'} gap-4 mb-6`}>
      {cards.map((card, i) => (
        <motion.div
          key={card.title}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.08 }}
          className={`rounded-xl border p-5 ${card.color}`}
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-gray-500">{card.title}</span>
            <div className={`p-2 rounded-lg ${card.iconBg} ${card.iconColor}`}>{card.icon}</div>
          </div>
          <div className="text-3xl font-bold text-gray-900">
            <AnimatedNumber value={card.value} suffix={card.suffix} decimals={card.decimals ?? 0} />
          </div>
          <p className="text-xs text-gray-400 mt-1">{card.sub}</p>
        </motion.div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Case Selector + Reassignment Detail
// ---------------------------------------------------------------------------

type CaseFilter = 'all' | 'outliers' | 'Dr. A' | 'Dr. B' | 'Dr. C';

function CaseReassignmentSection({
  manualReassignments,
  onReassign,
  onUndoReassign,
}: {
  manualReassignments: Map<number, number>;
  onReassign: (caseNum: number, physId: number) => void;
  onUndoReassign: (caseNum: number) => void;
}) {
  const [selectedCaseNum, setSelectedCaseNum] = useState<number | null>(null);
  const [filter, setFilter] = useState<CaseFilter>('outliers');
  const [search, setSearch] = useState('');

  const filteredCases = useMemo(() => {
    let cases = rData.caseReassignments;
    if (filter === 'outliers') cases = cases.filter((c) => c.outlierClass === 1);
    else if (filter === 'Dr. A') cases = cases.filter((c) => c.originalPhysician === 'Dr. A');
    else if (filter === 'Dr. B') cases = cases.filter((c) => c.originalPhysician === 'Dr. B');
    else if (filter === 'Dr. C') cases = cases.filter((c) => c.originalPhysician === 'Dr. C');
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      cases = cases.filter(
        (c) =>
          String(c.caseNum).includes(q) ||
          c.originalPhysician.toLowerCase().includes(q)
      );
    }
    return cases.sort((a, b) => b.ptInOut - a.ptInOut);
  }, [filter, search]);

  const selectedCase = useMemo(
    () => (selectedCaseNum != null ? rData.caseReassignments.find((c) => c.caseNum === selectedCaseNum) : null),
    [selectedCaseNum]
  );

  const originalProb = selectedCase
    ? selectedCase.predictions[String(selectedCase.originalPhysicianEnc)]?.probability ?? 0
    : 0;

  // Timing bar chart data for selected case
  const timingData = useMemo(() => {
    if (!selectedCase) return [];
    return TIMING_PHASES.map((phase) => ({
      name: TIMING_LABELS[phase] || phase,
      value: selectedCase.features[phase] ?? 0,
    }));
  }, [selectedCase]);

  const filters: { key: CaseFilter; label: string }[] = [
    { key: 'all', label: 'All' },
    { key: 'outliers', label: 'Outliers Only' },
    { key: 'Dr. A', label: 'Dr. A' },
    { key: 'Dr. B', label: 'Dr. B' },
    { key: 'Dr. C', label: 'Dr. C' },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 }}
      className="bg-white rounded-xl border border-gray-200 p-6 mb-6"
    >
      <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
        <Users size={18} className="text-gray-500" />
        Case Selector & Reassignment
      </h2>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Left: Case list */}
        <div className="lg:col-span-2">
          {/* Search */}
          <div className="relative mb-3">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
            <input
              type="text"
              placeholder="Search case # or physician..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full pl-8 pr-3 py-2 text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* Filter buttons */}
          <div className="flex flex-wrap gap-1.5 mb-3">
            {filters.map((f) => (
              <button
                key={f.key}
                onClick={() => setFilter(f.key)}
                className={`px-2.5 py-1 text-xs font-medium rounded-lg transition-colors cursor-pointer ${
                  filter === f.key
                    ? 'bg-[#1e40af] text-white'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>

          {/* Case list */}
          <div className="max-h-[420px] overflow-y-auto border border-gray-100 rounded-lg divide-y divide-gray-100">
            {filteredCases.map((c) => {
              const isSelected = selectedCaseNum === c.caseNum;
              const reassignedTo = manualReassignments.get(c.caseNum);
              const isReassigned = reassignedTo !== undefined;
              const reassignedPhysName = isReassigned ? PHYSICIAN_ENC_MAP[reassignedTo] : '';
              const reassignedProb = isReassigned
                ? c.predictions[String(reassignedTo)]?.probability ?? 1
                : 1;
              const outlierResolved = isReassigned && c.outlierClass === 1 && reassignedProb < 0.5;
              return (
                <button
                  key={c.caseNum}
                  onClick={() => setSelectedCaseNum(c.caseNum)}
                  className={`w-full text-left px-3 py-2.5 flex items-center gap-2 transition-colors cursor-pointer ${
                    isSelected ? 'bg-blue-50 border-l-3 border-l-blue-500' : 'hover:bg-gray-50'
                  }`}
                >
                  <span className="font-mono text-sm font-semibold text-gray-800 w-12">#{c.caseNum}</span>
                  <span
                    className="text-xs font-semibold px-1.5 py-0.5 rounded"
                    style={{ color: PHYSICIAN_COLORS[c.originalPhysician], backgroundColor: `${PHYSICIAN_COLORS[c.originalPhysician]}15` }}
                  >
                    {c.originalPhysician}
                  </span>
                  {isReassigned && (
                    <span className="flex items-center gap-0.5 text-[10px] font-semibold text-purple-700 bg-purple-50 px-1.5 py-0.5 rounded">
                      <ArrowRight size={10} />
                      {reassignedPhysName}
                    </span>
                  )}
                  <span className="text-xs text-gray-500 ml-auto">{c.ptInOut.toFixed(0)} min</span>
                  {c.outlierClass === 1 && (
                    outlierResolved ? (
                      <span className="text-[10px] font-bold px-1.5 py-0.5 rounded-full bg-emerald-100 text-emerald-700 flex items-center gap-0.5">
                        <Check size={10} />
                        RESOLVED
                      </span>
                    ) : (
                      <span className="text-[10px] font-bold px-1.5 py-0.5 rounded-full bg-red-100 text-red-700">
                        OUTLIER
                      </span>
                    )
                  )}
                </button>
              );
            })}
            {filteredCases.length === 0 && (
              <div className="p-6 text-center text-sm text-gray-400">No cases match your filter.</div>
            )}
          </div>
        </div>

        {/* Right: Case detail + physician cards */}
        <div className="lg:col-span-3">
          {!selectedCase ? (
            <div className="flex items-center justify-center h-full text-gray-400 text-sm">
              <div className="text-center">
                <ArrowRightLeft size={32} className="mx-auto mb-2 text-gray-300" />
                <p>Select a case from the list to see reassignment options</p>
              </div>
            </div>
          ) : (
            <AnimatePresence mode="wait">
              <motion.div
                key={selectedCase.caseNum}
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                transition={{ duration: 0.2 }}
              >
                {/* Case detail header */}
                <div className="bg-gray-50 rounded-lg p-4 mb-4">
                  <div className="flex items-center gap-2 mb-3">
                    <span className="text-sm font-bold text-gray-800">Case #{selectedCase.caseNum}</span>
                    <span
                      className="text-[10px] font-bold px-2 py-0.5 rounded-full"
                      style={{
                        backgroundColor: selectedCase.outlierClass === 1 ? '#fef2f2' : '#f0fdf4',
                        color: selectedCase.outlierClass === 1 ? '#dc2626' : '#16a34a',
                      }}
                    >
                      {selectedCase.outlierClass === 1 ? 'OUTLIER' : 'NORMAL'}
                    </span>
                    <span className="text-xs text-gray-500">
                      {selectedCase.originalPhysician} -- {selectedCase.ptInOut.toFixed(0)} min
                    </span>
                  </div>

                  {/* Timing stacked horizontal bar */}
                  <ResponsiveContainer width="100%" height={40}>
                    <BarChart data={[{ name: 'case', ...Object.fromEntries(timingData.map((t) => [t.name, t.value])) }]} layout="horizontal" barSize={24}>
                      <XAxis type="number" hide />
                      <YAxis type="category" dataKey="name" hide />
                      <Tooltip formatter={(val) => [`${Number(val).toFixed(1)} min`]} />
                      {timingData.map((t, i) => {
                        const colors = ['#3b82f6', '#8b5cf6', '#06b6d4', '#f59e0b', '#ef4444', '#ec4899', '#10b981'];
                        return <Bar key={t.name} dataKey={t.name} stackId="timing" fill={colors[i % colors.length]} />;
                      })}
                    </BarChart>
                  </ResponsiveContainer>
                  <div className="flex flex-wrap gap-2 mt-2">
                    {timingData.map((t, i) => {
                      const colors = ['#3b82f6', '#8b5cf6', '#06b6d4', '#f59e0b', '#ef4444', '#ec4899', '#10b981'];
                      return (
                        <span key={t.name} className="flex items-center gap-1 text-[10px] text-gray-500">
                          <span className="w-2 h-2 rounded-sm inline-block" style={{ backgroundColor: colors[i % colors.length] }} />
                          {t.name}: {t.value.toFixed(1)}m
                        </span>
                      );
                    })}
                  </div>
                </div>

                {/* Physician reassignment cards */}
                <div className="grid grid-cols-3 gap-3">
                  {PHYSICIAN_IDS.map((id) => {
                    const name = PHYSICIAN_ENC_MAP[id];
                    const pred = selectedCase.predictions[String(id)];
                    if (!pred) return null;
                    const isOriginal = id === selectedCase.originalPhysicianEnc;
                    const delta = pred.probability - originalProb;
                    const shapEntries = Object.entries(pred.shapTop5)
                      .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
                      .slice(0, 3);

                    const currentReassignment = manualReassignments.get(selectedCase.caseNum);
                    const isReassignedHere = currentReassignment === id;
                    const hasBeenReassigned = currentReassignment !== undefined;
                    const isOriginalDimmed = isOriginal && hasBeenReassigned;

                    return (
                      <motion.div
                        key={id}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: id * 0.05 }}
                        className={`relative rounded-xl border-2 p-4 transition-all ${
                          isReassignedHere
                            ? 'border-emerald-400 bg-emerald-50 ring-2 ring-emerald-200'
                            : isOriginalDimmed
                              ? 'border-gray-200 bg-gray-50 opacity-60'
                              : isOriginal
                                ? 'border-gray-300 bg-gray-50'
                                : 'border-gray-200 bg-white'
                        }`}
                      >
                        {isOriginal && (
                          <span className={`absolute -top-2.5 left-1/2 -translate-x-1/2 text-[9px] font-bold px-2 py-0.5 rounded-full ${
                            isOriginalDimmed ? 'bg-gray-400 text-white' : 'bg-gray-700 text-white'
                          }`}>
                            ORIGINAL
                          </span>
                        )}
                        {isReassignedHere && (
                          <span className="absolute -top-2.5 left-1/2 -translate-x-1/2 text-[9px] font-bold bg-emerald-600 text-white px-2 py-0.5 rounded-full">
                            REASSIGNED
                          </span>
                        )}

                        <div className="text-center mb-2">
                          <span className="text-sm font-bold" style={{ color: PHYSICIAN_COLORS[name] }}>
                            {name}
                          </span>
                        </div>

                        <MiniGauge probability={pred.probability} size={110} />

                        {!isOriginal && (
                          <div className="text-center mt-1">
                            <span
                              className={`text-xs font-bold ${
                                delta < 0 ? 'text-emerald-600' : delta > 0 ? 'text-red-600' : 'text-gray-400'
                              }`}
                            >
                              {delta < 0 ? '' : '+'}
                              {(delta * 100).toFixed(1)}% risk
                            </span>
                          </div>
                        )}

                        {/* Top 3 SHAP */}
                        <div className="mt-3 space-y-1.5">
                          {shapEntries.map(([feat, val]) => (
                            <div key={feat} className="flex items-center justify-between text-[10px]">
                              <span className="text-gray-500 truncate mr-1">{FEATURE_LABELS[feat] || feat}</span>
                              <span className={`font-mono font-semibold ${val > 0 ? 'text-red-500' : 'text-blue-500'}`}>
                                {val > 0 ? '+' : ''}
                                {val.toFixed(2)}
                              </span>
                            </div>
                          ))}
                        </div>

                        {/* Reassign / Undo button */}
                        {!isOriginal && (
                          <div className="mt-3">
                            {isReassignedHere ? (
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  onUndoReassign(selectedCase.caseNum);
                                }}
                                className="w-full py-1.5 px-3 text-xs font-semibold rounded-lg border-2 border-amber-300 bg-amber-50 text-amber-700 hover:bg-amber-100 transition-colors cursor-pointer flex items-center justify-center gap-1"
                              >
                                <X size={12} />
                                Undo
                              </button>
                            ) : (
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  onReassign(selectedCase.caseNum, id);
                                }}
                                className="w-full py-1.5 px-3 text-xs font-semibold rounded-lg border-2 border-blue-200 bg-blue-50 text-blue-700 hover:bg-blue-100 hover:border-blue-300 transition-colors cursor-pointer flex items-center justify-center gap-1"
                              >
                                <ArrowRightLeft size={12} />
                                Reassign to {name}
                              </button>
                            )}
                          </div>
                        )}
                      </motion.div>
                    );
                  })}
                </div>
              </motion.div>
            </AnimatePresence>
          )}
        </div>
      </div>
    </motion.div>
  );
}

// ---------------------------------------------------------------------------
// Batch Reassignment (collapsible)
// ---------------------------------------------------------------------------

function BatchReassignment() {
  const [open, setOpen] = useState(false);
  const [sourcePhys, setSourcePhys] = useState('Dr. B');
  const [targetPhys, setTargetPhys] = useState('Dr. A');
  const [showCases, setShowCases] = useState(false);

  const scenario = useMemo(
    () =>
      rData.batchScenarios.find(
        (s) => s.sourcePhysician === sourcePhys && s.targetPhysician === targetPhys
      ),
    [sourcePhys, targetPhys]
  );

  const physicians = ['Dr. A', 'Dr. B', 'Dr. C'];
  const targetOptions = physicians.filter((p) => p !== sourcePhys);

  // Ensure target is valid when source changes
  const handleSourceChange = (v: string) => {
    setSourcePhys(v);
    if (v === targetPhys) {
      const first = physicians.find((p) => p !== v);
      if (first) setTargetPhys(first);
    }
  };

  const resolvedPct = scenario && scenario.affectedCases > 0
    ? (scenario.outliersResolved / scenario.affectedCases) * 100
    : 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.4 }}
      className="bg-white rounded-xl border border-gray-200 mb-6"
    >
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between p-5 cursor-pointer"
      >
        <div className="flex items-center gap-2">
          <Shuffle size={18} className="text-[#1e40af]" />
          <h2 className="text-lg font-semibold text-gray-900">Batch Reassignment</h2>
        </div>
        {open ? <ChevronUp size={18} className="text-gray-400" /> : <ChevronDown size={18} className="text-gray-400" />}
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <div className="px-5 pb-5">
              {/* Dropdowns */}
              <div className="flex flex-wrap items-center gap-3 mb-4">
                <span className="text-sm text-gray-600">Reassign</span>
                <select
                  value={sourcePhys}
                  onChange={(e) => handleSourceChange(e.target.value)}
                  className="border border-gray-200 rounded-lg px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {physicians.map((p) => (
                    <option key={p} value={p}>{p}'s outliers</option>
                  ))}
                </select>
                <span className="text-sm text-gray-600">to</span>
                <select
                  value={targetPhys}
                  onChange={(e) => setTargetPhys(e.target.value)}
                  className="border border-gray-200 rounded-lg px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {targetOptions.map((p) => (
                    <option key={p} value={p}>{p}</option>
                  ))}
                </select>
              </div>

              {/* Results */}
              {scenario && (
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="flex items-center gap-4 mb-3">
                    <div>
                      <span className="text-xs text-gray-500">Outliers resolved</span>
                      <p className="text-2xl font-bold text-gray-900">
                        {scenario.outliersResolved}
                        <span className="text-sm font-normal text-gray-400"> / {scenario.affectedCases}</span>
                      </p>
                    </div>
                    <div className="flex-1">
                      <div className="w-full bg-gray-200 rounded-full h-3">
                        <motion.div
                          className="h-3 rounded-full"
                          style={{ backgroundColor: resolvedPct === 100 ? '#10b981' : resolvedPct > 0 ? '#f59e0b' : '#ef4444' }}
                          initial={{ width: 0 }}
                          animate={{ width: `${resolvedPct}%` }}
                          transition={{ duration: 0.5 }}
                        />
                      </div>
                      <span className="text-xs text-gray-400 mt-0.5 block">
                        {resolvedPct.toFixed(0)}% resolved
                        {scenario.outliersRemaining > 0 && ` -- ${scenario.outliersRemaining} remaining`}
                      </span>
                    </div>
                  </div>

                  {scenario.cases.length > 0 && (
                    <>
                      <button
                        onClick={() => setShowCases(!showCases)}
                        className="text-xs text-blue-600 hover:underline mb-2 cursor-pointer"
                      >
                        {showCases ? 'Hide' : 'Show'} affected cases ({scenario.cases.length})
                      </button>

                      <AnimatePresence>
                        {showCases && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            className="overflow-hidden"
                          >
                            <div className="max-h-[300px] overflow-y-auto">
                              <table className="w-full text-xs">
                                <thead>
                                  <tr className="border-b border-gray-200">
                                    <th className="text-left py-2 px-2 text-gray-500 font-medium">Case</th>
                                    <th className="text-left py-2 px-2 text-gray-500 font-medium">Before</th>
                                    <th className="text-left py-2 px-2 text-gray-500 font-medium">After</th>
                                    <th className="text-left py-2 px-2 text-gray-500 font-medium">Status</th>
                                  </tr>
                                </thead>
                                <tbody className="divide-y divide-gray-100">
                                  {scenario.cases.map((bc) => (
                                    <tr key={bc.caseNum}>
                                      <td className="py-1.5 px-2 font-mono">#{bc.caseNum}</td>
                                      <td className="py-1.5 px-2">
                                        <div className="flex items-center gap-1.5">
                                          <div className="w-16 bg-gray-200 rounded-full h-2">
                                            <div
                                              className="h-2 rounded-full bg-red-400"
                                              style={{ width: `${bc.beforeProb * 100}%` }}
                                            />
                                          </div>
                                          <span className="text-gray-600 tabular-nums">{(bc.beforeProb * 100).toFixed(1)}%</span>
                                        </div>
                                      </td>
                                      <td className="py-1.5 px-2">
                                        <div className="flex items-center gap-1.5">
                                          <div className="w-16 bg-gray-200 rounded-full h-2">
                                            <div
                                              className="h-2 rounded-full"
                                              style={{
                                                width: `${bc.afterProb * 100}%`,
                                                backgroundColor: bc.afterProb < 0.5 ? '#10b981' : '#f59e0b',
                                              }}
                                            />
                                          </div>
                                          <span className="text-gray-600 tabular-nums">{(bc.afterProb * 100).toFixed(1)}%</span>
                                        </div>
                                      </td>
                                      <td className="py-1.5 px-2">
                                        {bc.resolved ? (
                                          <span className="text-emerald-600 font-semibold">Resolved</span>
                                        ) : (
                                          <span className="text-amber-600 font-semibold">Remains</span>
                                        )}
                                      </td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </>
                  )}

                  {scenario.cases.length === 0 && (
                    <p className="text-sm text-gray-400 italic">
                      {sourcePhys} has no outlier cases to reassign.
                    </p>
                  )}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

// ---------------------------------------------------------------------------
// Schedule Optimizer (collapsible)
// ---------------------------------------------------------------------------

function ScheduleOptimizer() {
  const [open, setOpen] = useState(false);
  const [showTable, setShowTable] = useState(false);
  const [sortBy, setSortBy] = useState<'impact' | 'case'>('impact');

  const { optimalAssignment, physicianWorkload } = rData;

  const sortedChanges = useMemo(() => {
    const changes = [...optimalAssignment.changes];
    if (sortBy === 'impact') {
      changes.sort((a, b) => (b.beforeProb - b.afterProb) - (a.beforeProb - a.afterProb));
    } else {
      changes.sort((a, b) => a.caseNum - b.caseNum);
    }
    return changes;
  }, [optimalAssignment.changes, sortBy]);

  const workloadData = useMemo(() => {
    return Object.keys(physicianWorkload.current).map((phys) => ({
      physician: phys,
      current: physicianWorkload.current[phys],
      optimal: physicianWorkload.optimal[phys],
      color: PHYSICIAN_COLORS[phys],
    }));
  }, [physicianWorkload]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.5 }}
      className="bg-white rounded-xl border border-gray-200 mb-6"
    >
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between p-5 cursor-pointer"
      >
        <div className="flex items-center gap-2">
          <TrendingDown size={18} className="text-[#1e40af]" />
          <h2 className="text-lg font-semibold text-gray-900">Schedule Optimizer</h2>
        </div>
        {open ? <ChevronUp size={18} className="text-gray-400" /> : <ChevronDown size={18} className="text-gray-400" />}
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <div className="px-5 pb-5">
              {/* Before / After comparison */}
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div className="bg-red-50 border border-red-100 rounded-lg p-4 text-center">
                  <p className="text-xs text-gray-500 mb-1">Current Outliers</p>
                  <p className="text-3xl font-bold text-red-700">{optimalAssignment.currentOutliers}</p>
                </div>
                <div className="bg-emerald-50 border border-emerald-100 rounded-lg p-4 text-center">
                  <p className="text-xs text-gray-500 mb-1">Optimized Outliers</p>
                  <p className="text-3xl font-bold text-emerald-700">{optimalAssignment.optimizedOutliers}</p>
                </div>
              </div>

              {/* Workload comparison chart */}
              <div className="mb-4">
                <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
                  Workload Balance: Current vs Optimal
                </h3>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={workloadData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="physician" fontSize={12} />
                    <YAxis fontSize={11} label={{ value: 'Cases', angle: -90, position: 'insideLeft', fontSize: 12 }} />
                    <Tooltip formatter={(val, name) => [`${val} cases`, name === 'current' ? 'Current' : 'Optimal']} />
                    <Bar dataKey="current" name="Current" radius={[4, 4, 0, 0]}>
                      {workloadData.map((d, i) => (
                        <Cell key={i} fill={d.color} fillOpacity={0.5} />
                      ))}
                    </Bar>
                    <Bar dataKey="optimal" name="Optimal" radius={[4, 4, 0, 0]}>
                      {workloadData.map((d, i) => (
                        <Cell key={i} fill={d.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Caveat */}
              <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 flex items-start gap-2 mb-4">
                <Info size={16} className="text-amber-600 mt-0.5 shrink-0" />
                <p className="text-xs text-amber-800">
                  <strong>Caveat:</strong> The optimal assignment minimizes predicted outliers but creates a heavily
                  skewed workload (Dr. A: {physicianWorkload.optimal['Dr. A']} cases). In practice, workload balance,
                  scheduling constraints, and physician availability must be considered. This is an analytical upper
                  bound, not a practical recommendation.
                </p>
              </div>

              {/* Change table */}
              <button
                onClick={() => setShowTable(!showTable)}
                className="text-xs text-blue-600 hover:underline mb-2 cursor-pointer"
              >
                {showTable ? 'Hide' : 'Show'} reassigned cases ({sortedChanges.length})
              </button>

              <AnimatePresence>
                {showTable && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="overflow-hidden"
                  >
                    <div className="flex gap-2 mb-2">
                      <button
                        onClick={() => setSortBy('impact')}
                        className={`text-[10px] px-2 py-1 rounded cursor-pointer ${sortBy === 'impact' ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-500'}`}
                      >
                        Sort by impact
                      </button>
                      <button
                        onClick={() => setSortBy('case')}
                        className={`text-[10px] px-2 py-1 rounded cursor-pointer ${sortBy === 'case' ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-500'}`}
                      >
                        Sort by case #
                      </button>
                    </div>
                    <div className="max-h-[350px] overflow-y-auto">
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="border-b border-gray-200">
                            <th className="text-left py-2 px-2 text-gray-500 font-medium">Case</th>
                            <th className="text-left py-2 px-2 text-gray-500 font-medium">From</th>
                            <th className="text-left py-2 px-2 text-gray-500 font-medium">To</th>
                            <th className="text-left py-2 px-2 text-gray-500 font-medium">Before</th>
                            <th className="text-left py-2 px-2 text-gray-500 font-medium">After</th>
                            <th className="text-left py-2 px-2 text-gray-500 font-medium">Impact</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-100">
                          {sortedChanges.map((ch) => {
                            const impact = ch.beforeProb - ch.afterProb;
                            return (
                              <tr key={ch.caseNum}>
                                <td className="py-1.5 px-2 font-mono">#{ch.caseNum}</td>
                                <td className="py-1.5 px-2">
                                  <span style={{ color: PHYSICIAN_COLORS[ch.fromPhysician] }} className="font-semibold">
                                    {ch.fromPhysician}
                                  </span>
                                </td>
                                <td className="py-1.5 px-2">
                                  <span style={{ color: PHYSICIAN_COLORS[ch.toPhysician] }} className="font-semibold">
                                    {ch.toPhysician}
                                  </span>
                                </td>
                                <td className="py-1.5 px-2 tabular-nums">{(ch.beforeProb * 100).toFixed(1)}%</td>
                                <td className="py-1.5 px-2 tabular-nums">{(ch.afterProb * 100).toFixed(1)}%</td>
                                <td className="py-1.5 px-2">
                                  <span className={`font-semibold ${impact > 0 ? 'text-emerald-600' : 'text-gray-400'}`}>
                                    {impact > 0 ? '-' : ''}
                                    {(Math.abs(impact) * 100).toFixed(1)}%
                                  </span>
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

// ---------------------------------------------------------------------------
// Your Reassignments Panel
// ---------------------------------------------------------------------------

function YourReassignmentsPanel({
  manualReassignments,
  onUndoReassign,
  onClearAll,
}: {
  manualReassignments: Map<number, number>;
  onUndoReassign: (caseNum: number) => void;
  onClearAll: () => void;
}) {
  const entries = useMemo(() => {
    const result: {
      caseNum: number;
      fromPhysician: string;
      toPhysician: string;
      beforeProb: number;
      afterProb: number;
      isOutlier: boolean;
      resolved: boolean;
    }[] = [];
    manualReassignments.forEach((newPhysId, caseNum) => {
      const c = rData.caseReassignments.find((cr) => cr.caseNum === caseNum);
      if (!c) return;
      const beforeProb = c.predictions[String(c.originalPhysicianEnc)]?.probability ?? 0;
      const afterProb = c.predictions[String(newPhysId)]?.probability ?? 1;
      result.push({
        caseNum,
        fromPhysician: c.originalPhysician,
        toPhysician: PHYSICIAN_ENC_MAP[newPhysId],
        beforeProb,
        afterProb,
        isOutlier: c.outlierClass === 1,
        resolved: c.outlierClass === 1 && afterProb < 0.5,
      });
    });
    return result.sort((a, b) => a.caseNum - b.caseNum);
  }, [manualReassignments]);

  const currentOutliers = rData.caseReassignments.filter((c) => c.outlierClass === 1).length;
  const resolvedCount = entries.filter((e) => e.resolved).length;
  const newOutlierCount = currentOutliers - resolvedCount;

  if (manualReassignments.size === 0) return null;

  return (
    <motion.div
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: 'auto' }}
      exit={{ opacity: 0, height: 0 }}
      transition={{ duration: 0.3 }}
      className="bg-white rounded-xl border border-purple-200 p-5 mb-6 overflow-hidden"
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <UserCheck size={18} className="text-purple-600" />
          <h2 className="text-lg font-semibold text-gray-900">Your Reassignments</h2>
          <span className="text-xs font-bold bg-purple-100 text-purple-700 px-2 py-0.5 rounded-full">
            {manualReassignments.size} case{manualReassignments.size !== 1 ? 's' : ''}
          </span>
        </div>
        <button
          onClick={onClearAll}
          className="flex items-center gap-1 text-xs font-medium text-red-500 hover:text-red-700 hover:bg-red-50 px-2 py-1 rounded-lg transition-colors cursor-pointer"
        >
          <Trash2 size={12} />
          Clear All
        </button>
      </div>

      {/* Running tally */}
      <div className="flex items-center gap-4 mb-4 p-3 bg-gray-50 rounded-lg">
        <div className="text-center">
          <p className="text-xs text-gray-500">Current Outliers</p>
          <p className="text-xl font-bold text-red-600">{currentOutliers}</p>
        </div>
        <ArrowRight size={18} className="text-gray-400" />
        <div className="text-center">
          <p className="text-xs text-gray-500">After Your Changes</p>
          <p className={`text-xl font-bold ${newOutlierCount < currentOutliers ? 'text-emerald-600' : 'text-red-600'}`}>
            {newOutlierCount}
          </p>
        </div>
        {resolvedCount > 0 && (
          <span className="ml-auto text-xs font-semibold text-emerald-600 bg-emerald-50 px-2 py-1 rounded-lg">
            {resolvedCount} outlier{resolvedCount !== 1 ? 's' : ''} resolved
          </span>
        )}
      </div>

      {/* List of reassignments */}
      <div className="space-y-2 max-h-[250px] overflow-y-auto">
        <AnimatePresence>
          {entries.map((e) => (
            <motion.div
              key={e.caseNum}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 10, height: 0 }}
              transition={{ duration: 0.2 }}
              className={`flex items-center gap-3 p-2.5 rounded-lg border ${
                e.resolved
                  ? 'bg-emerald-50 border-emerald-200'
                  : e.isOutlier
                    ? 'bg-amber-50 border-amber-200'
                    : 'bg-gray-50 border-gray-200'
              }`}
            >
              <span className="font-mono text-sm font-semibold text-gray-800">#{e.caseNum}</span>
              <span className="text-xs font-semibold" style={{ color: PHYSICIAN_COLORS[e.fromPhysician] }}>
                {e.fromPhysician}
              </span>
              <ArrowRight size={12} className="text-gray-400" />
              <span className="text-xs font-semibold" style={{ color: PHYSICIAN_COLORS[e.toPhysician] }}>
                {e.toPhysician}
              </span>
              <span className="text-[10px] text-gray-400 tabular-nums">
                ({(e.beforeProb * 100).toFixed(1)}% → {(e.afterProb * 100).toFixed(1)}%)
              </span>
              {e.resolved && (
                <Check size={14} className="text-emerald-600 ml-auto" />
              )}
              <button
                onClick={() => onUndoReassign(e.caseNum)}
                className="ml-auto p-1 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded transition-colors cursor-pointer"
                title="Remove reassignment"
              >
                <X size={14} />
              </button>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </motion.div>
  );
}

// ---------------------------------------------------------------------------
// Main Export
// ---------------------------------------------------------------------------

export default function PatientReassignment() {
  const [manualReassignments, setManualReassignments] = useState<Map<number, number>>(new Map());

  const handleReassign = useCallback((caseNum: number, physId: number) => {
    setManualReassignments((prev) => {
      const next = new Map(prev);
      next.set(caseNum, physId);
      return next;
    });
  }, []);

  const handleUndoReassign = useCallback((caseNum: number) => {
    setManualReassignments((prev) => {
      const next = new Map(prev);
      next.delete(caseNum);
      return next;
    });
  }, []);

  const handleClearAll = useCallback(() => {
    setManualReassignments(new Map());
  }, []);

  return (
    <div>
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-[#1e40af]/10 rounded-lg">
          <ArrowRightLeft size={24} className="text-[#1e40af]" />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Patient Reassignment</h1>
          <p className="text-sm text-gray-500">Explore how physician reassignment impacts outlier risk predictions</p>
        </div>
      </div>

      <ImpactDashboard manualReassignments={manualReassignments} />
      <AnimatePresence>
        <YourReassignmentsPanel
          manualReassignments={manualReassignments}
          onUndoReassign={handleUndoReassign}
          onClearAll={handleClearAll}
        />
      </AnimatePresence>
      <CaseReassignmentSection
        manualReassignments={manualReassignments}
        onReassign={handleReassign}
        onUndoReassign={handleUndoReassign}
      />
      <BatchReassignment />
      <ScheduleOptimizer />
    </div>
  );
}
