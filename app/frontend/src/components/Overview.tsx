import { useEffect, useState, useMemo } from 'react';
import { LayoutDashboard, Clock, AlertTriangle, Users, Activity } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Cell,
} from 'recharts';
import { motion } from 'framer-motion';
import dashboardData from '../data/dashboard_data.json';
import type { DashboardData } from '../types';

const data = dashboardData as DashboardData;

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
  return <span>{display.toFixed(decimals)}{suffix}</span>;
}

interface MetricCardProps {
  title: string;
  value: number;
  suffix?: string;
  decimals?: number;
  icon: React.ReactNode;
  color: string;
  subtitle?: string;
}

function MetricCard({ title, value, suffix, decimals, icon, color, subtitle }: MetricCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-gray-500">{title}</span>
        <div className={`p-2 rounded-lg ${color}`}>{icon}</div>
      </div>
      <div className="text-3xl font-bold text-gray-900">
        <AnimatedNumber value={value} suffix={suffix} decimals={decimals} />
      </div>
      {subtitle && <p className="text-xs text-gray-400 mt-1">{subtitle}</p>}
    </motion.div>
  );
}

export default function Overview() {
  const { metadata, globalModel, distributions, physicians, cases } = data;
  const outlierRate = ((metadata.outlierCount / metadata.totalCases) * 100);

  // Build histogram data from distributions -- bins has 21 edges, counts has 20
  const histData = useMemo(() =>
    distributions.overall.counts.map((count: number, i: number) => {
      const binCenter = (distributions.overall.bins[i] + distributions.overall.bins[i + 1]) / 2;
      return {
        bin: `${Math.round(binCenter)}`,
        binCenter,
        count,
      };
    }),
    [distributions]
  );

  // Top feature importance
  const topFeatures = useMemo(() =>
    [...globalModel.featureImportance]
      .sort((a, b) => b.shapMean - a.shapMean)
      .slice(0, 5),
    [globalModel]
  );

  // Compute findings from data instead of hardcoding
  const findings = useMemo(() => {
    // Find physician with highest outlier rate
    const physEntries = Object.entries(physicians).map(([name, p]) => ({
      name,
      outlierCount: p.outlierCount,
      caseCount: p.caseCount,
      rate: (p.outlierCount / p.caseCount) * 100,
    }));
    const topPhys = physEntries.reduce((best, p) => p.rate > best.rate ? p : best, physEntries[0]);

    // Top SHAP feature
    const topFeature = [...globalModel.featureImportance].sort((a, b) => b.shapMean - a.shapMean)[0];

    // Top modifiable timing phases (exclude physician/scheduling)
    const nonModifiable = new Set(['PHYSICIAN_ENC', 'NOTE_CTI', 'NOTE_BOX', 'NOTE_PST', 'NOTE_SVC']);
    const topTimingPhases = [...globalModel.featureImportance]
      .filter(f => !nonModifiable.has(f.feature))
      .sort((a, b) => b.shapMean - a.shapMean)
      .slice(0, 2)
      .map(f => f.feature);

    // Compute outlier counts per physician from cases
    const outliersByPhys: Record<string, number> = {};
    cases.forEach(c => {
      if (c.outlierClass === 1) {
        outliersByPhys[c.physician] = (outliersByPhys[c.physician] || 0) + 1;
      }
    });

    return [
      `${metadata.outlierCount} cases (${outlierRate.toFixed(1)}%) exceed the 90th percentile threshold of ${metadata.threshold} min`,
      `${topPhys.name} has the highest outlier rate with ${topPhys.outlierCount} outliers out of ${topPhys.caseCount} cases (${topPhys.rate.toFixed(1)}%)`,
      `${topFeature.feature} is the top SHAP driver -- who performs the procedure matters most`,
      `${topTimingPhases.join(' and ')} are the top modifiable timing phases driving long cases`,
      `Distribution is right-skewed (skewness: ${metadata.skewness}) with heavy tails (kurtosis: ${metadata.kurtosis})`,
    ];
  }, [metadata, globalModel, physicians, cases, outlierRate]);

  return (
    <div>
      <div className="flex items-center gap-3 mb-6">
        <LayoutDashboard size={28} className="text-[#1e40af]" />
        <h1 className="text-2xl font-bold text-gray-900">Overview</h1>
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <MetricCard
          title="Total Cases"
          value={metadata.totalCases}
          icon={<Activity size={18} className="text-blue-600" />}
          color="bg-blue-50"
          subtitle="After cleaning (5 dropped)"
        />
        <MetricCard
          title="Outliers Detected"
          value={metadata.outlierCount}
          icon={<AlertTriangle size={18} className="text-amber-600" />}
          color="bg-amber-50"
          subtitle={`${outlierRate.toFixed(1)}% of all cases`}
        />
        <MetricCard
          title="90th Percentile Threshold"
          value={metadata.threshold}
          suffix=" min"
          decimals={1}
          icon={<Clock size={18} className="text-teal-600" />}
          color="bg-teal-50"
          subtitle="Global outlier cutoff"
        />
        <MetricCard
          title="Physicians"
          value={Object.keys(physicians).length}
          icon={<Users size={18} className="text-purple-600" />}
          color="bg-purple-50"
          subtitle={`Mean duration: ${metadata.meanDuration} min`}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* Distribution histogram */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
        >
          <h2 className="text-lg font-semibold text-gray-900 mb-4">PT IN-OUT Distribution</h2>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={histData} margin={{ top: 5, right: 20, bottom: 20, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="bin" fontSize={11} label={{ value: 'Duration (min)', position: 'insideBottom', offset: -10, fontSize: 12 }} />
              <YAxis fontSize={11} label={{ value: 'Count', angle: -90, position: 'insideLeft', fontSize: 12 }} />
              <Tooltip />
              <ReferenceLine x={`${Math.round(metadata.threshold)}`} stroke="#ef4444" strokeDasharray="5 5" label={{ value: '90th %ile', fill: '#ef4444', fontSize: 11 }} />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {histData.map((entry, i) => (
                  <Cell key={i} fill={entry.binCenter >= metadata.threshold ? '#ef4444' : '#3b82f6'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Top SHAP drivers */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
        >
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Top SHAP Drivers (Global)</h2>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={topFeatures} layout="vertical" margin={{ top: 5, right: 20, bottom: 5, left: 120 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis type="number" fontSize={11} label={{ value: 'Mean |SHAP|', position: 'insideBottom', offset: -5, fontSize: 12 }} />
              <YAxis type="category" dataKey="feature" fontSize={11} width={110} />
              <Tooltip />
              <Bar dataKey="shapMean" fill="#0d9488" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      {/* Key findings */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
      >
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Key Findings</h2>
        <div className="space-y-3">
          {findings.map((finding, i) => (
            <div key={i} className="flex items-start gap-3">
              <span className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-100 text-blue-700 flex items-center justify-center text-xs font-bold">
                {i + 1}
              </span>
              <p className="text-sm text-gray-700">{finding}</p>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Quick stats table */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mt-6"
      >
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Duration Statistics</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
          {[
            { label: 'Mean', value: `${metadata.meanDuration} min` },
            { label: 'Median', value: `${metadata.medianDuration} min` },
            { label: 'Std Dev', value: `${metadata.stdDuration} min` },
            { label: 'Range', value: `${metadata.minDuration}--${metadata.maxDuration} min` },
          ].map((stat) => (
            <div key={stat.label} className="p-3 bg-gray-50 rounded-lg">
              <p className="text-xs text-gray-500 mb-1">{stat.label}</p>
              <p className="text-lg font-semibold text-gray-900">{stat.value}</p>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  );
}
