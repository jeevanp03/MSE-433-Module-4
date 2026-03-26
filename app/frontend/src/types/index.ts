// ── Case ────────────────────────────────────────────────────────────────────

export interface Case {
  caseNum: number;
  date: string;
  physician: string;
  ptInOut: number;
  note: string | null;
  outlierClass: number;
  outlierLabel: string;
  physOutlierClass: number;
  physOutlierLabel: string;
  features: Record<string, number>;
}

// ── Global Model ────────────────────────────────────────────────────────────

export interface FeatureImportance {
  feature: string;
  importance: number;
  shapMean: number;
}

export interface GlobalModel {
  featureImportance: FeatureImportance[];
  threshold: number;
  params: Record<string, unknown>;
}

// ── Physicians ──────────────────────────────────────────────────────────────

export interface PhysicianData {
  caseCount: number;
  outlierCount: number;
  meanDuration: number;
  medianDuration: number;
  iqrThreshold: number;
  topDrivers: Record<string, number>;
  Q1: number;
  Q3: number;
  IQR: number;
  modelFitted: boolean;
}

// ── Feature Stats ───────────────────────────────────────────────────────────

export interface FeatureStats {
  mean: number;
  std: number;
  min: number;
  max: number;
  median: number;
  q25: number;
  q75: number;
}

// ── Distributions ───────────────────────────────────────────────────────────

export interface OverallDistribution {
  bins: number[];
  counts: number[];
}

export interface Distributions {
  overall: OverallDistribution;
  byPhysician: Record<string, OverallDistribution>;
}

// ── Trends ──────────────────────────────────────────────────────────────────

export interface LearningCurvePoint {
  caseNum: number;
  duration: number;
  physician: string;
}

export interface Trends {
  learningCurve: LearningCurvePoint[];
  learningCurveStats: Record<string, unknown>;
}

// ── Complexity ──────────────────────────────────────────────────────────────

export interface ProcedureType {
  type: string;
  totalCases: number;
  outlierCases: number;
  outlierRate: number;
  meanDuration: number;
  medianDuration: number;
}

export interface Complexity {
  procedureTypes: ProcedureType[];
}

// ── Physician Severity ──────────────────────────────────────────────────────

export interface PhysicianSeverity {
  n_cases: number;
  pt_in_out: { mean: number; std: number };
  abl_sites: { mean: number };
  applications: number;
  abl_duration_mean: number;
  abl_time_mean: number;
  repositioning_time_mean: number;
  pre_map_mean: number;
  tsp_mean: number;
  pct_additional_procedures: number;
  additional_breakdown: Record<string, number>;
  cases_2plus_additional: number;
  troubleshoot_cases: number;
}

// ── Metadata ────────────────────────────────────────────────────────────────

export interface Metadata {
  totalCases: number;
  outlierCount: number;
  threshold: number;
  skewness: number;
  kurtosis: number;
  meanDuration: number;
  medianDuration: number;
  stdDuration: number;
  minDuration: number;
  maxDuration: number;
  dateGenerated: string;
  featuresUsed: string[];
  physicianSeverity: Record<string, PhysicianSeverity>;
}

// ── Dashboard Data (top-level) ──────────────────────────────────────────────

export interface DashboardData {
  cases: Case[];
  globalModel: GlobalModel;
  physicians: Record<string, PhysicianData>;
  shapValues: number[][];
  featureStats: Record<string, FeatureStats>;
  distributions: Distributions;
  trends: Trends;
  complexity: Complexity;
  metadata: Metadata;
}

// ── What-If Data ────────────────────────────────────────────────────────────

export interface WhatIfFeatureRange {
  name: string;
  min: number;
  max: number;
  median: number;
  step: number;
}

export interface ResponsePoint {
  featureValue: number;
  probability: number;
  shapContributions: Record<string, number>;
}

export interface ResponseSurface {
  featureName: string;
  values: ResponsePoint[];
}

export interface WhatIfPreset {
  featureValues: Record<string, number>;
  outlierProbability: number;
}

export interface WhatIfData {
  features: WhatIfFeatureRange[];
  responseSurface: ResponseSurface[];
  presets: Record<string, WhatIfPreset>;
  allFeatures: string[];
  medians: Record<string, number>;
}

// ── Repositioning ──────────────────────────────────────────────────────────

export interface RepoGlobalStats {
  repoCorrelation: number;
  repoR2Pct: number;
  ablDurationCorrelation: number;
  ablTimeCorrelation: number;
  outlierMean: number;
  normalMean: number;
  diffMin: number;
  sitesVsRepoCorrelation: number;
  globalAblDurMean: number;
  globalAblTimeMean: number;
  globalRepoMean: number;
  repoPctOfAbl: number;
  clinicalFloorPerSite: number;
  clinicalFloorTotal: number;
  bestInClassPhys: string;
  bestInClassRate: number;
  totalProgramSavingsMin: number;
  avgSavingsPerCaseMin: number;
}

export interface RepoPhysicianStats {
  n: number;
  mean: number;
  median: number;
  std: number;
  cvPct: number;
  min: number;
  max: number;
  outlierMean: number | null;
  normalMean: number | null;
  repoPerSiteMean: number;
  repoPerSiteStd: number;
  ablDurationMean: number;
  ablTimeMean: number;
  repoPctOfAbl: number;
  rWithPtInout: number;
}

export interface SavingsProjection {
  currentRate: number;
  currentMeanRepo: number;
  meanAblSites: number;
  bestInClassTarget: number;
  bestInClassPhys: string;
  projectedRepoAtBest: number;
  savingsVsBestMin: number;
  clinicalFloorRate: number;
  projectedRepoAtFloor: number;
  savingsVsFloorMin: number;
}

export interface RepoCaseData {
  caseNum: number;
  physician: string;
  ablDuration: number;
  ablTime: number;
  repoTime: number;
  repoPerSite: number;
  numAbl: number;
  ptInOut: number;
  outlierClass: number;
}

export interface RepositioningData {
  globalStats: RepoGlobalStats;
  perPhysician: Record<string, RepoPhysicianStats>;
  savingsProjections: Record<string, SavingsProjection>;
  perCase: RepoCaseData[];
}

// ── Dashboard Data (top-level) ──────────────────────────────────────────────

export interface DashboardData {
  cases: Case[];
  globalModel: GlobalModel;
  physicians: Record<string, PhysicianData>;
  shapValues: number[][];
  featureStats: Record<string, FeatureStats>;
  distributions: Distributions;
  trends: Trends;
  complexity: Complexity;
  repositioning: RepositioningData;
  metadata: Metadata;
}

// ── Navigation ──────────────────────────────────────────────────────────────

export type TabId =
  | 'overview'
  | 'physicians'
  | 'shap'
  | 'outliers'
  | 'whatif'
  | 'trends'
  | 'repositioning';
