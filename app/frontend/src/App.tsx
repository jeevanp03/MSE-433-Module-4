import { useState } from 'react';
import Layout from './components/Layout';
import Overview from './components/Overview';
import PhysicianComparison from './components/PhysicianComparison';
import ShapExplorer from './components/ShapExplorer';
import OutlierDeepDive from './components/OutlierDeepDive';
import WhatIfSimulator from './components/WhatIfSimulator';
import Trends from './components/Trends';
import RepositioningAnalysis from './components/RepositioningAnalysis';
import PrepTracker from './components/PrepTracker';
import type { TabId } from './types';

const pages: Record<TabId, () => JSX.Element> = {
  overview: Overview,
  physicians: PhysicianComparison,
  shap: ShapExplorer,
  outliers: OutlierDeepDive,
  whatif: WhatIfSimulator,
  trends: Trends,
  repositioning: RepositioningAnalysis,
  preptracker: PrepTracker,
};

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>('overview');
  const Page = pages[activeTab];

  return (
    <Layout activeTab={activeTab} onTabChange={setActiveTab}>
      <Page />
    </Layout>
  );
}
