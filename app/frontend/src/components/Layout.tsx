import { useState, type ReactNode } from 'react';
import {
  LayoutDashboard,
  Users,
  BrainCircuit,
  AlertTriangle,
  SlidersHorizontal,
  ArrowRightLeft,
  TrendingUp,
  Activity,
  Menu,
  X,
} from 'lucide-react';
import type { TabId } from '../types';

interface NavItem {
  id: TabId;
  label: string;
  icon: ReactNode;
}

const navItems: NavItem[] = [
  { id: 'overview', label: 'Overview', icon: <LayoutDashboard size={20} /> },
  { id: 'physicians', label: 'Physicians', icon: <Users size={20} /> },
  { id: 'shap', label: 'SHAP Explorer', icon: <BrainCircuit size={20} /> },
  { id: 'outliers', label: 'Outlier Deep Dive', icon: <AlertTriangle size={20} /> },
  { id: 'whatif', label: 'What-If Simulator', icon: <SlidersHorizontal size={20} /> },
  { id: 'reassignment', label: 'Reassignment', icon: <ArrowRightLeft size={20} /> },
  { id: 'trends', label: 'Trends', icon: <TrendingUp size={20} /> },
];

interface LayoutProps {
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
  children: ReactNode;
}

export default function Layout({ activeTab, onTabChange, children }: LayoutProps) {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const handleNavClick = (tab: TabId) => {
    onTabChange(tab);
    setSidebarOpen(false);
  };

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/40 z-30 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed lg:static inset-y-0 left-0 z-40 w-64 bg-[#1e3a5f] text-white flex flex-col shrink-0 transform transition-transform duration-200 lg:translate-x-0 ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="p-5 border-b border-white/10 flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <Activity size={24} className="text-teal-400" />
              <span className="font-bold text-lg">AFib PFA</span>
            </div>
            <p className="text-xs text-blue-200 leading-tight">
              Outlier Analysis Dashboard
            </p>
          </div>
          <button
            className="lg:hidden p-1 text-blue-200 hover:text-white cursor-pointer"
            onClick={() => setSidebarOpen(false)}
            aria-label="Close sidebar"
          >
            <X size={20} />
          </button>
        </div>

        <nav className="flex-1 py-4" aria-label="Main navigation">
          {navItems.map((item) => {
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => handleNavClick(item.id)}
                aria-current={isActive ? 'page' : undefined}
                className={`w-full flex items-center gap-3 px-5 py-3 text-sm transition-colors cursor-pointer ${
                  isActive
                    ? 'bg-white/15 text-white border-r-3 border-teal-400'
                    : 'text-blue-200 hover:bg-white/8 hover:text-white'
                }`}
              >
                {item.icon}
                {item.label}
              </button>
            );
          })}
        </nav>

        <div className="p-4 border-t border-white/10 text-xs text-blue-300">
          <p>MSE 433 - Module 4</p>
          <p className="text-blue-400 mt-1">Case Study Analysis</p>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        {/* Mobile header bar */}
        <div className="lg:hidden sticky top-0 z-20 bg-[#1e3a5f] px-4 py-3 flex items-center gap-3">
          <button
            onClick={() => setSidebarOpen(true)}
            className="text-white p-1 cursor-pointer"
            aria-label="Open sidebar menu"
          >
            <Menu size={24} />
          </button>
          <span className="text-white font-semibold text-sm">
            {navItems.find(n => n.id === activeTab)?.label ?? 'Dashboard'}
          </span>
        </div>
        <div className="p-8">{children}</div>
      </main>
    </div>
  );
}
