import React from 'react';
import { Users, AlertTriangle, Activity, Server } from 'lucide-react';

const MetricCard = ({ title, value, change, icon: Icon, trend }: any) => (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 hover:border-slate-700 transition-colors shadow-sm">
        <div className="flex items-center justify-between mb-4">
            <div className="p-2 bg-slate-800 rounded-lg text-slate-400 group-hover:text-white transition-colors">
                <Icon className="w-5 h-5" />
            </div>
            <span className={`text-xs font-medium px-2 py-1 rounded-full ${trend === 'up' ? 'bg-emerald-500/10 text-emerald-500' : 'bg-red-500/10 text-red-500'
                }`}>
                {change}
            </span>
        </div>
        <h3 className="text-slate-400 text-sm font-medium">{title}</h3>
        <p className="text-2xl font-bold text-white mt-1">{value}</p>
    </div>
);

export const DashboardPage = () => {
    return (
        <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
            {/* Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <MetricCard
                    title="Total Users"
                    value="2,543"
                    change="+12.5%"
                    icon={Users}
                    trend="up"
                />
                <MetricCard
                    title="Active Alerts"
                    value="14"
                    change="+2 new"
                    icon={AlertTriangle}
                    trend="down"
                />
                <MetricCard
                    title="System Load"
                    value="42%"
                    change="-5.0%"
                    icon={Activity}
                    trend="up"
                />
                <MetricCard
                    title="Server Uptime"
                    value="99.99%"
                    change="Stable"
                    icon={Server}
                    trend="up"
                />
            </div>

            {/* Content Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-xl p-6 min-h-[400px]">
                    <h3 className="text-lg font-semibold text-white mb-6">Activity Overview</h3>
                    <div className="flex items-center justify-center h-64 border-2 border-dashed border-slate-800 rounded-lg text-slate-500">
                        Analytics Chart Placeholder
                    </div>
                </div>
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 h-fit">
                    <h3 className="text-lg font-semibold text-white mb-4">Recent Critical Events</h3>
                    <div className="space-y-4">
                        {[1, 2, 3, 4].map((i) => (
                            <div key={i} className="flex items-start gap-3 p-3 bg-slate-950/50 rounded-lg border border-slate-800 hover:bg-slate-800/50 transition-colors cursor-pointer">
                                <AlertTriangle className="w-5 h-5 text-amber-500 shrink-0 mt-0.5" />
                                <div>
                                    <p className="text-sm font-medium text-white">Abnormal Heart Rate</p>
                                    <p className="text-xs text-slate-500">Jane Doe • 2 mins ago</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};
