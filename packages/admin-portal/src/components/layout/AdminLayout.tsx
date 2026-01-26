import React from 'react';
import { Outlet, NavLink, useLocation } from 'react-router-dom';
import {
    LayoutDashboard,
    Users,
    Activity,
    Settings,
    ShieldAlert,
    FileText,
    LogOut,
    Bell
} from 'lucide-react';
import { cn } from '@/utils/cn';

export const AdminLayout = () => {
    const location = useLocation();
    const navItems = [
        { icon: LayoutDashboard, label: 'Dashboard', path: '/dashboard' },
        { icon: Users, label: 'User Management', path: '/users' },
        { icon: Activity, label: 'System Health', path: '/health' },
        { icon: FileText, label: 'Audit Logs', path: '/audit' },
        { icon: Settings, label: 'Settings', path: '/settings' },
    ];

    const currentTitle = navItems.find((item) => location.pathname.startsWith(item.path))?.label || 'Overview';

    return (
        <div className="min-h-screen bg-slate-950 flex font-sans text-slate-100">
            {/* Sidebar */}
            <aside className="w-64 bg-slate-900 border-r border-slate-800 flex flex-col fixed h-full z-20">
                <div className="p-6 border-b border-slate-800 flex items-center gap-3">
                    <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
                        <ShieldAlert className="w-5 h-5 text-white" />
                    </div>
                    <span className="font-bold text-lg tracking-tight text-white">ElderGuard</span>
                </div>

                <nav className="flex-1 p-4 space-y-1 overflow-y-auto">
                    <p className="px-3 text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2 mt-2">Menu</p>
                    {navItems.map((item) => (
                        <NavLink
                            key={item.path}
                            to={item.path}
                            className={({ isActive }) => cn(
                                "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200",
                                isActive
                                    ? "bg-blue-600/10 text-blue-500 border border-blue-600/20"
                                    : "text-slate-400 hover:text-slate-100 hover:bg-slate-800/50 border border-transparent"
                            )}
                        >
                            <item.icon className="w-5 h-5" />
                            {item.label}
                        </NavLink>
                    ))}
                </nav>

                <div className="p-4 border-t border-slate-800 bg-slate-900">
                    <div className="flex items-center gap-3 px-3 py-3 rounded-lg bg-slate-800/50 mb-3 border border-slate-800">
                        <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-blue-500 to-purple-500 flex items-center justify-center text-xs font-bold text-white shadow-inner">
                            AD
                        </div>
                        <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium text-white truncate">Admin User</p>
                            <p className="text-xs text-slate-500 truncate">Super Admin</p>
                        </div>
                    </div>
                    <button className="flex items-center gap-3 px-3 py-2 w-full text-sm font-medium text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded-lg transition-colors">
                        <LogOut className="w-5 h-5" />
                        Sign Out
                    </button>
                </div>
            </aside>

            {/* Main Content */}
            <main className="flex-1 ml-64 min-h-screen flex flex-col bg-slate-950">
                {/* Header */}
                <header className="h-16 bg-slate-900/80 backdrop-blur-md border-b border-slate-800 sticky top-0 z-10 px-8 flex items-center justify-between">
                    <h2 className="text-lg font-semibold text-white tracking-tight">{currentTitle}</h2>
                    <div className="flex items-center gap-4">
                        <button className="p-2 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800 transition-colors relative">
                            <Bell className="w-5 h-5" />
                            <span className="absolute top-2 right-2 w-2 h-2 rounded-full bg-red-500 border-2 border-slate-900 ring-2 ring-slate-900"></span>
                        </button>
                    </div>
                </header>

                <div className="p-8 max-w-7xl mx-auto w-full">
                    <Outlet />
                </div>
            </main>
        </div>
    );
};
