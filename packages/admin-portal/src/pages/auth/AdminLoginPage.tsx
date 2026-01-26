import React, { useState } from 'react';
import { ShieldCheck, AlertTriangle } from 'lucide-react';
import { Input } from '@/components/common/Input';
import { Button } from '@/components/common/Button';
import { motion } from 'framer-motion';

export const AdminLoginPage = () => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [passwordError, setPasswordError] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    // Mock login handler
    const handleLogin = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);
        setPasswordError('');

        // Simulate delay
        setTimeout(() => {
            setIsLoading(false);
            // For now just console log
            console.log('Login attempt', email);
        }, 1500);
    };

    return (
        <div className="min-h-screen flex items-center justify-center bg-slate-950 relative overflow-hidden font-sans text-slate-100">
            {/* Ambient background effects */}
            <div className="absolute inset-0 z-0 pointer-events-none">
                <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-900/10 rounded-full blur-3xl opacity-50"></div>
                <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-slate-800/10 rounded-full blur-3xl opacity-50"></div>
                {/* Grid pattern overlay could be added here via CSS or SVG */}
            </div>

            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="w-full max-w-md z-10 p-4"
            >
                <div className="bg-slate-900 border border-slate-800 rounded-xl shadow-2xl overflow-hidden backdrop-blur-sm">
                    {/* Header */}
                    <div className="bg-slate-950/50 p-6 text-center border-b border-slate-800">
                        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-blue-500/10 mb-4 ring-1 ring-blue-500/20">
                            <ShieldCheck className="w-8 h-8 text-blue-500" />
                        </div>
                        <h1 className="text-2xl font-bold text-white tracking-tight">Admin Portal</h1>
                        <p className="text-slate-400 text-sm mt-2">Secure Access Restricted</p>
                    </div>

                    {/* Form */}
                    <div className="p-8">
                        <form onSubmit={handleLogin} className="space-y-5">
                            <Input
                                label="Administrator Email"
                                type="email"
                                placeholder="admin@elderguard.ai"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                required
                                className="bg-slate-950 border-slate-800 focus:border-blue-500/50"
                            />

                            <Input
                                label="Password"
                                type="password"
                                placeholder="••••••••••••"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                required
                                error={passwordError}
                                className="bg-slate-950 border-slate-800 focus:border-blue-500/50"
                            />

                            <div className="flex items-center justify-between text-sm">
                                <label className="flex items-center text-slate-400 select-none cursor-pointer hover:text-slate-300 transition-colors">
                                    <input type="checkbox" className="mr-2 rounded border-slate-700 bg-slate-900 text-blue-500 focus:ring-blue-500/20 w-4 h-4" />
                                    Remember device
                                </label>
                                <a href="#" className="text-blue-500 hover:text-blue-400 font-medium transition-colors">Forgot password?</a>
                            </div>

                            <Button
                                type="submit"
                                className="w-full bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 border-none shadow-lg shadow-blue-900/20"
                                isLoading={isLoading}
                            >
                                Authenticate
                            </Button>
                        </form>

                        <div className="mt-8 flex items-start gap-3 p-4 bg-red-500/5 border border-red-500/10 rounded-lg">
                            <AlertTriangle className="w-5 h-5 text-red-500 shrink-0 mt-0.5" />
                            <p className="text-xs text-red-400/80 leading-relaxed font-medium">
                                Warning: Unauthorized access attempts are monitored and logged. IP address and device fingerprint will be recorded for security auditing.
                            </p>
                        </div>
                    </div>

                    {/* Footer */}
                    <div className="bg-slate-950 py-3 px-8 text-center border-t border-slate-800">
                        <p className="text-[10px] uppercase tracking-wider text-slate-600 font-mono">ElderGuard Security Systems v1.0.0</p>
                    </div>
                </div>
            </motion.div>
        </div>
    );
};
