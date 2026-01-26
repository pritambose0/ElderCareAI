import React, { useState } from 'react';
import { Smartphone, ArrowRight } from 'lucide-react';
import { Input } from '@/components/common/Input';
import { Button } from '@/components/common/Button';
import { motion } from 'framer-motion';

export const AdminTwoFAVerify = () => {
    const [code, setCode] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const handleVerify = (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);
        setTimeout(() => {
            setIsLoading(false);
            console.log('Verify code', code);
        }, 1500);
    };

    return (
        <div className="min-h-screen flex items-center justify-center bg-slate-950 relative overflow-hidden font-sans text-slate-100">
            <div className="absolute inset-0 z-0 pointer-events-none">
                <div className="absolute top-0 right-1/4 w-96 h-96 bg-blue-900/10 rounded-full blur-3xl opacity-50"></div>
                <div className="absolute bottom-0 left-1/4 w-96 h-96 bg-slate-800/10 rounded-full blur-3xl opacity-50"></div>
            </div>

            <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5 }}
                className="w-full max-w-md z-10 p-4"
            >
                <div className="bg-slate-900 border border-slate-800 rounded-xl shadow-2xl overflow-hidden backdrop-blur-sm">
                    <div className="bg-slate-950/50 p-6 text-center border-b border-slate-800">
                        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-blue-500/10 mb-4 ring-1 ring-blue-500/20">
                            <Smartphone className="w-8 h-8 text-blue-500" />
                        </div>
                        <h1 className="text-xl font-bold text-white tracking-tight">Two-Factor Authentication</h1>
                        <p className="text-slate-400 text-sm mt-2">Enter the code from your authenticator app</p>
                    </div>

                    <div className="p-8">
                        <form onSubmit={handleVerify} className="space-y-5">
                            <Input
                                label="Authentication Code"
                                type="text"
                                placeholder="000 000"
                                value={code}
                                onChange={(e) => setCode(e.target.value.replace(/\D/g, '').slice(0, 6))}
                                className="bg-slate-950 border-slate-800 focus:border-blue-500/50 text-center text-2xl tracking-widest font-mono"
                                maxLength={6}
                                autoFocus
                            />

                            <Button
                                type="submit"
                                className="w-full bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 border-none shadow-lg shadow-blue-900/20"
                                isLoading={isLoading}
                                disabled={code.length !== 6}
                            >
                                Verify Session <ArrowRight className="ml-2 w-4 h-4" />
                            </Button>
                        </form>

                        <div className="mt-6 text-center">
                            <button className="text-sm text-slate-500 hover:text-blue-400 transition-colors">
                                Lost access to your device?
                            </button>
                        </div>
                    </div>
                </div>
            </motion.div>
        </div>
    );
};
