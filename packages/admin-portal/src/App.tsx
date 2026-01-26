import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AdminLoginPage } from '@/pages/auth/AdminLoginPage';
import { AdminTwoFAVerify } from '@/pages/auth/AdminTwoFAVerify';
import { AdminLayout } from '@/components/layout/AdminLayout';
import { DashboardPage } from '@/pages/dashboard/DashboardPage';

function App() {
  // TODO: Add real auth state check
  const isAuthenticated = false;

  return (
    <BrowserRouter>
      <Routes>
        {/* Public Routes */}
        <Route path="/login" element={<AdminLoginPage />} />
        <Route path="/verify-2fa" element={<AdminTwoFAVerify />} />

        {/* Protected Routes */}
        <Route element={<AdminLayout />}>
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/users" element={<div className="text-slate-400">User Management Module</div>} />
          <Route path="/health" element={<div className="text-slate-400">System Health Module</div>} />
          <Route path="/audit" element={<div className="text-slate-400">Audit Logs Module</div>} />
          <Route path="/settings" element={<div className="text-slate-400">Settings Module</div>} />
        </Route>

        {/* Redirect root to login for now */}
        <Route path="/" element={<Navigate to="/login" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
