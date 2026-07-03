'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { ProfileModal } from './ProfileModal';

const navItems = [
  { href: '/workflows', label: 'Mes workflows' },
  { href: '/tasks', label: 'Mes taches' },
  { href: '/volunteers', label: 'Volontaires' },
];

export function ManagerNav() {
  const pathname = usePathname();

  return (
    <div
      className="mb-6 rounded-2xl p-4 backdrop-blur-xl"
      style={{
        background: 'linear-gradient(135deg, rgba(0, 32, 96, 0.8) 0%, rgba(0, 20, 64, 0.8) 100%)',
        border: '2px solid rgba(0, 180, 240, 0.3)',
        boxShadow: '0 8px 32px rgba(0, 32, 96, 0.5)',
      }}
    >
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap items-center gap-3">
          <Link href="/workflows" className="text-lg font-bold text-cyan-300">
            VolunSys Manager
          </Link>
          {navItems.map((item) => {
            const active = pathname === item.href || pathname.startsWith(item.href + '/');
            return (
              <Link
                key={item.href}
                href={item.href}
                className="rounded-xl px-4 py-2 text-sm font-semibold transition-all"
                style={{
                  background: active ? 'rgba(0, 212, 255, 0.2)' : 'transparent',
                  color: active ? '#00D4FF' : '#7DD3FC',
                  border: active ? '1px solid rgba(0, 212, 255, 0.4)' : '1px solid transparent',
                }}
              >
                {item.label}
              </Link>
            );
          })}
        </div>
        <ProfileModal />
      </div>
    </div>
  );
}
