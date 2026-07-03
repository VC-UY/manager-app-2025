// app/layout.tsx
import './globals.css';
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import { AuthProvider } from '@/contexts/AuthContext';

const inter = Inter({ subsets: ['latin'] });

const siteUrl = process.env.NEXT_PUBLIC_API_URL?.replace('/api', '').replace(':8002', '') || 'https://manager-vc-uy.npe-techs.com';

export const metadata: Metadata = {
  metadataBase: new URL('https://manager-vc-uy.npe-techs.com'),
  title: {
    default: 'VolunSys-UY1 Manager | Calcul volontaire',
    template: '%s | VolunSys Manager',
  },
  description:
    'Soumettez et suivez vos workflows de calcul scientifique sur le reseau volontaire VolunSys-UY1.',
  keywords: ['VolunSys', 'manager', 'workflow', 'calcul volontaire', 'calcul distribue'],
  icons: {
    icon: '/logo.svg',
    apple: '/logo.svg',
  },
  openGraph: {
    title: 'VolunSys-UY1 Manager',
    description: 'Orchestration de workflows de calcul volontaire.',
    url: siteUrl,
    siteName: 'VolunSys-UY1 Manager',
    locale: 'fr_CM',
    type: 'website',
    images: [{ url: '/logo.svg', width: 120, height: 120, alt: 'VolunSys-UY1' }],
  },
  robots: { index: true, follow: true },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="fr">
      <body className={inter.className}>
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
