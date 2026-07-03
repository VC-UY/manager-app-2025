// app/layout.tsx
import './globals.css';
import type { Metadata } from 'next';
import { AuthProvider } from '@/contexts/AuthContext';

const siteUrl = 'https://manager-vc-uy.npe-techs.com';

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
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
      <body className="min-h-screen antialiased" style={{ fontFamily: 'Inter, system-ui, sans-serif' }}>
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
