// components/Navigation.tsx
import Link from 'next/link';
import { usePathname } from 'next/navigation';

const Navigation = () => {
  const pathname = usePathname();

  const isActive = (path: string) => {
    return pathname === path || pathname?.startsWith(path + '/');
  };

  return (
    <nav className="bg-gray-800 text-white">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-8">
            <Link href="/" className="flex-shrink-0 text-xl font-bold">
              Workflow Manager
            </Link>
            
            <div className="hidden md:block">
              <div className="flex items-baseline space-x-4">
                <Link href="/workflows" className={`px-3 py-2 rounded-md text-sm font-medium ${isActive('/workflows') ? 'bg-gray-900 text-white' : 'text-gray-300 hover:bg-gray-700 hover:text-white'}`}>
                  Workflows
                </Link>
                <Link href="/tasks" className={`px-3 py-2 rounded-md text-sm font-medium ${isActive('/tasks') ? 'bg-gray-900 text-white' : 'text-gray-300 hover:bg-gray-700 hover:text-white'}`}>
                  Tâches
                </Link>
                <Link href="/volunteers" className={`px-3 py-2 rounded-md text-sm font-medium ${isActive('/volunteers') ? 'bg-gray-900 text-white' : 'text-gray-300 hover:bg-gray-700 hover:text-white'}`}>
                  Volontaires
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;