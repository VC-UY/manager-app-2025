import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

const protectedPrefixes = ['/workflows', '/tasks', '/volunteers'];

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;
  const isProtected = protectedPrefixes.some((p) => pathname.startsWith(p));
  if (!isProtected) return NextResponse.next();

  const token = request.cookies.get('manager_token')?.value;
  if (!token) {
    const login = new URL('/login', request.url);
    login.searchParams.set('next', pathname);
    return NextResponse.redirect(login);
  }
  return NextResponse.next();
}

export const config = {
  matcher: ['/workflows/:path*', '/tasks/:path*', '/volunteers/:path*'],
};
