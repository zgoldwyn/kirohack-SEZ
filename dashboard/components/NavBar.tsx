"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_LINKS = [
  { href: "/", label: "Overview" },
  { href: "/nodes", label: "Nodes" },
  { href: "/jobs", label: "Jobs" },
];

export default function NavBar() {
  const pathname = usePathname();

  return (
    <nav className="bg-gray-900 text-white shadow-md">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-14 items-center justify-between">
          <div className="flex items-center gap-8">
            <Link href="/" className="text-lg font-semibold tracking-tight">
              ML Trainer
            </Link>
            <div className="flex gap-1">
              {NAV_LINKS.map(({ href, label }) => {
                const isActive =
                  href === "/"
                    ? pathname === "/"
                    : pathname.startsWith(href);
                return (
                  <Link
                    key={href}
                    href={href}
                    className={`rounded px-3 py-1.5 text-sm font-medium transition-colors ${
                      isActive
                        ? "bg-gray-700 text-white"
                        : "text-gray-300 hover:bg-gray-700 hover:text-white"
                    }`}
                  >
                    {label}
                  </Link>
                );
              })}
            </div>
          </div>
          <Link
            href="/jobs/new"
            className="rounded bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-500 transition-colors"
          >
            + New Job
          </Link>
        </div>
      </div>
    </nav>
  );
}
