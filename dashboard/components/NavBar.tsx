"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_LINKS = [
  { href: "/", label: "Overview", icon: "◉" },
  { href: "/nodes", label: "Nodes", icon: "⬡" },
  { href: "/jobs", label: "Jobs", icon: "▦" },
];

export default function NavBar() {
  const pathname = usePathname();

  return (
    <nav className="bg-gradient-to-r from-slate-900 via-indigo-950 to-slate-900 text-white shadow-lg">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          <div className="flex items-center gap-10">
            <Link href="/" className="flex items-center gap-2.5">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-500 text-sm font-bold shadow-md shadow-indigo-500/30">
                ML
              </div>
              <span className="text-lg font-semibold tracking-tight">
                Trainer
              </span>
            </Link>
            <div className="flex gap-1">
              {NAV_LINKS.map(({ href, label, icon }) => {
                const isActive =
                  href === "/"
                    ? pathname === "/"
                    : pathname.startsWith(href);
                return (
                  <Link
                    key={href}
                    href={href}
                    className={`flex items-center gap-1.5 rounded-lg px-3.5 py-2 text-sm font-medium transition-all duration-200 ${
                      isActive
                        ? "bg-white/15 text-white shadow-sm"
                        : "text-slate-300 hover:bg-white/10 hover:text-white"
                    }`}
                  >
                    <span className="text-xs opacity-70">{icon}</span>
                    {label}
                  </Link>
                );
              })}
            </div>
          </div>
          <Link
            href="/jobs/new"
            className="rounded-lg bg-indigo-500 px-4 py-2 text-sm font-medium text-white shadow-md shadow-indigo-500/30 hover:bg-indigo-400 hover:shadow-lg hover:shadow-indigo-500/40 transition-all duration-200"
          >
            + New Job
          </Link>
        </div>
      </div>
    </nav>
  );
}
