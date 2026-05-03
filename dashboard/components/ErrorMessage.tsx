interface ErrorMessageProps {
  message?: string;
}

export default function ErrorMessage({ message }: ErrorMessageProps) {
  return (
    <div className="rounded-xl bg-red-50 border border-red-200 p-4 text-sm text-red-700 flex items-start gap-3">
      <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-red-100 text-red-500 text-xs">
        !
      </span>
      <div>
        <p className="font-semibold">Something went wrong</p>
        <p className="mt-0.5 text-red-600">{message || "Please try again."}</p>
      </div>
    </div>
  );
}
