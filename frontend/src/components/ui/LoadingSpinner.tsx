import { m } from "framer-motion";
import { Bot } from "lucide-react";

interface LoadingSpinnerProps {
  isUploading?: boolean;
}

export function LoadingSpinner({ isUploading = false }: LoadingSpinnerProps) {
  return (
    <div className="flex items-center gap-3">
      <div className="flex h-10 w-10 items-center justify-center rounded-full bg-linear-to-r from-blue-600 to-blue-700 shadow-sm">
        <Bot className="h-5 w-5 text-white" />
      </div>
      <div className="rounded-xl rounded-bl-none bg-white p-4 shadow-md border border-gray-100">
        <div className="flex items-center gap-2">
          <m.div
            animate={{ rotate: 360 }}
            transition={{
              repeat: Infinity,
              duration: 1,
              ease: "linear",
            }}
            className="h-4 w-4 rounded-full border-2 border-blue-600 border-t-transparent"
          />
          <p className="text-sm font-medium text-gray-700">
            {isUploading ? "Анализ файлов..." : "Анализ обращения..."}
          </p>
        </div>
        <div className="mt-2 h-1 w-48 bg-gray-200 rounded-full overflow-hidden">
          <m.div
            initial={{ width: "0%" }}
            animate={{ width: "100%" }}
            transition={{
              repeat: Infinity,
              duration: 1.5,
              ease: "easeInOut",
            }}
            className="h-full bg-linear-to-r from-blue-400 to-blue-600"
          />
        </div>
      </div>
    </div>
  );
}
