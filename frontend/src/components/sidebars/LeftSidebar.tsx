import { m } from "framer-motion";
import { BarChart3, ChevronUp as ChevronUpIcon } from "lucide-react";

interface LeftSidebarProps {
  isMobile: boolean;
  showStatistics: boolean;
  toggleStatistics: () => void;
  totalRequests: number;
  fileMessages: number;
}

export function LeftSidebar({
  isMobile,
  showStatistics,
  toggleStatistics,
  totalRequests,
  fileMessages,
}: LeftSidebarProps) {
  return (
    !isMobile && (
      <m.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.5 }}
        className="w-20 lg:w-64 bg-linear-to-b from-blue-600 to-blue-800 border-r border-blue-700 flex flex-col"
      >
        <div className="h-full p-4 flex flex-col">
          <div className="mb-8">
            <div className="flex items-center justify-center gap-2">
              <img src="/logo.png" alt="Logo" className="h-30 w-30" />
            </div>
          </div>

          <m.button
            onClick={toggleStatistics}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="flex items-center justify-center lg:justify-start gap-2 mb-4 p-2 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
          >
            <BarChart3 className="h-5 w-5 text-white" />
            {!isMobile && (
              <m.span
                initial={{ opacity: 0, width: 0 }}
                animate={{ opacity: 1, width: "auto" }}
                exit={{ opacity: 0, width: 0 }}
                className="text-white text-sm font-medium whitespace-nowrap overflow-hidden"
              >
                {showStatistics ? "Скрыть статистику" : "Показать статистику"}
              </m.span>
            )}
            <m.div
              animate={{ rotate: showStatistics ? 180 : 0 }}
              transition={{ duration: 0.3 }}
              className="text-white"
            >
              <ChevronUpIcon className="h-4 w-4" />
            </m.div>
          </m.button>

          <div className="flex-1">
            <div className="space-y-4">
              {showStatistics && (
                <div className="space-y-3">
                  <div className="p-3 rounded-lg bg-white/10 backdrop-blur-sm">
                    <div className="text-white text-center">
                      <div className="text-white text-xl font-bold mb-1">
                        {totalRequests}
                      </div>
                      <div className="text-white/70 text-sm">Запросов</div>
                    </div>
                  </div>
                  <div className="p-3 rounded-lg bg-white/10 backdrop-blur-sm">
                    <div className="text-white text-center">
                      <div className="text-white text-xl font-bold mb-1">
                        {fileMessages}
                      </div>
                      <div className="text-white/70 text-sm">Файлов</div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </m.div>
    )
  );
}
