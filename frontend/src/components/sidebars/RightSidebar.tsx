import { m } from "framer-motion";
import { BarChart3 } from "lucide-react";

interface RightSidebarProps {
  isMobile: boolean;
}

export function RightSidebar({ isMobile }: RightSidebarProps) {
  return (
    !isMobile && (
      <m.div
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="w-20 lg:w-64 bg-linear-to-b from-blue-50 to-white border-l border-gray-200"
      >
        <div className="h-full p-4 flex flex-col">
          <div className="mb-8">
            <div className="flex items-center justify-center lg:justify-start gap-2">
              <BarChart3 className="h-6 w-6 text-blue-600" />
              <span className="hidden lg:inline text-gray-700 font-semibold text-md">
                Информация
              </span>
            </div>
          </div>

          <div className="flex-1">
            <div className="mb-6">
              <div className="text-gray-500 text-sm uppercase tracking-wider mb-3 hidden lg:block">
                Как работает
              </div>
              <div className="space-y-3">
                <div className="p-3 rounded-lg bg-blue-50/50 border border-blue-100">
                  <div className="text-md font-medium text-blue-700 mb-1">
                    Текстовый ввод
                  </div>
                  <div className="text-sm text-gray-600">
                    Введите текст обращения напрямую
                  </div>
                </div>

                <div className="p-3 rounded-lg bg-blue-50/50 border border-blue-100">
                  <div className="text-md font-medium text-blue-700 mb-1">
                    Загрузка файлов
                  </div>
                  <div className="text-sm text-gray-600">
                    Загрузите один или несколько файлов
                  </div>
                </div>

                <div className="p-3 rounded-lg bg-blue-50/50 border border-blue-100">
                  <div className="text-md font-medium text-blue-700 mb-1">
                    Готовый ответ
                  </div>
                  <div className="text-sm text-gray-600">Получите ответ</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </m.div>
    )
  );
}
