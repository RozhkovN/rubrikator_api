import { m } from "framer-motion";
import { FileText, Clock } from "lucide-react";

export function WelcomeScreen() {
  return (
    <m.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
      className="flex h-full flex-col items-center justify-center p-8 text-center"
    >
      <m.div
        animate={{
          y: [0, -5, 0],
        }}
        transition={{
          repeat: Infinity,
          duration: 3,
          ease: "easeInOut",
        }}
        className="mb-8"
      >
        <div className="relative">
          <div className="absolute inset-0 bg-blue-100 rounded-full blur-xl opacity-30"></div>
          <div className="relative bg-linear-to-r from-blue-600 to-blue-700 p-6 rounded-2xl shadow-lg">
            <FileText className="h-12 w-12 text-white" />
          </div>
        </div>
      </m.div>
      <m.h2
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="mb-4 text-3xl font-bold text-gray-800"
      >
        Начните работу с рубрикатором
      </m.h2>
      <m.p
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="max-w-md text-md text-gray-600 mb-6"
      >
        Введите текст или загрузите файлы для классификации
        <br /> Получите рубрику и готовый ответ
      </m.p>
      <m.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="rounded-lg bg-blue-50 p-4 border border-blue-100 max-w-md"
      >
        <div className="flex items-center gap-2 text-blue-600 mb-2">
          <Clock className="h-4 w-4" />
          <span className="font-medium">Пример использования:</span>
        </div>
        <p className="text-sm text-gray-700 italic">
          "Банк заблокировал мою карту без предупреждения согласно 161-ФЗ"
        </p>
      </m.div>
    </m.div>
  );
}
