import { m, AnimatePresence } from "framer-motion";
import { ChevronDown, ChevronUp, Copy, Check } from "lucide-react";
import type { Rubric } from "../../types/chat";
import { useClipboard } from "../../hooks/useClipboard";

interface AdditionalRubricProps {
  rubric: Rubric;
  index: number;
  messageId: string;
  isExpanded: boolean;
  toggleRubricExpansion: (rubricId: number) => void;
  copyToClipboard: (text: string, messageId: string) => void;
}

export function AdditionalRubric({
  rubric,
  index,
  messageId,
  isExpanded,
  toggleRubricExpansion,
  copyToClipboard,
}: AdditionalRubricProps) {
  const { copiedId } = useClipboard();
  const currentCopyId = `${messageId}-${rubric.rubric_id}`;

  return (
    <div>
      <m.button
        onClick={() => toggleRubricExpansion(rubric.rubric_id)}
        className={`w-full flex items-center justify-between rounded-lg p-3 hover:bg-gray-50 transition-colors ${
          isExpanded
            ? "bg-gray-50 border border-gray-200"
            : "bg-white border border-gray-100"
        }`}
      >
        <div className="flex items-center gap-3">
          <div className="flex h-6 w-6 items-center justify-center rounded-full bg-gray-300 text-gray-600">
            <span className="text-xs font-bold">{index + 2}</span>
          </div>
          <div className="text-left">
            <span className="text-sm text-gray-700">{rubric.short_name}</span>
            <div className="text-sm text-gray-500">
              Уверенность: {(rubric.confidence * 100).toFixed(1)}%
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-500">
            {isExpanded ? "Скрыть" : "Показать"}
          </span>
          {isExpanded ? (
            <ChevronUp className="h-4 w-4 text-gray-500" />
          ) : (
            <ChevronDown className="h-4 w-4 text-gray-500" />
          )}
        </div>
      </m.button>
      <AnimatePresence>
        {isExpanded && (
          <m.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="mt-2 p-4 bg-blue-50 rounded-lg border border-blue-100">
              <div className="space-y-3">
                <div className="text-sm font-medium text-blue-700">
                  Полное название:
                </div>
                <div className="text-sm text-gray-700 leading-relaxed">
                  {rubric.rubric_name}
                </div>
                <div className="pt-3 border-t border-blue-100">
                  <div className="text-sm font-medium text-blue-700 mb-2">
                    Ответ:
                  </div>
                  <div className="text-sm text-gray-700 leading-relaxed bg-white p-3 rounded border border-blue-100 max-h-40 overflow-y-auto">
                    {rubric.response_template}
                  </div>
                </div>
                <div className="flex items-center justify-between text-sm text-gray-500">
                  <div>
                    Уверенность:{" "}
                    <span className="font-semibold">
                      {(rubric.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <m.button
                    onClick={() =>
                      copyToClipboard(rubric.response_template, currentCopyId)
                    }
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className="flex items-center gap-1 text-xs text-blue-600 hover:text-blue-700"
                  >
                    {copiedId === currentCopyId ? (
                      <>
                        <Check className="h-3 w-3" />
                        Скопировано
                      </>
                    ) : (
                      <>
                        <Copy className="h-3 w-3" />
                        Копировать
                      </>
                    )}
                  </m.button>
                </div>
              </div>
            </div>
          </m.div>
        )}
      </AnimatePresence>
    </div>
  );
}
