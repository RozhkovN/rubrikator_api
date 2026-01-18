import { m, AnimatePresence } from "framer-motion";
import {
  FileText,
  MessageSquare,
  Eye,
  EyeOff,
  Copy,
  Check,
  BarChart3,
} from "lucide-react";
import type { Rubric } from "../../types/chat";
import { formatTemplateText } from "../../utils/formatters";
import { useClipboard } from "../../hooks/useClipboard";
import { AdditionalRubric } from "./AdditionalRubric";

interface RubricContentProps {
  rubric: Rubric;
  allRubrics: Rubric[] | null;
  messageId: string;
  showTemplate: boolean;
  toggleTemplateVisibility: (messageId: string) => void;
  toggleRubricExpansion: (rubricId: number) => void;
  expandedRubrics: { [rubricId: number]: boolean };
  copyToClipboard: (text: string, messageId: string) => void;
}

export function RubricContent({
  rubric,
  allRubrics,
  messageId,
  showTemplate,
  toggleTemplateVisibility,
  toggleRubricExpansion,
  expandedRubrics,
  copyToClipboard,
}: RubricContentProps) {
  const { copiedId } = useClipboard();

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-gray-100">
        <h3 className="text-2xl font-bold text-gray-800">
          Результат классификации
        </h3>
        {rubric && (
          <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-blue-50">
            <BarChart3 className="h-4 w-4 text-blue-600" />
            <span className="text-md font-bold text-blue-700">
              {(rubric.confidence * 100).toFixed(1)}%
            </span>
          </div>
        )}
      </div>
      <div className="space-y-4">
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-sm font-medium text-gray-500">
            <FileText className="h-4 w-4" />
            Наиболее подходящая рубрика
          </div>
          <div className="text-base font-semibold text-gray-800">
            {rubric.short_name}
          </div>
        </div>

        <div className="space-y-2">
          <div className="text-sm font-medium text-gray-500">
            Полное описание
          </div>
          <div className="text-sm text-gray-700 leading-relaxed p-3 bg-gray-50 rounded-lg">
            {rubric.rubric_name}
          </div>
        </div>

        <div className="border-t border-gray-100 pt-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <MessageSquare className="h-4 w-4 text-blue-600" />
              <h4 className="text-sm font-medium text-gray-700">Ответ</h4>
            </div>
            <m.button
              onClick={() => toggleTemplateVisibility(messageId)}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="flex items-center gap-2 text-sm text-blue-600 hover:text-blue-700"
            >
              {showTemplate ? (
                <>
                  <EyeOff className="h-4 w-4" />
                  Скрыть ответ
                </>
              ) : (
                <>
                  <Eye className="h-4 w-4" />
                  Показать ответ
                </>
              )}
            </m.button>
          </div>

          <AnimatePresence>
            {showTemplate && (
              <m.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.3 }}
                className="overflow-hidden"
              >
                <div className="bg-blue-50 rounded-lg border border-blue-100 overflow-hidden">
                  <div className="p-4 max-h-96 overflow-y-auto">
                    <div className="space-y-3">
                      {formatTemplateText(rubric.response_template).map(
                        (paragraph, index) => (
                          <m.p
                            key={index}
                            initial={{ opacity: 0, y: 5 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: index * 0.05 }}
                            className="text-sm text-gray-700 leading-relaxed"
                          >
                            {paragraph}
                          </m.p>
                        ),
                      )}
                    </div>
                  </div>
                  <div className="px-4 py-3 bg-blue-100 border-t border-blue-200">
                    <div className="flex justify-between items-center text-xs text-blue-700">
                      <div className="font-medium">
                        Длина ответа: {rubric.response_template.length} символов
                      </div>
                      <m.button
                        onClick={() =>
                          copyToClipboard(rubric.response_template, messageId)
                        }
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        className="flex items-center gap-1 hover:text-blue-800"
                      >
                        {copiedId === messageId ? (
                          <>
                            <Check className="h-3 w-3" />
                            Скопировано
                          </>
                        ) : (
                          <>
                            <Copy className="h-3 w-3" />
                            Копировать ответ
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
      </div>

      {allRubrics && allRubrics.length > 1 && (
        <m.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="border-t border-gray-100 pt-4"
        >
          <h4 className="mb-3 text-sm font-medium text-gray-700">
            Дополнительные варианты
          </h4>
          <div className="space-y-2">
            {allRubrics.slice(1).map((additionalRubric, index) => (
              <AdditionalRubric
                key={additionalRubric.rubric_id}
                rubric={additionalRubric}
                index={index}
                messageId={messageId}
                toggleRubricExpansion={toggleRubricExpansion}
                isExpanded={expandedRubrics[additionalRubric.rubric_id]}
                copyToClipboard={copyToClipboard}
              />
            ))}
          </div>
        </m.div>
      )}
    </div>
  );
}
