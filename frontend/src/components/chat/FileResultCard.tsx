import { m, AnimatePresence } from "framer-motion";
import {
  File,
  BarChart3,
  MessageSquare,
  Eye,
  EyeOff,
  Copy,
  Check,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import type {
  FileResultWithOptions,
  SimpleFileResult,
  ExpandedFileRubric,
  FileTemplateVisibility,
  Rubric,
} from "../../types/chat";
import { formatTemplateText } from "../../utils/formatters";
import { useClipboard } from "../../hooks/useClipboard";

interface FileResultCardProps {
  result: FileResultWithOptions | SimpleFileResult;
  fileIndex: number;
  isFileResultWithOptions: (
    result: FileResultWithOptions | SimpleFileResult,
  ) => result is FileResultWithOptions;
  toggleFileTemplateVisibility: (fileIndex: number) => void;
  toggleFileRubricExpansion: (fileIndex: number, rubricId: number) => void;
  fileTemplateVisibility: FileTemplateVisibility;
  expandedFileRubrics: ExpandedFileRubric;
  copyToClipboard: (text: string, messageId: string) => void;
}

export function FileResultCard({
  result,
  fileIndex,
  isFileResultWithOptions,
  toggleFileTemplateVisibility,
  toggleFileRubricExpansion,
  fileTemplateVisibility,
  expandedFileRubrics,
  copyToClipboard,
}: FileResultCardProps) {
  const { copiedId } = useClipboard();

  // Рендер дополнительных рубрик
  const renderAdditionalFileRubric = (
    rubric: Rubric,
    rubricIndex: number,
    fileIndex: number,
  ) => {
    const key = `${fileIndex}-${rubric.rubric_id}`;
    const isExpanded = expandedFileRubrics[key] || false;
    const currentCopyId = `file-${fileIndex}-${rubric.rubric_id}`;

    return (
      <div key={rubric.rubric_id}>
        <m.button
          onClick={() => toggleFileRubricExpansion(fileIndex, rubric.rubric_id)}
          className={`w-full flex items-center justify-between rounded-lg p-3 hover:bg-gray-50 transition-colors ${
            isExpanded
              ? "bg-gray-50 border border-gray-200"
              : "bg-white border border-gray-100"
          }`}
        >
          <div className="flex items-center gap-3">
            <div className="flex h-6 w-6 items-center justify-center rounded-full bg-gray-300 text-gray-600">
              <span className="text-xs font-bold">{rubricIndex + 2}</span>
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
  };

  if (isFileResultWithOptions(result)) {
    const { best_match: rubric, all_predictions: allRubrics } = result;
    const showTemplate = fileTemplateVisibility[fileIndex] || false;
    const currentCopyId = `file-${fileIndex}`;

    return (
      <m.div
        key={fileIndex}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: fileIndex * 0.1 }}
        className="border border-gray-200 rounded-lg overflow-hidden"
      >
        <div className="bg-gray-50 p-3 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <File className="h-4 w-4 text-blue-600" />
              <div className="font-medium text-gray-700 truncate">
                {result.filename}
              </div>
            </div>
            <div className="flex items-center gap-2 px-2 py-1 rounded-full bg-blue-50">
              <BarChart3 className="h-3 w-3 text-blue-600" />
              <span className="text-xs font-bold text-blue-700">
                {(rubric.confidence * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>

        <div className="p-4">
          <div className="space-y-4">
            <div>
              <div className="text-sm font-medium text-gray-500 mb-1">
                Рубрика:
              </div>
              <div className="text-sm font-semibold text-gray-800">
                {rubric.short_name}
              </div>
            </div>

            <div className="border-t border-gray-100 pt-3">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <MessageSquare className="h-4 w-4 text-blue-600" />
                  <div className="text-sm font-medium text-gray-700">Ответ</div>
                </div>
                <div className="flex items-center gap-2">
                  <m.button
                    onClick={() => toggleFileTemplateVisibility(fileIndex)}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className="flex items-center gap-1 text-sm text-blue-600 hover:text-blue-700"
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
                  <m.button
                    onClick={() =>
                      copyToClipboard(rubric.response_template, currentCopyId)
                    }
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className="flex items-center gap-1 text-sm text-blue-600 hover:text-blue-700"
                  >
                    {copiedId === currentCopyId ? (
                      <>
                        <Check className="h-4 w-4" />
                        Скопировано
                      </>
                    ) : (
                      <>
                        <Copy className="h-4 w-4" />
                        Копировать
                      </>
                    )}
                  </m.button>
                </div>
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
                    <div className="bg-blue-50 rounded-lg border border-blue-100 overflow-hidden mt-2">
                      <div className="p-4 max-h-64 overflow-y-auto">
                        <div className="space-y-3">
                          {formatTemplateText(rubric.response_template).map(
                            (paragraph, paragraphIndex) => (
                              <m.p
                                key={paragraphIndex}
                                initial={{ opacity: 0, y: 5 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: paragraphIndex * 0.05 }}
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
                            Длина ответа: {rubric.response_template.length}{" "}
                            символов
                          </div>
                        </div>
                      </div>
                    </div>
                  </m.div>
                )}
              </AnimatePresence>
            </div>

            {allRubrics && allRubrics.length > 1 && (
              <div className="border-t border-gray-100 pt-3">
                <h4 className="mb-2 text-sm font-medium text-gray-700">
                  Дополнительные варианты
                </h4>
                <div className="space-y-2">
                  {allRubrics
                    .slice(1)
                    .map((additionalRubric, rubricIndex) =>
                      renderAdditionalFileRubric(
                        additionalRubric,
                        rubricIndex,
                        fileIndex,
                      ),
                    )}
                </div>
              </div>
            )}
          </div>
        </div>
      </m.div>
    );
  } else {
    // SimpleFileResult
    const currentCopyId = `file-${fileIndex}`;
    const simpleResult = result as SimpleFileResult;

    return (
      <m.div
        key={fileIndex}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: fileIndex * 0.1 }}
        className="border border-gray-200 rounded-lg overflow-hidden"
      >
        <div className="bg-gray-50 p-3 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <File className="h-4 w-4 text-blue-600" />
              <div className="font-medium text-gray-700 truncate">
                {simpleResult.filename}
              </div>
            </div>
            <div className="flex items-center gap-2 px-2 py-1 rounded-full bg-blue-50">
              <BarChart3 className="h-3 w-3 text-blue-600" />
              <span className="text-xs font-bold text-blue-700">
                {(simpleResult.confidence * 100).toFixed(1)}%
              </span>
            </div>
          </div>
          {simpleResult.text && (
            <div className="mt-2 text-sm text-gray-600 italic">
              "{simpleResult.text}"
            </div>
          )}
        </div>

        <div className="p-4">
          {simpleResult.error ? (
            <div className="text-sm text-red-500 bg-red-50 p-3 rounded">
              Ошибка: {simpleResult.error}
            </div>
          ) : (
            <div className="space-y-3">
              <div>
                <div className="text-sm font-medium text-gray-500 mb-1">
                  Рубрика:
                </div>
                <div className="text-sm font-semibold text-gray-800">
                  {simpleResult.short_name}
                </div>
              </div>

              <div className="border-t border-gray-100 pt-3">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <MessageSquare className="h-4 w-4 text-blue-600" />
                    <div className="text-sm font-medium text-gray-700">
                      Ответ
                    </div>
                  </div>
                  <m.button
                    onClick={() =>
                      copyToClipboard(
                        simpleResult.response_template,
                        currentCopyId,
                      )
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
                <div className="text-sm text-gray-700 leading-relaxed bg-blue-50 p-3 rounded border border-blue-100 max-h-32 overflow-y-auto">
                  {simpleResult.response_template}
                </div>
              </div>
            </div>
          )}
        </div>
      </m.div>
    );
  }
}
