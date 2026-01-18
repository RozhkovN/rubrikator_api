import { m } from "framer-motion";
import { User, Upload, Bot, Check, Copy } from "lucide-react";
import type {
  Message,
  FileResultWithOptions,
  SimpleFileResult,
} from "../../types/chat";
import { LoadingSpinner } from "../ui/LoadingSpinner";
import { RubricContent } from "./RubricContent";
import { FileResultsContent } from "./FileResultsContent";
import { useClipboard } from "../../hooks/useClipboard";
import { formatResults } from "../../utils/formatters";

interface MessageBubbleProps {
  message: Message;
  isUploading: boolean;
  toggleTemplateVisibility: (messageId: string) => void;
  toggleRubricExpansion: (rubricId: number) => void;
  toggleFileTemplateVisibility: (fileIndex: number) => void;
  toggleFileRubricExpansion: (fileIndex: number, rubricId: number) => void;
  expandedRubrics: { [rubricId: number]: boolean };
  expandedFileRubrics: { [key: string]: boolean };
  fileTemplateVisibility: { [fileIndex: number]: boolean };
}

export function MessageBubble({
  message,
  isUploading,
  toggleTemplateVisibility,
  toggleRubricExpansion,
  toggleFileTemplateVisibility,
  toggleFileRubricExpansion,
  expandedRubrics,
  expandedFileRubrics,
  fileTemplateVisibility,
}: MessageBubbleProps) {
  const { copiedId, copyToClipboard } = useClipboard();

  const isFileResultWithOptions = (
    result: FileResultWithOptions | SimpleFileResult,
  ): result is FileResultWithOptions => {
    return "best_match" in result && "all_predictions" in result;
  };

  if (message.isUser) {
    return (
      <m.div
        whileHover={{ scale: 1.01 }}
        whileTap={{ scale: 0.99 }}
        className="group relative max-w-[85%]"
      >
        <div className="flex items-end gap-3">
          <div className="order-2">
            <div className="rounded-xl rounded-br-none bg-blue-600 p-4 text-white shadow-md">
              <div className="flex items-center gap-2 mb-2">
                {message.isFileMessage ? (
                  <Upload className="h-4 w-4 text-white/90" />
                ) : (
                  <User className="h-4 w-4 text-white/90" />
                )}
                <p className="text-lg font-medium">
                  {message.isFileMessage ? "Файлы" : "Вы"}
                </p>
              </div>
              <p className="text-lg text-white/95">{message.text}</p>
            </div>
          </div>
          <div className="order-1">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-blue-700 shadow-sm">
              {message.isFileMessage ? (
                <Upload className="h-5 w-5 text-white" />
              ) : (
                <User className="h-5 w-5 text-white" />
              )}
            </div>
          </div>
        </div>
      </m.div>
    );
  }

  if (message.isLoading) {
    return <LoadingSpinner isUploading={isUploading} />;
  }

  if (message.error) {
    return (
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-full bg-linear-to-r from-red-600 to-red-700 shadow-sm">
          <Bot className="h-5 w-5 text-white" />
        </div>
        <div className="rounded-xl rounded-bl-none bg-white p-4 shadow-md border border-red-100">
          <div className="flex items-center gap-2">
            <p className="text-sm font-medium text-red-600">Произошла ошибка</p>
          </div>
          <p className="mt-1 text-sm text-red-500">{message.error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full max-w-[85%]">
      <m.div
        initial={{ opacity: 0, x: -10 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.2 }}
        className="flex items-start gap-3"
      >
        <div className="flex h-10 w-10 items-center justify-center rounded-full bg-linear-to-r from-blue-600 to-blue-700 shadow-sm">
          <Bot className="h-5 w-5 text-white" />
        </div>
        <div className="flex-1">
          <m.div
            whileHover={{ scale: 1.005 }}
            className="rounded-xl rounded-bl-none bg-white p-5 shadow-md border border-gray-100"
          >
            {message.fileResults ? (
              <FileResultsContent
                fileResults={message.fileResults}
                isFileResultWithOptions={isFileResultWithOptions}
                toggleFileTemplateVisibility={toggleFileTemplateVisibility}
                toggleFileRubricExpansion={toggleFileRubricExpansion}
                fileTemplateVisibility={fileTemplateVisibility}
                expandedFileRubrics={expandedFileRubrics}
                copyToClipboard={copyToClipboard}
              />
            ) : message.rubric ? (
              <RubricContent
                rubric={message.rubric}
                allRubrics={message.allRubrics || null}
                messageId={message.id}
                showTemplate={message.showTemplate || false}
                toggleTemplateVisibility={toggleTemplateVisibility}
                toggleRubricExpansion={toggleRubricExpansion}
                expandedRubrics={expandedRubrics}
                copyToClipboard={copyToClipboard}
              />
            ) : (
              <div className="text-gray-500 text-center py-4">
                Нет данных для отображения
              </div>
            )}
          </m.div>

          {message.rubric && (
            <div className="mt-3 flex justify-end">
              <m.button
                onClick={() =>
                  copyToClipboard(
                    formatResults(message.rubric, message.allRubrics || null),
                    message.id,
                  )
                }
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="flex items-center gap-2 rounded-lg bg-gray-100 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-200 transition-colors"
              >
                {copiedId === message.id ? (
                  <>
                    <Check className="h-4 w-4" />
                    Скопировано
                  </>
                ) : (
                  <>
                    <Copy className="h-4 w-4" />
                    Копировать все данные
                  </>
                )}
              </m.button>
            </div>
          )}
        </div>
      </m.div>
    </div>
  );
}
