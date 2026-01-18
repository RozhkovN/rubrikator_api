import type {
  FileResultWithOptions,
  SimpleFileResult,
  ExpandedFileRubric,
  FileTemplateVisibility,
} from "../../types/chat";
import { FileResultCard } from "./FileResultCard";

interface FileResultsContentProps {
  fileResults: (FileResultWithOptions | SimpleFileResult)[];
  isFileResultWithOptions: (
    result: FileResultWithOptions | SimpleFileResult,
  ) => result is FileResultWithOptions;
  toggleFileTemplateVisibility: (fileIndex: number) => void;
  toggleFileRubricExpansion: (fileIndex: number, rubricId: number) => void;
  fileTemplateVisibility: FileTemplateVisibility;
  expandedFileRubrics: ExpandedFileRubric;
  copyToClipboard: (text: string, messageId: string) => void;
}

export function FileResultsContent({
  fileResults,
  isFileResultWithOptions,
  toggleFileTemplateVisibility,
  toggleFileRubricExpansion,
  fileTemplateVisibility,
  expandedFileRubrics,
  copyToClipboard,
}: FileResultsContentProps) {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-gray-100">
        <h3 className="text-2xl font-bold text-gray-800">
          Результаты анализа файлов
        </h3>
      </div>
      <div className="grid grid-cols-3 gap-2 mb-4 p-3 bg-gray-50 rounded-lg">
        <div className="text-center">
          <div className="text-lg font-bold text-blue-600">
            {fileResults.length}
          </div>
          <div className="text-xs text-gray-600">Всего файлов</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-green-600">
            {fileResults.filter((f) => !f.error).length}
          </div>
          <div className="text-xs text-gray-600">Успешно</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-red-600">
            {fileResults.filter((f) => f.error).length}
          </div>
          <div className="text-xs text-gray-600">С ошибками</div>
        </div>
      </div>

      {fileResults.map((result, fileIndex) => (
        <FileResultCard
          key={fileIndex}
          result={result}
          fileIndex={fileIndex}
          isFileResultWithOptions={isFileResultWithOptions}
          toggleFileTemplateVisibility={toggleFileTemplateVisibility}
          toggleFileRubricExpansion={toggleFileRubricExpansion}
          fileTemplateVisibility={fileTemplateVisibility}
          expandedFileRubrics={expandedFileRubrics}
          copyToClipboard={copyToClipboard}
        />
      ))}
    </div>
  );
}
