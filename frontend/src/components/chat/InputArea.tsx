import { m } from "framer-motion";
import { Send, Upload, X, File } from "lucide-react";
import React from "react";

interface InputAreaProps {
  inputValue: string;
  setInputValue: (value: string) => void;
  handleSendMessage: () => void;
  isLoading: boolean;
  isMobile: boolean;
  files: File[];
  setFiles: React.Dispatch<React.SetStateAction<File[]>>;
  isUploading: boolean;
}

export function InputArea({
  inputValue,
  setInputValue,
  handleSendMessage,
  isLoading,
  isMobile,
  files,
  setFiles,
}: InputAreaProps) {
  const inputRef = React.useRef<HTMLTextAreaElement>(null);
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  // Обработка загрузки файлов
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selectedFiles = Array.from(e.target.files);
      setFiles((prev) => [...prev, ...selectedFiles]);
    }
    e.target.value = "";
  };

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="border-t border-gray-200 bg-white p-4">
      <m.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mx-auto max-w-4xl"
      >
        {files.length > 0 && (
          <m.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-3"
          >
            <div className="text-sm font-medium text-gray-700 mb-2">
              Выбранные файлы ({files.length}):
            </div>
            <div className="space-y-2 max-h-32 overflow-y-auto">
              {files.map((file, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between bg-blue-50 rounded-lg p-2 border border-blue-100"
                >
                  <div className="flex items-center gap-2">
                    <File className="h-4 w-4 text-blue-600" />
                    <span className="text-sm text-gray-700 truncate max-w-xs">
                      {file.name}
                    </span>
                    <span className="text-xs text-gray-500">
                      ({(file.size / 1024).toFixed(1)} KB)
                    </span>
                  </div>
                  <button
                    onClick={() => removeFile(index)}
                    className="p-1 hover:bg-blue-100 rounded-full transition-colors"
                  >
                    <X className="h-4 w-4 text-gray-500" />
                  </button>
                </div>
              ))}
            </div>
          </m.div>
        )}

        <div className="relative">
          <m.textarea
            ref={inputRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Введите текст обращения или загрузите файлы..."
            className="w-full resize-none rounded-xl border border-gray-300 bg-white p-4 pr-36 text-gray-800 transition-all duration-200 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 focus:outline-none"
            rows={isMobile ? 2 : 3}
            disabled={isLoading}
          />

          {inputValue.length > 0 && (
            <m.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              className="absolute left-4 -top-10 text-xs text-gray-500"
            >
              {inputValue.length} символов
            </m.div>
          )}

          <m.button
            onClick={handleUploadClick}
            disabled={isLoading}
            whileHover={!isLoading ? { scale: 1.05 } : {}}
            whileTap={!isLoading ? { scale: 0.95 } : {}}
            className={`absolute right-20 bottom-3 flex h-10 items-center gap-2 rounded-lg px-3 shadow-sm transition-all duration-200 ${
              isLoading
                ? "bg-gray-300 cursor-not-allowed"
                : "bg-gray-100 hover:bg-gray-200"
            }`}
          >
            <Upload className="h-4 w-4 text-gray-700" />
            <span className="text-sm text-gray-700">
              {isMobile ? "Файл" : "Загрузить"}
            </span>
          </m.button>

          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            className="hidden"
            multiple
            accept=".txt,.pdf,.doc,.docx,.rtf"
          />

          <m.button
            onClick={handleSendMessage}
            disabled={isLoading || (!inputValue && files.length === 0)}
            whileHover={
              !isLoading && (inputValue || files.length > 0)
                ? { scale: 1.1 }
                : {}
            }
            whileTap={
              !isLoading && (inputValue || files.length > 0)
                ? { scale: 0.9 }
                : {}
            }
            className={`absolute right-3 bottom-3 flex h-12 w-12 items-center justify-center rounded-lg shadow-sm transition-all duration-200 ${
              isLoading || (!inputValue && files.length === 0)
                ? "bg-gray-300 cursor-not-allowed"
                : "bg-blue-600 hover:bg-blue-700 hover:shadow-md"
            }`}
          >
            {isLoading ? (
              <m.div
                animate={{ rotate: 360 }}
                transition={{
                  repeat: Infinity,
                  duration: 1,
                  ease: "linear",
                }}
                className="h-6 w-6 rounded-full border-2 border-white border-t-transparent"
              />
            ) : (
              <Send className="h-5 w-5 text-white" />
            )}
          </m.button>
        </div>

        <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
          <p className="text-xs text-gray-500">
            Нажмите Enter для отправки, Shift+Enter для новой строки
          </p>
          <div className="text-xs text-gray-500">
            Поддерживаемые форматы: .pdf, .doc, .docx
          </div>
        </div>
      </m.div>
    </div>
  );
}
