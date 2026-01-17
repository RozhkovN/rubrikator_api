import { useState, useRef, useEffect, type ChangeEvent } from "react";
import { m, AnimatePresence, LazyMotion, domAnimation } from "framer-motion";
import axios from "axios";
import {
  Send,
  Copy,
  Check,
  FileText,
  User,
  Bot,
  Clock,
  BarChart3,
  ChevronDown,
  ChevronUp,
  Upload,
  X,
  File,
  MessageSquare,
  Eye,
  EyeOff,
} from "lucide-react";

interface Rubric {
  rubric_id: number;
  rubric_name: string;
  short_name: string;
  response_template: string;
  confidence: number;
}

interface APIResponse {
  text: string;
  best_match: Rubric;
  all_predictions: Rubric[] | null;
}

// Обновленный интерфейс для файлов с вариантами
interface FileResultWithOptions {
  filename: string;
  text: string;
  best_match: Rubric;
  all_predictions: Rubric[] | null;
  error: string | null;
}

// Старый интерфейс для обратной совместимости
interface SimpleFileResult {
  filename: string;
  text: string;
  rubric_id: number;
  rubric_name: string;
  short_name: string;
  response_template: string;
  confidence: number;
  error: string | null;
}

interface MultiFileResponse {
  results: (FileResultWithOptions | SimpleFileResult)[];
  total: number;
  success: number;
  failed: number;
}

type Message = {
  id: string;
  text: string;
  isUser: boolean;
  isLoading?: boolean;
  rubric?: Rubric;
  allRubrics?: Rubric[] | null;
  error?: string;
  fileName?: string;
  isFileMessage?: boolean;
  // Обновлено для поддержки обоих форматов
  fileResults?: (FileResultWithOptions | SimpleFileResult)[];
  showTemplate?: boolean;
};

type ExpandedRubric = {
  [rubricId: number]: boolean;
};

type ExpandedFileRubric = {
  [key: string]: boolean; // ключ: `${fileIndex}-${rubricId}`
};

type FileTemplateVisibility = {
  [fileIndex: number]: boolean;
};

export default function App() {
  const API_URL = import.meta.env.VITE_API_URL;
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [files, setFiles] = useState<File[]>([]);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [expandedRubrics, setExpandedRubrics] = useState<ExpandedRubric>({});
  // Новые состояния для файлов с вариантами
  const [expandedFileRubrics, setExpandedFileRubrics] =
    useState<ExpandedFileRubric>({});
  const [fileTemplateVisibility, setFileTemplateVisibility] =
    useState<FileTemplateVisibility>({});

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Адаптация для мобильных устройств
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);

  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth < 768);
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const toggleRubricExpansion = (rubricId: number) => {
    setExpandedRubrics((prev) => ({
      ...prev,
      [rubricId]: !prev[rubricId],
    }));
  };

  // Новые функции для управления файловыми рубриками
  const toggleFileRubricExpansion = (fileIndex: number, rubricId: number) => {
    const key = `${fileIndex}-${rubricId}`;
    setExpandedFileRubrics((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  const toggleFileTemplateVisibility = (fileIndex: number) => {
    setFileTemplateVisibility((prev) => ({
      ...prev,
      [fileIndex]: !prev[fileIndex],
    }));
  };

  const toggleTemplateVisibility = (messageId: string) => {
    setMessages((prev) =>
      prev.map((msg) =>
        msg.id === messageId
          ? { ...msg, showTemplate: !msg.showTemplate }
          : msg,
      ),
    );
  };

  const copyToClipboard = async (text: string, messageId: string) => {
    try {
      if (!navigator.clipboard) {
        throw new Error("Clipboard API not supported");
      }

      await navigator.clipboard.writeText(text);
      setCopiedId(messageId);
      setTimeout(() => setCopiedId(null), 2000);
    } catch (err) {
      const textArea = document.createElement("textarea");
      textArea.value = text;
      document.body.appendChild(textArea);
      textArea.select();

      try {
        const successful = document.execCommand("copy");
        if (successful) {
          setCopiedId(messageId);
          setTimeout(() => setCopiedId(null), 2000);
        }
      } finally {
        document.body.removeChild(textArea);
      }
    }
  };

  const formatResults = (
    rubric: Rubric | undefined,
    allRubrics: Rubric[] | null,
  ) => {
    if (!rubric) {
      return "Нет данных для отображения.";
    }

    let result = `Название: ${rubric.rubric_name}\n`;
    result += `Краткое название: ${rubric.short_name}\n`;
    result += `Уверенность: ${(rubric.confidence * 100).toFixed(2)}%\n\n`;
    result += `Ответ:\n${rubric.response_template}`;

    if (allRubrics && allRubrics.length > 1) {
      result += "\n\nВсе варианты:\n";
      allRubrics.forEach((r, index) => {
        result += `${index + 1}. ${r.short_name} (${(r.confidence * 100).toFixed(2)}%)\n`;
      });
    }

    return result;
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
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

  const handleSendMessage = async () => {
    const text = inputValue.trim();
    const hasFiles = files.length > 0;

    if ((!text && !hasFiles) || isLoading) return;

    const userMessageId = `user-${Date.now()}`;
    const loadingMessageId = `loading-${Date.now()}`;

    setMessages((prev) => [
      ...prev,
      {
        id: userMessageId,
        text: hasFiles ? `Файлы: ${files.map((f) => f.name).join(", ")}` : text,
        isUser: true,
        isLoading: false,
        isFileMessage: hasFiles,
        fileName: hasFiles ? files.map((f) => f.name).join(", ") : undefined,
      },
    ]);

    setInputValue("");
    setFiles([]);
    setIsLoading(true);
    setIsUploading(hasFiles);
    setExpandedRubrics({});
    setExpandedFileRubrics({});
    setFileTemplateVisibility({});

    setMessages((prev) => [
      ...prev,
      {
        id: loadingMessageId,
        text: "",
        isUser: false,
        isLoading: true,
      },
    ]);

    try {
      if (hasFiles) {
        const formData = new FormData();

        if (files.length === 1) {
          formData.append("file", files[0]);
          const response = await axios.post<APIResponse>(
            `${API_URL}/classify/file`,
            formData,
            {
              params: { top_k: 2 },
              headers: {
                "Content-Type": "multipart/form-data",
              },
            },
          );

          setTimeout(() => {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === loadingMessageId
                  ? {
                      ...msg,
                      isLoading: false,
                      rubric: response.data.best_match,
                      allRubrics: response.data.all_predictions,
                      isFileMessage: true,
                      fileName: files[0].name,
                      showTemplate: true,
                    }
                  : msg,
              ),
            );
          }, 500);
        } else {
          files.forEach((file) => {
            formData.append(`files`, file);
          });

          const response = await axios.post<MultiFileResponse>(
            `${API_URL}/classify/files`,
            formData,
            {
              params: { top_k: 2 },
              headers: {
                "Content-Type": "multipart/form-data",
              },
            },
          );

          setTimeout(() => {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === loadingMessageId
                  ? {
                      ...msg,
                      isLoading: false,
                      isFileMessage: true,
                      fileResults: response.data.results,
                      showTemplate: true,
                    }
                  : msg,
              ),
            );
          }, 500);
        }
      } else {
        const response = await axios.post<APIResponse>(
          `${API_URL}/classify`,
          {
            text: text,
            top_k: 2,
          },
          {
            headers: {
              "Content-Type": "application/json",
            },
          },
        );

        setTimeout(() => {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === loadingMessageId
                ? {
                    ...msg,
                    isLoading: false,
                    rubric: response.data.best_match,
                    allRubrics: response.data.all_predictions,
                    showTemplate: true,
                  }
                : msg,
            ),
          );
        }, 500);
      }
    } catch (error) {
      let errorMessage = "Неизвестная ошибка";

      if (axios.isAxiosError(error)) {
        if (error.response) {
          errorMessage = `Ошибка ${error.response.status}: ${
            error.response.data.message || error.message
          }`;
        } else if (error.request) {
          errorMessage = `Нет ответа от сервера. ${error.message}`;
        } else {
          errorMessage = `Ошибка запроса: ${error.message}`;
        }
      } else if (error instanceof Error) {
        errorMessage = error.message;
      }

      setTimeout(() => {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === loadingMessageId
              ? {
                  ...msg,
                  isLoading: false,
                  error: errorMessage,
                }
              : msg,
          ),
        );
      }, 500);
    } finally {
      setIsLoading(false);
      setIsUploading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const formatTemplateText = (text: string) => {
    return text
      .split("\n")
      .filter((line) => line.trim())
      .map((line) => line.trim());
  };

  // Функция для проверки типа результата файла
  const isFileResultWithOptions = (
    result: FileResultWithOptions | SimpleFileResult,
  ): result is FileResultWithOptions => {
    return "best_match" in result && "all_predictions" in result;
  };

  // Рендер контента для файла с вариантами
  const renderFileResultWithOptions = (
    result: FileResultWithOptions,
    fileIndex: number,
  ) => {
    const {
      best_match: rubric,
      all_predictions: allRubrics,
      text: fileText,
    } = result;
    const showTemplate = fileTemplateVisibility[fileIndex] || false;

    return (
      <m.div
        key={fileIndex}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: fileIndex * 0.1 }}
        className="border border-gray-200 rounded-lg overflow-hidden"
      >
        {/* Заголовок файла */}
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
          {fileText && (
            <div className="mt-2 text-sm text-gray-600 italic">
              "{fileText}"
            </div>
          )}
        </div>

        {/* Содержимое файла */}
        <div className="p-4">
          <div className="space-y-4">
            {/* Основная рубрика */}
            <div>
              <div className="text-sm font-medium text-gray-500 mb-1">
                Рубрика:
              </div>
              <div className="text-sm font-semibold text-gray-800">
                {rubric.short_name}
              </div>
            </div>

            {/* Ответ */}
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
                    className="flex items-center gap-1 text-xs text-blue-600 hover:text-blue-700"
                  >
                    {showTemplate ? (
                      <>
                        <EyeOff className="h-3 w-3" />
                        Скрыть ответ
                      </>
                    ) : (
                      <>
                        <Eye className="h-3 w-3" />
                        Показать ответ
                      </>
                    )}
                  </m.button>
                  <m.button
                    onClick={() =>
                      copyToClipboard(
                        rubric.response_template,
                        `file-${fileIndex}`,
                      )
                    }
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className="flex items-center gap-1 text-xs text-blue-600 hover:text-blue-700"
                  >
                    <Copy className="h-3 w-3" />
                    Копировать
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

            {/* Дополнительные варианты */}
            {allRubrics && allRubrics.length > 1 && (
              <div className="border-t border-gray-100 pt-3">
                <h4 className="mb-2 text-sm font-medium text-gray-700">
                  Дополнительные варианты
                </h4>
                <div className="space-y-2">
                  {allRubrics.slice(1).map((additionalRubric, rubricIndex) => {
                    const key = `${fileIndex}-${additionalRubric.rubric_id}`;
                    const isExpanded = expandedFileRubrics[key] || false;

                    return (
                      <div key={additionalRubric.rubric_id}>
                        <m.button
                          onClick={() =>
                            toggleFileRubricExpansion(
                              fileIndex,
                              additionalRubric.rubric_id,
                            )
                          }
                          className={`w-full flex items-center justify-between rounded-lg p-3 hover:bg-gray-50 transition-colors ${
                            isExpanded
                              ? "bg-gray-50 border border-gray-200"
                              : "bg-white border border-gray-100"
                          }`}
                        >
                          <div className="flex items-center gap-3">
                            <div className="flex h-6 w-6 items-center justify-center rounded-full bg-gray-300 text-gray-600">
                              <span className="text-xs font-bold">
                                {rubricIndex + 2}
                              </span>
                            </div>
                            <div className="text-left">
                              <span className="text-sm text-gray-700">
                                {additionalRubric.short_name}
                              </span>
                              <div className="text-sm text-gray-500">
                                Уверенность:{" "}
                                {(additionalRubric.confidence * 100).toFixed(1)}
                                %
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
                                    {additionalRubric.rubric_name}
                                  </div>
                                  <div className="pt-3 border-t border-blue-100">
                                    <div className="text-sm font-medium text-blue-700 mb-2">
                                      Ответ:
                                    </div>
                                    <div className="text-sm text-gray-700 leading-relaxed bg-white p-3 rounded border border-blue-100 max-h-40 overflow-y-auto">
                                      {additionalRubric.response_template}
                                    </div>
                                  </div>
                                  <div className="flex items-center justify-between text-sm text-gray-500">
                                    <div>
                                      Уверенность:{" "}
                                      <span className="font-semibold">
                                        {(
                                          additionalRubric.confidence * 100
                                        ).toFixed(1)}
                                        %
                                      </span>
                                    </div>
                                    <m.button
                                      onClick={() =>
                                        copyToClipboard(
                                          additionalRubric.response_template,
                                          `file-${fileIndex}-${additionalRubric.rubric_id}`,
                                        )
                                      }
                                      whileHover={{ scale: 1.05 }}
                                      whileTap={{ scale: 0.95 }}
                                      className="flex items-center gap-1 text-xs text-blue-600 hover:text-blue-700"
                                    >
                                      <Copy className="h-3 w-3" />
                                      Копировать
                                    </m.button>
                                  </div>
                                </div>
                              </div>
                            </m.div>
                          )}
                        </AnimatePresence>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        </div>
      </m.div>
    );
  };

  // Рендер контента для старого формата файла
  const renderSimpleFileResult = (
    result: SimpleFileResult,
    fileIndex: number,
  ) => {
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
                {(result.confidence * 100).toFixed(1)}%
              </span>
            </div>
          </div>
          {result.text && (
            <div className="mt-2 text-sm text-gray-600 italic">
              "{result.text}"
            </div>
          )}
        </div>

        <div className="p-4">
          {result.error ? (
            <div className="text-sm text-red-500 bg-red-50 p-3 rounded">
              Ошибка: {result.error}
            </div>
          ) : (
            <div className="space-y-3">
              <div>
                <div className="text-sm font-medium text-gray-500 mb-1">
                  Рубрика:
                </div>
                <div className="text-sm font-semibold text-gray-800">
                  {result.short_name}
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
                        result.response_template,
                        `file-${fileIndex}`,
                      )
                    }
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className="flex items-center gap-1 text-xs text-blue-600 hover:text-blue-700"
                  >
                    <Copy className="h-3 w-3" />
                    Копировать
                  </m.button>
                </div>
                <div className="text-sm text-gray-700 leading-relaxed bg-blue-50 p-3 rounded border border-blue-100 max-h-32 overflow-y-auto">
                  {result.response_template}
                </div>
              </div>
            </div>
          )}
        </div>
      </m.div>
    );
  };

  const renderFileResults = (
    fileResults: (FileResultWithOptions | SimpleFileResult)[],
  ) => (
    <div className="space-y-4">
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

      {fileResults.map((result, fileIndex) => {
        if (isFileResultWithOptions(result)) {
          return renderFileResultWithOptions(result, fileIndex);
        } else {
          return renderSimpleFileResult(result, fileIndex);
        }
      })}
    </div>
  );

  const renderRubricContent = (
    rubric: Rubric,
    allRubrics: Rubric[] | null,
    messageId: string,
  ) => (
    <div className="space-y-5">
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

        {/* Шаблон ответа */}
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
              {messages.find((m) => m.id === messageId)?.showTemplate ? (
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
            {messages.find((m) => m.id === messageId)?.showTemplate && (
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
                        <Copy className="h-3 w-3" />
                        Копировать ответ
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
            {allRubrics.slice(1).map((rubric, index) => (
              <div key={rubric.rubric_id}>
                <m.button
                  onClick={() => toggleRubricExpansion(rubric.rubric_id)}
                  className={`w-full flex items-center justify-between rounded-lg p-3 hover:bg-gray-50 transition-colors ${
                    expandedRubrics[rubric.rubric_id]
                      ? "bg-gray-50 border border-gray-200"
                      : "bg-white border border-gray-100"
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div className="flex h-6 w-6 items-center justify-center rounded-full bg-gray-300 text-gray-600">
                      <span className="text-xs font-bold">{index + 2}</span>
                    </div>
                    <div className="text-left">
                      <span className="text-sm text-gray-700">
                        {rubric.short_name}
                      </span>
                      <div className="text-sm text-gray-500">
                        Уверенность: {(rubric.confidence * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-gray-500">
                      {expandedRubrics[rubric.rubric_id]
                        ? "Скрыть"
                        : "Показать"}
                    </span>
                    {expandedRubrics[rubric.rubric_id] ? (
                      <ChevronUp className="h-4 w-4 text-gray-500" />
                    ) : (
                      <ChevronDown className="h-4 w-4 text-gray-500" />
                    )}
                  </div>
                </m.button>
                <AnimatePresence>
                  {expandedRubrics[rubric.rubric_id] && (
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
                                copyToClipboard(
                                  rubric.response_template,
                                  messageId + "-" + rubric.rubric_id,
                                )
                              }
                              whileHover={{ scale: 1.05 }}
                              whileTap={{ scale: 0.95 }}
                              className="flex items-center gap-1 text-xs text-blue-600 hover:text-blue-700"
                            >
                              <Copy className="h-3 w-3" />
                              Копировать
                            </m.button>
                          </div>
                        </div>
                      </div>
                    </m.div>
                  )}
                </AnimatePresence>
              </div>
            ))}
          </div>
        </m.div>
      )}
    </div>
  );

  return (
    <LazyMotion features={domAnimation}>
      <div className="flex h-screen w-screen overflow-hidden bg-gray-50">
        {/* Левая боковая рамка */}
        {!isMobile && (
          <m.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
            className="w-20 lg:w-64 bg-linear-to-b from-blue-600 to-blue-800 border-r border-blue-700"
          >
            <div className="h-full p-4 flex flex-col">
              <div className="mb-8">
                <div className="flex items-center justify-center lg:justify-start gap-2">
                  <FileText className="h-6 w-6 text-white" />
                  <span className="hidden lg:inline text-white font-semibold text-sm">
                    рубрикатор
                  </span>
                </div>
              </div>

              <div className="flex-1 flex flex-col items-center lg:items-start gap-4">
                <div className="text-white/80 text-xs uppercase tracking-wider hidden lg:block">
                  Статистика
                </div>
                <div className="w-full p-3 rounded-lg bg-white/10 backdrop-blur-sm">
                  <div className="text-white text-sm font-medium mb-1">
                    Анализов
                  </div>
                  <div className="text-white text-2xl font-bold">
                    {messages.filter((m) => !m.isUser && !m.isLoading).length}
                  </div>
                </div>

                <div className="w-full p-3 rounded-lg bg-white/10 backdrop-blur-sm">
                  <div className="text-white text-sm font-medium mb-1">
                    Файлов
                  </div>
                  <div className="text-white text-2xl font-bold">
                    {messages.filter((m) => m.isFileMessage).length}
                  </div>
                </div>
              </div>
            </div>
          </m.div>
        )}

        <div className="flex flex-1 flex-col">
          <m.header
            initial={{ y: -50, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5 }}
            className="sticky top-0 z-20 bg-white border-b border-gray-200 shadow-sm"
          >
            <div className="mx-auto max-w-4xl px-4 py-4 sm:px-6">
              <div className="flex items-center justify-center">
                <div className="flex items-center gap-3">
                  <m.div
                    animate={{
                      rotate: [0, 5, -5, 0],
                    }}
                    transition={{
                      repeat: Infinity,
                      duration: 8,
                      ease: "easeInOut",
                    }}
                  >
                    <FileText className="h-8 w-8 text-blue-600" />
                  </m.div>
                  <div>
                    <h1 className="text-2xl font-bold text-gray-800 sm:text-3xl">
                      Рубрикатор обращений
                    </h1>
                    <p className="text-center text-sm text-gray-500 mt-1">
                      Автоматическая классификация
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </m.header>

          <main className="flex flex-1 flex-col overflow-hidden">
            <div className="flex-1 overflow-y-auto p-4 sm:p-6">
              <div className="mx-auto max-w-4xl">
                <AnimatePresence mode="popLayout">
                  {messages.length === 0 ? (
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
                          <span className="font-medium">
                            Пример использования:
                          </span>
                        </div>
                        <p className="text-sm text-gray-700 italic">
                          "Банк заблокировал мою карту без предупреждения
                          согласно 161-ФЗ"
                        </p>
                      </m.div>
                    </m.div>
                  ) : (
                    <div className="space-y-6">
                      {messages.map((message) => (
                        <m.div
                          key={message.id}
                          layout
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: -10 }}
                          transition={{ duration: 0.3 }}
                          className={`flex ${message.isUser ? "justify-end" : "justify-start"}`}
                        >
                          {message.isUser ? (
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
                                    <p className="text-lg text-white/95">
                                      {message.text}
                                    </p>
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
                          ) : message.isLoading ? (
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
                                    {isUploading
                                      ? "Анализ файлов..."
                                      : "Анализ обращения..."}
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
                          ) : message.error ? (
                            <div className="flex items-center gap-3">
                              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-linear-to-r from-red-600 to-red-700 shadow-sm">
                                <Bot className="h-5 w-5 text-white" />
                              </div>
                              <div className="rounded-xl rounded-bl-none bg-white p-4 shadow-md border border-red-100">
                                <div className="flex items-center gap-2">
                                  <p className="text-sm font-medium text-red-600">
                                    Произошла ошибка
                                  </p>
                                </div>
                                <p className="mt-1 text-sm text-red-500">
                                  {message.error}
                                </p>
                              </div>
                            </div>
                          ) : (
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
                                    <div className="flex items-center justify-between mb-4 pb-3 border-b border-gray-100">
                                      <h3 className="text-2xl font-bold text-gray-800">
                                        {message.isFileMessage
                                          ? "Результаты анализа файлов"
                                          : "Результат классификации"}
                                      </h3>
                                      {message.rubric && (
                                        <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-blue-50">
                                          <BarChart3 className="h-4 w-4 text-blue-600" />
                                          <span className="text-md font-bold text-blue-700">
                                            {(
                                              message.rubric.confidence * 100
                                            ).toFixed(1)}
                                            %
                                          </span>
                                        </div>
                                      )}
                                    </div>

                                    {message.fileResults ? (
                                      renderFileResults(message.fileResults)
                                    ) : message.rubric ? (
                                      renderRubricContent(
                                        message.rubric,
                                        message.allRubrics || null,
                                        message.id,
                                      )
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
                                            formatResults(
                                              message.rubric,
                                              message.allRubrics || null,
                                            ),
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
                          )}
                        </m.div>
                      ))}
                      <div ref={messagesEndRef} />
                    </div>
                  )}
                </AnimatePresence>
              </div>
            </div>

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
                      className="absolute left-4 -top-7 text-xs text-gray-500"
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
          </main>
        </div>

        {!isMobile && (
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
                  <span className="hidden lg:inline text-gray-700 font-semibold text-sm">
                    Информация
                  </span>
                </div>
              </div>

              <div className="flex-1">
                <div className="mb-6">
                  <div className="text-gray-500 text-xs uppercase tracking-wider mb-3 hidden lg:block">
                    Как работает
                  </div>
                  <div className="space-y-3">
                    <div className="p-3 rounded-lg bg-blue-50/50 border border-blue-100">
                      <div className="text-sm font-medium text-blue-700 mb-1">
                        📝 Текстовый ввод
                      </div>
                      <div className="text-xs text-gray-600">
                        Введите текст обращения напрямую
                      </div>
                    </div>

                    <div className="p-3 rounded-lg bg-blue-50/50 border border-blue-100">
                      <div className="text-sm font-medium text-blue-700 mb-1">
                        📎 Загрузка файлов
                      </div>
                      <div className="text-xs text-gray-600">
                        Загрузите один или несколько файлов
                      </div>
                    </div>

                    <div className="p-3 rounded-lg bg-blue-50/50 border border-blue-100">
                      <div className="text-sm font-medium text-blue-700 mb-1">
                        📋 Готовый ответ
                      </div>
                      <div className="text-xs text-gray-600">
                        Получите ответ для выбранной рубрики
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </m.div>
        )}
      </div>
    </LazyMotion>
  );
}
