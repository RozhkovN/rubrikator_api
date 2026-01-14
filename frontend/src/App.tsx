import { useState, useRef, useEffect } from "react";
import { m, AnimatePresence } from "framer-motion";
import axios from "axios";

interface Rubric {
  rubric_id: number;
  rubric_name: string;
  short_name: string;
  confidence: number;
}

interface APIResponse {
  text: string;
  best_match: Rubric;
  all_predictions: Rubric[] | null;
}

type Message = {
  id: string;
  text: string;
  isUser: boolean;
  isLoading?: boolean;
  rubric?: Rubric;
  allRubrics?: Rubric[] | null;
  error?: string;
};

export default function App() {
  const API_URL = import.meta.env.VITE_API_URL;
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

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
    result += `Уверенность: ${(rubric.confidence * 100).toFixed(2)}%\n`;

    if (allRubrics && allRubrics.length > 1) {
      result += "\nВсе варианты:\n";
      allRubrics.forEach((r, index) => {
        result += `${index + 1}. ${r.short_name} (${(r.confidence * 100).toFixed(2)}%)\n`;
      });
    }

    return result;
  };

  const handleSendMessage = async () => {
    const text = inputValue.trim();
    if (!text || isLoading) return;

    const userMessageId = `user-${Date.now()}`;
    const loadingMessageId = `loading-${Date.now()}`;

    // Добавляем сообщение пользователя
    setMessages((prev) => [
      ...prev,
      {
        id: userMessageId,
        text: text,
        isUser: true,
        isLoading: false,
      },
    ]);

    // Очищаем поле ввода
    setInputValue("");
    setIsLoading(true);

    // Добавляем сообщение загрузки
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
      const response = await axios.post<APIResponse>(
        `${API_URL}/classify`,
        {
          text: text,
          top_k: 1,
        },
        {
          headers: {
            "Content-Type": "application/json",
          },
        },
      );

      // Обновляем сообщение загрузки с результатами
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === loadingMessageId
            ? {
                ...msg,
                isLoading: false,
                rubric: response.data.best_match,
                allRubrics: response.data.all_predictions,
              }
            : msg,
        ),
      );
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
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-gray-50">
      {/* Скрываем боковые рамки на телефоне */}
      {!isMobile && <div className="w-16 bg-indigo-50 md:w-32 lg:w-72"></div>}

      {/* Основное содержимое */}
      <div className="flex flex-1 flex-col">
        <header className="sticky top-0 z-10 bg-white shadow-sm">
          <div className="mx-auto flex max-w-4xl justify-center px-4 py-4 sm:px-6">
            <m.h1
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.9 }}
              className="text-lg font-bold text-indigo-600 sm:text-xl md:text-3xl"
            >
              Классификатор обращений
            </m.h1>
          </div>
        </header>

        <main className="flex flex-1 flex-col overflow-hidden">
          <div className="flex-1 space-y-4 overflow-y-auto p-2 sm:p-4">
            <AnimatePresence>
              {messages.length === 0 ? (
                <m.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex h-full flex-col items-center justify-center p-4 text-center sm:p-8"
                >
                  <m.div
                    animate={{
                      scale: [1, 1.1, 1],
                      rotate: [0, 5, -5, 0],
                    }}
                    transition={{
                      repeat: Infinity,
                      duration: 4,
                      ease: "easeInOut",
                    }}
                    className="mb-4 sm:mb-6"
                  >
                    <svg
                      className="h-16 w-16 text-indigo-400 sm:h-20 sm:w-20"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2z"
                      />
                    </svg>
                  </m.div>
                  <h2 className="mb-2 text-xl font-bold text-gray-700 sm:text-2xl">
                    Добро пожаловать в Классификатор обращений
                  </h2>
                  <p className="max-w-md text-sm text-gray-500 sm:text-base">
                    Этот инструмент поможет классифицировать ваши обращения по
                    рубрикам. Введите текст обращения для анализа.
                  </p>
                </m.div>
              ) : (
                messages.map((message) => (
                  <m.div
                    key={message.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                    className={`group relative flex ${message.isUser ? "justify-end" : "justify-start"}`}
                  >
                    {message.isUser ? (
                      <m.div
                        whileHover={{ scale: isMobile ? 1 : 1.01 }}
                        className="max-w-[90%] rounded-2xl rounded-tr-none bg-indigo-600 p-3 text-white shadow-md sm:max-w-3xl sm:p-4"
                      >
                        <span className="text-sm sm:text-base">
                          {message.text}
                        </span>
                      </m.div>
                    ) : message.isLoading ? (
                      <m.div className="flex max-w-[90%] items-center space-x-2 rounded-2xl rounded-tl-none bg-white p-3 shadow-md sm:max-w-3xl sm:p-4">
                        <m.div
                          animate={{ rotate: 360 }}
                          transition={{
                            repeat: Infinity,
                            duration: 1,
                            ease: "linear",
                          }}
                          className="h-4 w-4 rounded-full border-2 border-indigo-600 border-t-transparent sm:h-5 sm:w-5"
                        />
                        <span className="text-sm sm:text-base">
                          Классифицируем...
                        </span>
                      </m.div>
                    ) : message.error ? (
                      <div className="max-w-[90%] rounded-2xl rounded-tl-none bg-white p-3 text-sm text-red-500 shadow-md sm:max-w-3xl sm:p-4 sm:text-base">
                        Ошибка: {message.error}
                      </div>
                    ) : (
                      <div className="w-full max-w-[90%] sm:max-w-3xl">
                        <m.div
                          whileHover={{ scale: isMobile ? 1 : 1.01 }}
                          className="rounded-2xl rounded-tl-none bg-white p-4 shadow-md sm:p-6"
                        >
                          <h3 className="mb-3 text-base font-bold text-indigo-600 sm:mb-4 sm:text-lg">
                            Результат классификации:
                          </h3>

                          {message.rubric && (
                            <div className="space-y-4">
                              {/* Основная информация о рубрике */}
                              <div className="space-y-4">
                                <div>
                                  <div className="text-sm font-medium text-gray-500 sm:text-base">
                                    Краткое название
                                  </div>
                                  <div className="text-base font-semibold text-gray-800 sm:text-lg">
                                    {message.rubric.short_name}
                                  </div>
                                </div>

                                <div>
                                  <div className="text-sm font-medium text-gray-500 sm:text-base">
                                    Полное название
                                  </div>
                                  <div className="text-sm text-gray-700 sm:text-base">
                                    {message.rubric.rubric_name}
                                  </div>
                                </div>

                                <div className="grid grid-cols-2 gap-4">
                                  <div>
                                    <div className="text-sm font-medium text-gray-500 sm:text-base">
                                      ID рубрики
                                    </div>
                                    <div className="text-base font-bold text-indigo-600 sm:text-lg">
                                      {message.rubric.rubric_id}
                                    </div>
                                  </div>

                                  <div>
                                    <div className="text-sm font-medium text-gray-500 sm:text-base">
                                      Уверенность
                                    </div>
                                    <div className="text-base font-bold text-indigo-600 sm:text-lg">
                                      {(
                                        message.rubric.confidence * 100
                                      ).toFixed(2)}
                                      %
                                    </div>
                                  </div>
                                </div>
                              </div>

                              {/* Все предсказания, если есть */}
                              {message.allRubrics &&
                                message.allRubrics.length > 1 && (
                                  <div className="border-t border-gray-200 pt-4">
                                    <h4 className="mb-2 text-sm font-medium text-gray-700 sm:text-base">
                                      Все варианты:
                                    </h4>
                                    <div className="space-y-2">
                                      {message.allRubrics.map(
                                        (rubric, index) => (
                                          <div
                                            key={rubric.rubric_id}
                                            className={`flex items-center justify-between rounded p-2 ${index === 0 ? "bg-indigo-50" : "bg-gray-50"}`}
                                          >
                                            <div className="flex items-center">
                                              <span className="mr-2 text-sm font-medium text-gray-600">
                                                {index + 1}.
                                              </span>
                                              <span className="text-sm text-gray-700">
                                                {rubric.short_name}
                                              </span>
                                            </div>
                                            <span className="text-sm font-medium text-gray-600">
                                              {(
                                                rubric.confidence * 100
                                              ).toFixed(2)}
                                              %
                                            </span>
                                          </div>
                                        ),
                                      )}
                                    </div>
                                  </div>
                                )}
                            </div>
                          )}
                        </m.div>

                        {/* Кнопка копирования */}
                        <div className="mt-1 flex justify-start sm:mt-2">
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
                            className="rounded-lg bg-indigo-100 p-1 text-indigo-700 hover:bg-indigo-200 focus:outline-none sm:p-2"
                            whileHover={{ scale: isMobile ? 1 : 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            title="Копировать результат"
                          >
                            {copiedId === message.id ? (
                              <svg
                                className="h-4 w-4 text-green-500 sm:h-5 sm:w-5"
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                              >
                                <path
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                  strokeWidth={2}
                                  d="M5 13l4 4L19 7"
                                />
                              </svg>
                            ) : (
                              <svg
                                className="h-4 w-4 sm:h-5 sm:w-5"
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                              >
                                <path
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                  strokeWidth={2}
                                  d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3"
                                />
                              </svg>
                            )}
                          </m.button>
                        </div>
                      </div>
                    )}
                  </m.div>
                ))
              )}
              <div ref={messagesEndRef} />
            </AnimatePresence>
          </div>

          {/* Область ввода */}
          <div className="border-t border-gray-200 bg-white p-2 sm:p-4">
            <div className="mx-auto max-w-4xl">
              <div className="relative">
                <m.textarea
                  ref={inputRef}
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Введите текст обращения для классификации..."
                  className="w-full resize-none rounded-xl border border-gray-300 p-3 pr-12 text-sm transition-all focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 sm:p-4 sm:pr-16 sm:text-base"
                  rows={isMobile ? 2 : 3}
                  whileFocus={{
                    boxShadow: "0 0 0 3px rgba(99, 102, 241, 0.3)",
                  }}
                />

                {/* Кнопка отправки */}
                <m.button
                  onClick={handleSendMessage}
                  disabled={isLoading || !inputValue}
                  className="absolute right-2 bottom-2 rounded-lg bg-indigo-600 p-1 text-white transition-colors hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50 sm:right-3 sm:bottom-3 sm:p-2"
                  whileHover={{ scale: isLoading ? 1 : 1.1 }}
                  whileTap={{ scale: isLoading ? 1 : 0.95 }}
                >
                  {isLoading ? (
                    <m.div
                      animate={{ rotate: 360 }}
                      transition={{ repeat: Infinity, duration: 1 }}
                      className="h-4 w-4 rounded-full border-2 border-white border-t-transparent sm:h-5 sm:w-5"
                    />
                  ) : (
                    <svg
                      className="h-4 w-4 sm:h-5 sm:w-5"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                      />
                    </svg>
                  )}
                </m.button>
              </div>

              <p className="mt-1 text-center text-xs text-gray-500 sm:mt-2">
                Нажмите Enter для отправки, Shift+Enter для новой строки
              </p>
            </div>
          </div>
        </main>
      </div>
      {!isMobile && <div className="w-16 bg-indigo-50 md:w-32 lg:w-72"></div>}
    </div>
  );
}
