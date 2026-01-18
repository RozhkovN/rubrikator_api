import { useState } from "react";
import axios from "axios";
import type {
  Message,
  APIResponse,
  MultiFileResponse,
  ExpandedRubric,
  ExpandedFileRubric,
  FileTemplateVisibility,
} from "../types/chat";

export function useChat() {
  const API_URL = import.meta.env.VITE_API_URL;

  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [files, setFiles] = useState<File[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [expandedRubrics, setExpandedRubrics] = useState<ExpandedRubric>({});
  const [expandedFileRubrics, setExpandedFileRubrics] =
    useState<ExpandedFileRubric>({});
  const [fileTemplateVisibility, setFileTemplateVisibility] =
    useState<FileTemplateVisibility>({});

  // Переключение раскрытия рубрики
  const toggleRubricExpansion = (rubricId: number) => {
    setExpandedRubrics((prev) => ({
      ...prev,
      [rubricId]: !prev[rubricId],
    }));
  };

  // Переключение раскрытия рубрики файла
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

    // Сброс значений
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
            // Инициализируем видимость шаблонов для всех файлов
            const newFileTemplateVisibility: FileTemplateVisibility = {};
            response.data.results.forEach((_, index) => {
              newFileTemplateVisibility[index] = true;
            });
            setFileTemplateVisibility(newFileTemplateVisibility);

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

      // Отображение ошибки
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

  return {
    messages,
    setMessages,
    inputValue,
    setInputValue,
    isLoading,
    isUploading,
    files,
    setFiles,
    handleSendMessage,
    toggleRubricExpansion,
    toggleFileRubricExpansion,
    toggleFileTemplateVisibility,
    toggleTemplateVisibility,
    expandedRubrics,
    expandedFileRubrics,
    fileTemplateVisibility,
    totalRequests: messages.filter((m) => !m.isUser && !m.isLoading).length,
    fileMessages: messages.filter((m) => m.isFileMessage).length,
  };
}
