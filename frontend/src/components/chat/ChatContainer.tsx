import { m } from "framer-motion";
import { FileText } from "lucide-react";
import { useChat } from "../../hooks/useChat";
import { MessageList } from "./MessageList";
import { InputArea } from "./InputArea";
import { useState, useEffect } from "react";

export function ChatContainer() {
  const {
    messages,
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
  } = useChat();

  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);

  // Изменение размера экрана для мобильных устройств
  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth < 768);
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <div className="flex flex-1 flex-col relative">
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
                <p className="text-center text-md text-gray-500 mt-1">
                  Автоматическая классификация
                </p>
              </div>
            </div>
          </div>
        </div>
      </m.header>

      <main className="flex flex-1 flex-col overflow-hidden">
        <MessageList
          messages={messages}
          isUploading={isUploading}
          toggleTemplateVisibility={toggleTemplateVisibility}
          toggleRubricExpansion={toggleRubricExpansion}
          toggleFileTemplateVisibility={toggleFileTemplateVisibility}
          toggleFileRubricExpansion={toggleFileRubricExpansion}
          expandedRubrics={expandedRubrics}
          expandedFileRubrics={expandedFileRubrics}
          fileTemplateVisibility={fileTemplateVisibility}
        />

        <InputArea
          inputValue={inputValue}
          setInputValue={setInputValue}
          handleSendMessage={handleSendMessage}
          isLoading={isLoading}
          isMobile={isMobile}
          files={files}
          setFiles={setFiles}
          isUploading={isUploading}
        />
      </main>
    </div>
  );
}
