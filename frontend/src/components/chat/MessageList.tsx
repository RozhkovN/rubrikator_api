import { useRef, useEffect } from "react";
import { m, AnimatePresence } from "framer-motion";
import type { Message } from "../../types/chat";
import { MessageBubble } from "./MessageBubble";
import { WelcomeScreen } from "../WelcomeScreen";

interface MessageListProps {
  messages: Message[];
  isUploading: boolean;
  toggleTemplateVisibility: (messageId: string) => void;
  toggleRubricExpansion: (rubricId: number) => void;
  toggleFileTemplateVisibility: (fileIndex: number) => void;
  toggleFileRubricExpansion: (fileIndex: number, rubricId: number) => void;
  expandedRubrics: { [rubricId: number]: boolean };
  expandedFileRubrics: { [key: string]: boolean };
  fileTemplateVisibility: { [fileIndex: number]: boolean };
}

export function MessageList({
  messages,
  isUploading,
  toggleTemplateVisibility,
  toggleRubricExpansion,
  toggleFileTemplateVisibility,
  toggleFileRubricExpansion,
  expandedRubrics,
  expandedFileRubrics,
  fileTemplateVisibility,
}: MessageListProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Скролл к низу при обновлении сообщений
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div className="flex-1 overflow-y-auto p-4 sm:p-6">
      <div className="mx-auto max-w-4xl">
        <AnimatePresence mode="popLayout">
          {messages.length === 0 ? (
            <WelcomeScreen />
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
                  className={`flex ${
                    message.isUser ? "justify-end" : "justify-start"
                  }`}
                >
                  <MessageBubble
                    message={message}
                    isUploading={isUploading}
                    toggleTemplateVisibility={toggleTemplateVisibility}
                    toggleRubricExpansion={toggleRubricExpansion}
                    toggleFileTemplateVisibility={toggleFileTemplateVisibility}
                    toggleFileRubricExpansion={toggleFileRubricExpansion}
                    expandedRubrics={expandedRubrics}
                    expandedFileRubrics={expandedFileRubrics}
                    fileTemplateVisibility={fileTemplateVisibility}
                  />
                </m.div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
