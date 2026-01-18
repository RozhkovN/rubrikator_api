import { useState } from "react";

export function useClipboard() {
  const [copiedId, setCopiedId] = useState<string | null>(null);

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

  return { copiedId, copyToClipboard };
}
