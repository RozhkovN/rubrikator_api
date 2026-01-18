import { useState, useEffect } from "react";
import { LazyMotion, domAnimation } from "framer-motion";
import { ChatContainer } from "./components/chat/ChatContainer";
import { LeftSidebar } from "./components/sidebars/LeftSidebar";
import { RightSidebar } from "./components/sidebars/RightSidebar";
import { useChat } from "./hooks/useChat";

export default function App() {
  const { totalRequests, fileMessages } = useChat();

  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
  const [showStatistics, setShowStatistics] = useState(false);

  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth < 768);
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const toggleStatistics = () => {
    setShowStatistics(!showStatistics);
  };

  return (
    <LazyMotion features={domAnimation}>
      <div className="flex h-screen w-screen overflow-hidden bg-gray-50">
        <LeftSidebar
          isMobile={isMobile}
          showStatistics={showStatistics}
          toggleStatistics={toggleStatistics}
          totalRequests={totalRequests}
          fileMessages={fileMessages}
        />

        <ChatContainer />

        <RightSidebar isMobile={isMobile} />
      </div>
    </LazyMotion>
  );
}
