import { ChatContainer } from "@/components/ChatContainer";

export default function HomePage() {
  return (
    <div
      suppressHydrationWarning
      className="flex min-h-[100dvh] w-full items-center justify-center sm:p-4"
    >
      <ChatContainer />
    </div>
  );
}
