"use client";

import * as React from "react";
import { motion } from "framer-motion";
import { RotateCcw, SendHorizontal, ShieldCheck } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { DraftTransactionCard, formatVnd } from "@/components/DraftTransactionCard";
import { AuthPinModal } from "@/components/AuthPinModal";
import {
  cancelDraftTransaction,
  resetDemoAccount,
  sendChatMessage,
  type ConfirmApiResponse,
  type DraftObject,
} from "@/services/demoService";

type ChatItem =
  | { id: string; kind: "user"; text: string }
  | { id: string; kind: "draft"; draft: DraftObject };

const QUICK_COMMANDS = [
  "Chuyển 500k cho thằng Nam",
  "Chuyển 200k cho sếp",
  "Chuyển 1 triệu cho Minh",
];

/**
 * ChatContainer — Khung chat Mobile-first (~430px):
 * - Header chứa nút Reset Demo
 * - Danh sách tin nhắn user + DraftTransactionCard (tuyệt đối không render text của LLM)
 * - Gợi ý lệnh nhanh + thanh nhập lệnh
 */
export function ChatContainer() {
  const [items, setItems] = React.useState<ChatItem[]>([]);
  const [inputValue, setInputValue] = React.useState("");
  const [isSending, setIsSending] = React.useState(false);
  const [isResetting, setIsResetting] = React.useState(false);
  const [cancellingTxId, setCancellingTxId] = React.useState<string | null>(null);
  const [activeDraft, setActiveDraft] = React.useState<DraftObject | null>(null);
  const [pinModalOpen, setPinModalOpen] = React.useState(false);

  const listEndRef = React.useRef<HTMLDivElement>(null);
  const msgCounterRef = React.useRef(0);

  React.useEffect(() => {
    listEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [items, isSending]);

  const handleSend = async (rawText: string) => {
    const message = rawText.trim();
    if (!message || isSending) return;

    msgCounterRef.current += 1;
    const userMsgId = `u_${msgCounterRef.current}`;
    setItems((prev) => [...prev, { id: userMsgId, kind: "user", text: message }]);
    setInputValue("");
    setIsSending(true);

    try {
      const res = await sendChatMessage(message);
      if (res.draft) {
        setItems((prev) => [
          ...prev,
          { id: `d_${res.draft!.tx_id}`, kind: "draft", draft: res.draft! },
        ]);
      } else if (res.decision === "CLARIFY") {
        toast.info("Cần làm rõ thông tin người nhận hoặc số tiền.");
      } else if (res.decision === "BLOCK_ANOMALY") {
        toast.error("Giao dịch bị chặn bởi lớp kiểm duyệt an toàn.");
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Không thể gửi yêu cầu";
      toast.error(msg);
    } finally {
      setIsSending(false);
    }
  };

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    void handleSend(inputValue);
  };

  const handleOpenConfirm = (draft: DraftObject) => {
    setActiveDraft(draft);
    setPinModalOpen(true);
  };

  const handleConfirmSuccess = (res: ConfirmApiResponse) => {
    setItems((prev) =>
      prev.map((item) =>
        item.kind === "draft" && item.draft.tx_id === res.tx_id
          ? { ...item, draft: { ...item.draft, status: "COMPLETED" } }
          : item
      )
    );
  };

  const handleCancelDraft = async (draft: DraftObject) => {
    setCancellingTxId(draft.tx_id);
    try {
      const res = await cancelDraftTransaction(draft.tx_id);
      setItems((prev) =>
        prev.map((item) =>
          item.kind === "draft" && item.draft.tx_id === res.tx_id
            ? { ...item, draft: { ...item.draft, status: "CANCELLED" } }
            : item
        )
      );
      toast.info("Đã hủy bản nháp giao dịch.");
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Không thể hủy giao dịch";
      toast.error(msg);
    } finally {
      setCancellingTxId(null);
    }
  };

  const handleResetDemo = async () => {
    if (isResetting) return;
    setIsResetting(true);
    try {
      const res = await resetDemoAccount();
      setItems([]);
      setActiveDraft(null);
      setPinModalOpen(false);
      toast.success(`Đã reset tài khoản Demo (${formatVnd(res.balance)})`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Reset tài khoản Demo thất bại";
      toast.error(msg);
    } finally {
      setIsResetting(false);
    }
  };

  return (
    <div className="flex h-[100dvh] w-full max-w-[430px] flex-col overflow-hidden bg-zinc-50 sm:h-[820px] sm:max-h-[92dvh] sm:rounded-3xl sm:border sm:border-zinc-200 sm:shadow-xl">
      {/* Header */}
      <header className="flex items-center justify-between border-b border-zinc-200 bg-white px-4 py-3">
        <div className="flex items-center gap-2">
          <div className="flex size-8 items-center justify-center rounded-xl bg-[#A50064] text-white">
            <ShieldCheck className="size-4" />
          </div>
          <h1 className="text-sm font-bold tracking-tight text-zinc-900">
            SafePay Agent
          </h1>
        </div>

        <Button
          type="button"
          variant="outline"
          size="sm"
          className="h-8 gap-1.5 rounded-full px-3 text-xs"
          disabled={isResetting}
          onClick={handleResetDemo}
        >
          <RotateCcw className={`size-3.5 ${isResetting ? "animate-spin" : ""}`} />
          Reset Demo
        </Button>
      </header>

      {/* Message List (chỉ hiển thị tin nhắn user + DraftTransactionCard từ DB) */}
      <main className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
        {items.map((item) =>
          item.kind === "user" ? (
            <motion.div
              key={item.id}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.16 }}
              className="flex justify-end"
            >
              <div className="max-w-[82%] rounded-2xl rounded-br-sm bg-[#A50064] px-3.5 py-2 text-sm text-white shadow-xs">
                {item.text}
              </div>
            </motion.div>
          ) : (
            <div key={item.id} className="flex justify-start">
              <DraftTransactionCard
                draft={item.draft}
                onConfirmClick={handleOpenConfirm}
                onCancelClick={handleCancelDraft}
                isCancelling={cancellingTxId === item.draft.tx_id}
              />
            </div>
          )
        )}

        {isSending && (
          <div className="flex justify-start">
            <div className="rounded-2xl border border-zinc-200 bg-white px-3.5 py-2 text-xs text-zinc-500">
              Đang tạo bản nháp từ DB...
            </div>
          </div>
        )}

        <div ref={listEndRef} />
      </main>

      {/* Quick Suggestions + Input Bar */}
      <footer className="border-t border-zinc-200 bg-white px-3 pt-2.5 pb-3 space-y-2.5">
        <div className="flex gap-1.5 overflow-x-auto pb-0.5 no-scrollbar">
          {QUICK_COMMANDS.map((cmd) => (
            <button
              key={cmd}
              type="button"
              disabled={isSending}
              onClick={() => void handleSend(cmd)}
              className="shrink-0 rounded-full border border-zinc-200 bg-zinc-50 px-3 py-1 text-xs text-zinc-700 transition-colors hover:border-pink-200 hover:bg-pink-50 hover:text-[#A50064] disabled:opacity-50 cursor-pointer"
            >
              {cmd}
            </button>
          ))}
        </div>

        <form onSubmit={handleFormSubmit} className="flex items-center gap-2">
          <Input
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Nhập lệnh chuyển tiền..."
            disabled={isSending}
            className="flex-1"
          />
          <Button
            type="submit"
            size="icon"
            disabled={isSending || !inputValue.trim()}
            aria-label="Gửi lệnh"
          >
            <SendHorizontal className="size-4" />
          </Button>
        </form>
      </footer>

      {/* Modal PIN / Sinh trắc học */}
      <AuthPinModal
        open={pinModalOpen}
        onOpenChange={setPinModalOpen}
        draft={activeDraft}
        onConfirmSuccess={handleConfirmSuccess}
      />
    </div>
  );
}
