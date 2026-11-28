"use client";

import * as React from "react";
import { motion } from "framer-motion";
import { Paperclip, RotateCcw, SendHorizontal, ShieldCheck, X } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { DraftTransactionCard, formatVnd } from "@/components/DraftTransactionCard";
import { AuthPinModal } from "@/components/AuthPinModal";
import {
  cancelDraftTransaction,
  getWalletBalance,
  resetDemoAccount,
  sendChatMessage,
  type ChatApiResponse,
  type ClarifyOption,
  type ConfirmApiResponse,
  type DraftObject,
} from "@/services/demoService";

type ChatItem =
  | { id: string; kind: "user"; text: string }
  | { id: string; kind: "draft"; draft: DraftObject }
  | { id: string; kind: "notice"; text: string; options?: ClarifyOption[] };

const MAX_RECEIPT_BYTES = 5 * 1024 * 1024;

function readReceipt(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1]);
    reader.onerror = () => reject(new Error("Không đọc được ảnh hóa đơn."));
    reader.readAsDataURL(file);
  });
}

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
  const [receipt, setReceipt] = React.useState<File | null>(null);
  const [balance, setBalance] = React.useState<number | null>(null);
  const [clarifyId, setClarifyId] = React.useState<string | null>(null);
  const [isLoadingBalance, setIsLoadingBalance] = React.useState(true);
  const [isSending, setIsSending] = React.useState(false);
  const [isResetting, setIsResetting] = React.useState(false);
  const [cancellingTxId, setCancellingTxId] = React.useState<string | null>(null);
  const [activeDraft, setActiveDraft] = React.useState<DraftObject | null>(null);
  const [pinModalOpen, setPinModalOpen] = React.useState(false);

  const listEndRef = React.useRef<HTMLDivElement>(null);
  const uploadRef = React.useRef<HTMLInputElement>(null);
  const operation = React.useRef(false);
  const pending = React.useRef<{
    message: string; file?: File; option?: ClarifyOption; clarifyId: string | null; intentId: string;
  } | null>(null);
  const isBusy = isSending || isResetting || cancellingTxId !== null || pinModalOpen || isLoadingBalance;

  React.useEffect(() => {
    let mounted = true;
    getWalletBalance()
      .then((data) => { if (mounted) setBalance(data.balance); })
      .catch(() => { if (mounted) toast.error("Không tải được số dư ví demo."); })
      .finally(() => { if (mounted) setIsLoadingBalance(false); });
    return () => { mounted = false; };
  }, []);

  React.useEffect(() => {
    listEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [items, isSending]);

  const handleSend = async (rawText: string, file?: File, option?: ClarifyOption) => {
    const message = rawText.trim();
    if (!message || isBusy || operation.current) return;
    if (option && !clarifyId) return;

    operation.current = true;
    setIsSending(true);
    const attempt = pending.current;
    const intentId = attempt && attempt.message === message && attempt.file === file
      && attempt.option === option && attempt.clarifyId === clarifyId ? attempt.intentId : crypto.randomUUID();
    pending.current = { message, file, option, clarifyId, intentId };

    try {
      const imageBase64 = file ? await readReceipt(file) : undefined;
      const res: ChatApiResponse = await sendChatMessage(message, intentId, {
        imageBase64,
        clarifyChoice: option && clarifyId ? {
          ref_intent_id: clarifyId,
          contact_id: option.contact_id ?? undefined,
          amount: option.amount ?? undefined,
        } : undefined,
      });
      pending.current = null;
      setItems((prev) => {
        const next: ChatItem[] = [...prev, {
          id: `user-${intentId}`, kind: "user", text: file ? `${message} · ${file.name}` : message,
        }];
        if (res.draft) next.push({ id: intentId, kind: "draft", draft: res.draft });
        else if (res.decision === "CLARIFY") next.push({
          id: intentId, kind: "notice",
          text: res.clarify_options?.kind === "CONTACT" ? "Chọn người nhận."
            : res.clarify_options?.kind === "CURRENCY_UNIT" ? "Xác nhận số tiền."
            : "Cần làm rõ thông tin giao dịch. Hãy nhập thêm chi tiết.",
          options: res.clarify_options?.options,
        });
        else if (res.decision === "BLOCK_ANOMALY") next.push({
          id: intentId, kind: "notice", text: "Giao dịch bị chặn.",
        });
        return next;
      });
      setClarifyId(res.clarify_options?.options.length ? intentId : null);
      if (res.new_balance != null) setBalance(res.new_balance);
      if (!option) setInputValue("");
      if (file) setReceipt(null);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Không thể gửi yêu cầu");
    } finally {
      operation.current = false;
      setIsSending(false);
    }
  };

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    void handleSend(inputValue, receipt ?? undefined);
  };

  const handleReceiptSelection = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    if (!["image/png", "image/jpeg", "image/webp"].includes(file.type) || file.size > MAX_RECEIPT_BYTES) {
      toast.error("Dùng ảnh PNG, JPEG hoặc WebP tối đa 5 MB.");
      return;
    }
    setReceipt(file);
  };

  const handleOpenConfirm = (draft: DraftObject) => {
    if (isBusy || operation.current) return;
    setActiveDraft(draft);
    setPinModalOpen(true);
  };

  const handleConfirmSuccess = (res: ConfirmApiResponse) => {
    setBalance(res.new_balance);
    setItems((prev) =>
      prev.map((item) =>
        item.kind === "draft" && item.draft.tx_id === res.tx_id
          ? { ...item, draft: { ...item.draft, status: "COMPLETED" } }
          : item
      )
    );
  };

  const handleCancelDraft = async (draft: DraftObject) => {
    if (isBusy || operation.current) return;
    operation.current = true;
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
      operation.current = false;
      setCancellingTxId(null);
    }
  };

  const handleResetDemo = async () => {
    if (isBusy || operation.current) return;
    operation.current = true;
    setIsResetting(true);
    try {
      const res = await resetDemoAccount();
      pending.current = null;
      setItems([]);
      setClarifyId(null);
      setInputValue("");
      setReceipt(null);
      setBalance(res.balance);
      setActiveDraft(null);
      setPinModalOpen(false);
      toast.success(`Đã reset tài khoản Demo (${formatVnd(res.balance)})`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Reset tài khoản Demo thất bại";
      toast.error(msg);
    } finally {
      operation.current = false;
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
          <div>
            <h1 className="text-sm font-bold tracking-tight text-zinc-900">SafePay Agent</h1>
            <p className="text-xs text-zinc-600">
              Số dư demo <strong>{balance == null ? "—" : formatVnd(balance)}</strong>
            </p>
          </div>
        </div>

        <Button
          type="button"
          variant="outline"
          size="sm"
          className="h-8 gap-1.5 rounded-full px-3 text-xs"
          disabled={isBusy}
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
          ) : item.kind === "notice" ? (
            <div key={item.id} className="rounded-xl border border-zinc-200 bg-white p-3 text-sm">
              <p>{item.text}</p>
              {item.id === clarifyId && item.options?.map((option) => (
                <Button
                  key={option.contact_id ?? option.amount}
                  type="button"
                  variant="outline"
                  size="sm"
                  className="mr-2 mt-2"
                  disabled={isBusy}
                  onClick={() => void handleSend(option.label, undefined, option)}
                >
                  {option.label}
                </Button>
              ))}
            </div>
          ) : (
            <div key={item.id} className="flex justify-start">
              <DraftTransactionCard
                draft={item.draft}
                onConfirmClick={handleOpenConfirm}
                onCancelClick={handleCancelDraft}
                isCancelling={cancellingTxId === item.draft.tx_id}
                actionsDisabled={isBusy}
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
              disabled={isBusy}
              onClick={() => void handleSend(cmd)}
              className="shrink-0 rounded-full border border-zinc-200 bg-zinc-50 px-3 py-1 text-xs text-zinc-700 transition-colors hover:border-pink-200 hover:bg-pink-50 hover:text-[#A50064] disabled:opacity-50 cursor-pointer"
            >
              {cmd}
            </button>
          ))}
        </div>

        {receipt && (
          <div className="flex items-center gap-2 text-xs text-zinc-700">
            <span className="min-w-0 truncate">{receipt.name}</span>
            <Button type="button" variant="ghost" size="icon" aria-label="Bỏ ảnh"
              disabled={isBusy} onClick={() => setReceipt(null)}>
              <X className="size-4" />
            </Button>
          </div>
        )}
        <form onSubmit={handleFormSubmit} className="flex items-center gap-2">
          <input ref={uploadRef} type="file" accept="image/png,image/jpeg,image/webp"
            onChange={handleReceiptSelection} hidden />
          <Button type="button" variant="outline" size="icon" aria-label="Đính kèm hóa đơn"
            disabled={isBusy} onClick={() => uploadRef.current?.click()}>
            <Paperclip className="size-4" />
          </Button>
          <Input
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder={receipt ? "Chia hóa đơn cho 4 người rồi chuyển cho thằng Nam" : "Nhập lệnh chuyển tiền..."}
            disabled={isBusy}
            className="flex-1"
          />
          <Button
            type="submit"
            size="icon"
            disabled={isBusy || !inputValue.trim()}
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
