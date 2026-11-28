"use client";

import * as React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Fingerprint, Lock, ScanFace } from "lucide-react";
import { toast } from "sonner";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  confirmDraftTransaction,
  type ConfirmApiResponse,
  type DraftObject,
} from "@/services/demoService";
import { formatVnd } from "@/components/DraftTransactionCard";

export interface AuthPinModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  draft: DraftObject | null;
  onConfirmSuccess: (res: ConfirmApiResponse) => void;
}

const BIOMETRIC_DEMO_PIN = "123456";

/**
 * AuthPinModal — Xác thực giao dịch bằng mã PIN 6 số hoặc mô phỏng Quét vân tay / FaceID
 * (chạy animation 1 giây rồi gọi API chốt giao dịch).
 */
export function AuthPinModal({
  open,
  onOpenChange,
  draft,
  onConfirmSuccess,
}: AuthPinModalProps) {
  const [pin, setPin] = React.useState("");
  const [saveAlias, setSaveAlias] = React.useState(false);
  const suggestedAlias = draft?.source_type === "OCR_RECEIPT" ? null : draft?.suggested_alias;
  const [isSubmitting, setIsSubmitting] = React.useState(false);
  const [isScanning, setIsScanning] = React.useState(false);

  const handleDialogOpenChange = (next: boolean) => {
    if (isScanning || isSubmitting) return;
    if (!next) {
      setPin("");
      setSaveAlias(false);
      setIsSubmitting(false);
      setIsScanning(false);
    }
    onOpenChange(next);
  };

  const executeConfirm = async (pinCode: string, persistAlias = false) => {
    if (!draft || isSubmitting) return;
    setIsSubmitting(true);
    try {
      const res = await confirmDraftTransaction(draft.tx_id, pinCode, persistAlias);
      toast.success(
        `Chuyển ${formatVnd(draft.amount)} thành công. Số dư mới: ${formatVnd(res.new_balance)}`
      );
      if (res.alias_saved) toast.success(`Đã lưu biệt danh “${res.alias_saved}”.`);
      onConfirmSuccess(res);
      setPin("");
      setSaveAlias(false);
      onOpenChange(false);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Xác thực giao dịch thất bại";
      toast.error(msg);
    } finally {
      setIsSubmitting(false);
      setIsScanning(false);
    }
  };

  const handlePinSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (pin.length !== 6) {
      toast.error("Vui lòng nhập đủ 6 chữ số mã PIN");
      return;
    }
    await executeConfirm(pin, Boolean(suggestedAlias && saveAlias));
  };

  const handleBiometricScan = async () => {
    if (!draft || isSubmitting || isScanning) return;
    setIsScanning(true);
    // Chạy animation mô phỏng quét vân tay / FaceID đúng 1 giây rồi gọi API chốt giao dịch
    await new Promise((resolve) => setTimeout(resolve, 1000));
    await executeConfirm(BIOMETRIC_DEMO_PIN);
  };

  return (
    <Dialog open={open} onOpenChange={handleDialogOpenChange}>
      <DialogContent>
        <DialogHeader>
          <div className="mx-auto mb-1 flex size-10 items-center justify-center rounded-full bg-pink-50 text-[#A50064]">
            <Lock className="size-5" />
          </div>
          <DialogTitle>Xác thực giao dịch</DialogTitle>
          <DialogDescription>
            {draft ? (
              <span>
                {draft.recipient_name} · <span className="font-mono">{draft.recipient_account}</span> ·{" "}
                <strong className="text-zinc-900">{formatVnd(draft.amount)}</strong>
              </span>
            ) : (
              "Nhập mã PIN 6 số hoặc xác thực sinh trắc học"
            )}
          </DialogDescription>
        </DialogHeader>

        <AnimatePresence mode="wait">
          {isScanning ? (
            <motion.div
              key="scanning"
              initial={{ opacity: 0, scale: 0.94 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.94 }}
              className="flex flex-col items-center justify-center gap-3 py-6"
            >
              <div className="relative flex size-20 items-center justify-center rounded-full bg-pink-50">
                <motion.div
                  className="absolute inset-0 rounded-full border-2 border-[#A50064]/50"
                  animate={{ scale: [1, 1.28, 1], opacity: [0.85, 0.15, 0.85] }}
                  transition={{ duration: 1, repeat: Infinity, ease: "easeInOut" }}
                />
                <motion.div
                  className="absolute left-2 right-2 h-0.5 rounded-full bg-[#A50064]"
                  animate={{ y: [-24, 24, -24] }}
                  transition={{ duration: 0.9, repeat: Infinity, ease: "easeInOut" }}
                />
                <Fingerprint className="size-10 text-[#A50064]" />
              </div>
              <p className="text-xs font-medium text-zinc-600">
                Đang xác thực sinh trắc học...
              </p>
            </motion.div>
          ) : (
            <motion.form
              key="pin-form"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onSubmit={handlePinSubmit}
              className="space-y-3"
            >
              <div className="space-y-1.5">
                <Input
                  type="password"
                  inputMode="numeric"
                  pattern="[0-9]*"
                  maxLength={6}
                  autoFocus
                  placeholder="Nhập mã PIN 6 số (demo: 123456)"
                  value={pin}
                  onChange={(e) => setPin(e.target.value.replace(/\D/g, "").slice(0, 6))}
                  disabled={isSubmitting}
                  className="text-center font-mono text-base tracking-[0.45em]"
                />
              </div>

              {suggestedAlias && (
                <label className="flex items-center gap-2 text-sm text-zinc-700">
                  <input
                    type="checkbox"
                    checked={saveAlias}
                    onChange={(event) => setSaveAlias(event.target.checked)}
                    disabled={isSubmitting}
                  />
                  <span>Lưu người nhận với biệt danh “{suggestedAlias}” khi xác nhận PIN</span>
                </label>
              )}
              <Button
                type="submit"
                className="w-full"
                disabled={isSubmitting || pin.length !== 6}
              >
                {isSubmitting ? "Đang xử lý..." : "Xác nhận PIN"}
              </Button>

              <div className="relative flex items-center py-1">
                <div className="flex-1 border-t border-zinc-200" />
                <span className="px-2.5 text-[11px] uppercase text-zinc-400">hoặc</span>
                <div className="flex-1 border-t border-zinc-200" />
              </div>

              <Button
                type="button"
                variant="outline"
                className="w-full border-pink-200 text-[#A50064] hover:bg-pink-50 hover:text-[#8A0054]"
                disabled={isSubmitting}
                onClick={handleBiometricScan}
              >
                <Fingerprint className="size-4" />
                <ScanFace className="size-4" />
                Quét vân tay / FaceID
              </Button>
            </motion.form>
          )}
        </AnimatePresence>
      </DialogContent>
    </Dialog>
  );
}
