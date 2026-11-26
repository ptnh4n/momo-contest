"use client";

import * as React from "react";
import { motion } from "framer-motion";
import {
  CheckCircle2,
  Clock,
  ShieldAlert,
  ShieldCheck,
  XCircle,
} from "lucide-react";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { DraftDbStatus, DraftObject } from "@/services/demoService";

export interface DraftTransactionCardProps {
  draft: DraftObject;
  onConfirmClick: (draft: DraftObject) => void;
  onCancelClick?: (draft: DraftObject) => void;
  isCancelling?: boolean;
}

export function formatVnd(amount: number): string {
  return `${new Intl.NumberFormat("vi-VN").format(amount)} VNĐ`;
}

function getStatusMeta(status: DraftDbStatus): {
  label: "PENDING" | "COMPLETED" | "CANCELLED" | "BLOCKED";
  variant: "pending" | "completed" | "cancelled" | "blocked";
  Icon: React.ComponentType<{ className?: string }>;
} {
  switch (status) {
    case "COMPLETED":
      return { label: "COMPLETED", variant: "completed", Icon: CheckCircle2 };
    case "CANCELLED":
      return { label: "CANCELLED", variant: "cancelled", Icon: XCircle };
    case "BLOCKED":
      return { label: "BLOCKED", variant: "blocked", Icon: ShieldAlert };
    case "DRAFT":
    case "PENDING":
    default:
      return { label: "PENDING", variant: "pending", Icon: Clock };
  }
}

/**
 * DraftTransactionCard — Render trực tiếp dữ liệu thô từ object Draft DB
 * (recipient_account, recipient_name, bank_code, amount, status).
 * Tuyệt đối không render text giải thích của LLM.
 */
export function DraftTransactionCard({
  draft,
  onConfirmClick,
  onCancelClick,
  isCancelling = false,
}: DraftTransactionCardProps) {
  const { label, variant, Icon } = getStatusMeta(draft.status);
  const isPending = draft.status === "DRAFT" || draft.status === "PENDING";

  return (
    <motion.div
      initial={{ opacity: 0, y: 10, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.22, ease: "easeOut" }}
      className="w-full"
    >
      <Card className="overflow-hidden border-zinc-200/90 bg-white shadow-sm">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 border-b border-zinc-100 bg-zinc-50/70 px-4 py-3">
          <div className="flex items-center gap-1.5">
            <ShieldCheck className="size-4 text-[#A50064]" />
            <CardTitle className="text-xs font-semibold uppercase tracking-wide text-zinc-700">
              Lệnh chuyển tiền (DB)
            </CardTitle>
          </div>
          <Badge variant={variant}>
            <Icon className="size-3" />
            {label}
          </Badge>
        </CardHeader>

        <CardContent className="space-y-3 px-4 py-3.5">
          <div className="flex items-baseline justify-between border-b border-dashed border-zinc-100 pb-2.5">
            <span className="text-xs text-zinc-500">Số tiền</span>
            <span className="text-lg font-bold tracking-tight text-[#A50064]">
              {formatVnd(draft.amount)}
            </span>
          </div>

          <div className="space-y-2 text-sm">
            <div className="flex items-center justify-between gap-3">
              <span className="text-xs text-zinc-500 shrink-0">Tên chủ TK</span>
              <span className="font-semibold text-zinc-900 text-right truncate">
                {draft.recipient_name}
              </span>
            </div>

            <div className="flex items-center justify-between gap-3">
              <span className="text-xs text-zinc-500 shrink-0">Số tài khoản</span>
              <span className="font-mono text-xs font-semibold text-zinc-900 text-right">
                {draft.recipient_account}
                {draft.bank_code ? ` · ${draft.bank_code}` : ""}
              </span>
            </div>
          </div>
        </CardContent>

        {isPending && (
          <CardFooter className="gap-2 border-t border-zinc-100 bg-zinc-50/40 px-4 py-3">
            {onCancelClick && (
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-9 px-3 text-xs"
                disabled={isCancelling}
                onClick={() => onCancelClick(draft)}
              >
                Hủy
              </Button>
            )}
            <Button
              type="button"
              size="sm"
              className="h-9 flex-1 text-xs font-semibold"
              disabled={isCancelling}
              onClick={() => onConfirmClick(draft)}
            >
              Xác nhận chuyển
            </Button>
          </CardFooter>
        )}
      </Card>
    </motion.div>
  );
}
