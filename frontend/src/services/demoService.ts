/**
 * demoService.ts — Lớp gọi API theo hợp đồng OpenAPI (backend/openapi/safepay.openapi.json).
 *
 * Quy tắc bảo mật:
 * - Frontend chỉ sử dụng object `draft` từ DB để render thẻ giao dịch.
 * - Tuyệt đối không dùng `reply_text` của LLM để hiển thị thông tin chuyển tiền.
 * - Tự động fallback sang dữ liệu mẫu (khớp backend/app/db/seed.py) khi backend chưa khởi chạy
 *   (chỉ khi lỗi kết nối mạng; các lỗi HTTP 4xx/5xx từ server thật luôn được ném thẳng).
 */

export type DraftDbStatus = "DRAFT" | "PENDING" | "COMPLETED" | "CANCELLED" | "BLOCKED";
export type SourceType = "CHAT_CONTACT" | "CHAT_RAW_ACCT" | "OCR_RECEIPT";
export type DecisionType = "PASS_AUTO" | "REQUIRE_HITL" | "CLARIFY" | "BLOCK_ANOMALY";

export interface DraftObject {
  tx_id: string;
  recipient_name: string;
  recipient_account: string;
  bank_code: string;
  amount: number;
  note?: string | null;
  source_type: SourceType;
  suggested_alias?: string | null;
  status: DraftDbStatus;
}

export interface ChatApiResponse {
  decision: DecisionType;
  risk_score: number;
  xai_reasons: string[];
  draft: DraftObject | null;
  new_balance?: number | null;
}

export interface ConfirmApiResponse {
  status: "COMPLETED";
  new_balance: number;
  tx_id: string;
  alias_saved?: string | null;
}

export interface CancelApiResponse {
  status: "CANCELLED";
  tx_id: string;
}

export interface ResetApiResponse {
  status: "RESET_OK";
  balance: number;
}

export interface ApiErrorPayload {
  error_code?: string;
  message?: string;
}

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/+$/, "") || "http://localhost:8000";

export const DEMO_USER_ID = "u_demo";
const DEMO_PIN = "123456";

function generateUuidV4(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

// ---------------------------------------------------------------------------
// Fallback demo state (khớp backend/app/db/seed.py) khi backend chưa bật
// ---------------------------------------------------------------------------
const mockState = {
  balance: 10_000_000,
  drafts: new Map<string, DraftObject>(),
};

const SEED_CONTACTS = [
  {
    keywords: ["thằng nam", "thang nam", "nam"],
    recipient_name: "Nguyễn Văn Nam",
    recipient_account: "9001234567",
    bank_code: "MB",
  },
  {
    keywords: ["sếp hùng", "sep hung", "sếp", "sep", "hùng", "hung"],
    recipient_name: "Trần Quốc Hùng",
    recipient_account: "9007654321",
    bank_code: "VCB",
  },
  {
    keywords: ["minh"],
    recipient_name: "Lê Quang Minh",
    recipient_account: "9001112222",
    bank_code: "TCB",
  },
];

function parseAmountVnd(message: string): number {
  const lower = message.toLowerCase();
  const trMatch = lower.match(/(\d+(?:[.,]\d+)?)\s*(?:tr|triệu|trieu)/);
  if (trMatch) {
    return Math.round(parseFloat(trMatch[1].replace(",", ".")) * 1_000_000);
  }
  const kMatch = lower.match(/(\d+(?:[.,]\d+)?)\s*(?:k|nghìn|nghin|ngàn|ngan)/);
  if (kMatch) {
    return Math.round(parseFloat(kMatch[1].replace(",", ".")) * 1_000);
  }
  const rawMatch = lower.match(/\b(\d{4,9})\b/);
  if (rawMatch) {
    return parseInt(rawMatch[1], 10);
  }
  return 500_000;
}

function buildMockDraft(message: string): ChatApiResponse {
  const lower = message.toLowerCase();
  const amount = parseAmountVnd(message);

  // Kiểm tra STK thô (8-16 chữ số)
  const rawAcctMatch = message.match(/\b(\d{8,16})\b/);
  let recipient_name = "Nguyễn Văn Nam";
  let recipient_account = "9001234567";
  let bank_code = "MB";
  let source_type: SourceType = "CHAT_CONTACT";

  if (rawAcctMatch) {
    recipient_account = rawAcctMatch[1];
    recipient_name = "Người nhận theo STK";
    bank_code = "VCB";
    source_type = "CHAT_RAW_ACCT";
  } else {
    const matched = SEED_CONTACTS.find((c) => c.keywords.some((kw) => lower.includes(kw)));
    if (matched) {
      recipient_name = matched.recipient_name;
      recipient_account = matched.recipient_account;
      bank_code = matched.bank_code;
    }
  }

  const tx_id = `tx_${generateUuidV4().slice(0, 8)}`;
  const draft: DraftObject = {
    tx_id,
    recipient_name,
    recipient_account,
    bank_code,
    amount,
    note: message.trim(),
    source_type,
    suggested_alias: null,
    status: "DRAFT",
  };
  mockState.drafts.set(tx_id, draft);

  return {
    decision: "REQUIRE_HITL",
    risk_score: 0.15,
    xai_reasons: [],
    draft,
    new_balance: null,
  };
}

function isNetworkError(err: unknown): boolean {
  return err instanceof TypeError;
}

async function parseError(res: Response): Promise<Error> {
  try {
    const data = (await res.json()) as ApiErrorPayload;
    return new Error(data.message || data.error_code || `HTTP ${res.status}`);
  } catch {
    return new Error(`Yêu cầu thất bại (HTTP ${res.status})`);
  }
}

// ---------------------------------------------------------------------------
// Public API functions
// ---------------------------------------------------------------------------

/**
 * Gửi tin nhắn tới POST /api/chat để nhận object Draft từ DB.
 */
export async function sendChatMessage(
  message: string,
  intentId: string = generateUuidV4()
): Promise<ChatApiResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: DEMO_USER_ID,
        intent_id: intentId,
        message,
      }),
    });

    if (!res.ok) {
      throw await parseError(res);
    }

    const data = await res.json();
    return {
      decision: data.decision,
      risk_score: data.risk_score ?? 0,
      xai_reasons: data.xai_reasons ?? [],
      draft: data.draft ?? null,
      new_balance: data.new_balance ?? null,
    };
  } catch (err) {
    if (isNetworkError(err)) {
      await new Promise((r) => setTimeout(r, 350));
      return buildMockDraft(message);
    }
    throw err;
  }
}

/**
 * Xác nhận giao dịch bằng PIN qua POST /api/drafts/{tx_id}/confirm.
 */
export async function confirmDraftTransaction(
  txId: string,
  pinCode: string,
  saveAlias: boolean = false
): Promise<ConfirmApiResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/drafts/${encodeURIComponent(txId)}/confirm`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: DEMO_USER_ID,
        pin_code: pinCode,
        save_alias: saveAlias,
      }),
    });

    if (!res.ok) {
      throw await parseError(res);
    }

    return (await res.json()) as ConfirmApiResponse;
  } catch (err) {
    if (isNetworkError(err)) {
      await new Promise((r) => setTimeout(r, 250));
      if (pinCode !== DEMO_PIN) {
        throw new Error("Mã PIN không chính xác (PIN demo: 123456)");
      }
      const draft = mockState.drafts.get(txId);
      if (!draft) {
        throw new Error("Không tìm thấy bản nháp giao dịch");
      }
      if (draft.status === "COMPLETED") {
        return { status: "COMPLETED", new_balance: mockState.balance, tx_id: txId, alias_saved: null };
      }
      if (draft.amount > mockState.balance) {
        throw new Error("Số dư ví không đủ để thực hiện giao dịch");
      }
      mockState.balance -= draft.amount;
      draft.status = "COMPLETED";
      return {
        status: "COMPLETED",
        new_balance: mockState.balance,
        tx_id: txId,
        alias_saved: null,
      };
    }
    throw err;
  }
}

/**
 * Hủy bản nháp giao dịch qua POST /api/drafts/{tx_id}/cancel.
 */
export async function cancelDraftTransaction(txId: string): Promise<CancelApiResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/drafts/${encodeURIComponent(txId)}/cancel`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: DEMO_USER_ID }),
    });

    if (!res.ok) {
      throw await parseError(res);
    }

    return (await res.json()) as CancelApiResponse;
  } catch (err) {
    if (isNetworkError(err)) {
      await new Promise((r) => setTimeout(r, 200));
      const draft = mockState.drafts.get(txId);
      if (draft) {
        draft.status = "CANCELLED";
      }
      return { status: "CANCELLED", tx_id: txId };
    }
    throw err;
  }
}

/**
 * Khôi phục số dư ví demo (10.000.000 VNĐ) và danh bạ mẫu qua POST /api/demo/reset.
 */
export async function resetDemoAccount(): Promise<ResetApiResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/demo/reset`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: DEMO_USER_ID }),
    });

    if (!res.ok) {
      throw await parseError(res);
    }

    return (await res.json()) as ResetApiResponse;
  } catch (err) {
    if (isNetworkError(err)) {
      await new Promise((r) => setTimeout(r, 250));
      mockState.balance = 10_000_000;
      mockState.drafts.clear();
      return { status: "RESET_OK", balance: 10_000_000 };
    }
    throw err;
  }
}
