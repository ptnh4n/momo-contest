/** Browser client for the SafePay sandbox API. Render transaction details from draft objects. */
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

export interface ClarifyOption {
  label: string;
  contact_id?: string | null;
  amount?: number | null;
}

export interface ChatApiResponse {
  decision: DecisionType;
  risk_score: number;
  xai_reasons: string[];
  draft: DraftObject | null;
  new_balance?: number | null;
  clarify_options?: {
    kind: "CONTACT" | "CURRENCY_UNIT";
    options: ClarifyOption[];
  } | null;
}

export interface ConfirmApiResponse {
  status: "COMPLETED";
  new_balance: number;
  tx_id: string;
  alias_saved?: string | null;
}

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/+$/, "") || "http://localhost:8000";

export const DEMO_USER_ID = "u_demo";

async function request<T>(path: string, payload?: object): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, {
      cache: "no-store",
      ...(payload && {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }),
    });
  } catch (cause) {
    throw new Error("Không kết nối được ví demo. Hãy kiểm tra backend và thử lại.", { cause });
  }

  if (!res.ok) {
    const data = await res.json().catch(() => null);
    throw new Error(data?.message || data?.error_code || `Yêu cầu thất bại (HTTP ${res.status})`);
  }
  return res.json() as Promise<T>;
}

export function sendChatMessage(
  message: string,
  intentId: string,
  options?: {
    imageBase64?: string;
    clarifyChoice?: { ref_intent_id: string; contact_id?: string; amount?: number };
  }
) {
  return request<ChatApiResponse>("/api/chat", {
    user_id: DEMO_USER_ID,
    intent_id: intentId,
    message,
    image_base64: options?.imageBase64,
    clarify_choice: options?.clarifyChoice,
  });
}

export function confirmDraftTransaction(txId: string, pinCode: string, saveAlias = false) {
  return request<ConfirmApiResponse>(`/api/drafts/${encodeURIComponent(txId)}/confirm`, {
    user_id: DEMO_USER_ID,
    pin_code: pinCode,
    save_alias: saveAlias,
  });
}

export function cancelDraftTransaction(txId: string) {
  return request<{ status: "CANCELLED"; tx_id: string }>(`/api/drafts/${encodeURIComponent(txId)}/cancel`, {
    user_id: DEMO_USER_ID,
  });
}

export function resetDemoAccount() {
  return request<{ status: "RESET_OK"; balance: number }>("/api/demo/reset", {
    user_id: DEMO_USER_ID,
  });
}

export function getWalletBalance() {
  return request<{ balance: number }>(`/api/wallet/state?user_id=${encodeURIComponent(DEMO_USER_ID)}`);
}
