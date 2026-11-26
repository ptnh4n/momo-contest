import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-[11px] font-semibold tracking-wide transition-colors",
  {
    variants: {
      variant: {
        pending: "border-amber-200 bg-amber-50 text-amber-700",
        completed: "border-emerald-200 bg-emerald-50 text-emerald-700",
        cancelled: "border-zinc-200 bg-zinc-100 text-zinc-600",
        blocked: "border-red-200 bg-red-50 text-red-700",
        outline: "border-zinc-200 text-zinc-700",
      },
    },
    defaultVariants: {
      variant: "outline",
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}

export { Badge, badgeVariants };
