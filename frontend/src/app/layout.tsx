import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { Toaster } from "sonner";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "SafePay Agent",
  description: "Trợ lý Giao dịch AI SafePay Agent — Raw Draft DB & HITL",
};

const STRIP_EXTENSION_ATTRS_SCRIPT = `(function(){try{var c=function(r){if(!r||!r.querySelectorAll)return;var l=r.querySelectorAll("[bis_skin_checked]");for(var i=0;i<l.length;i++)l[i].removeAttribute("bis_skin_checked")};c(document.documentElement);var o=new MutationObserver(function(m){for(var i=0;i<m.length;i++){var x=m[i];if(x.type==="attributes"&&x.attributeName==="bis_skin_checked"&&x.target){x.target.removeAttribute("bis_skin_checked")}else if(x.type==="childList"){for(var j=0;j<x.addedNodes.length;j++){var n=x.addedNodes[j];if(n&&n.nodeType===1){if(n.hasAttribute&&n.hasAttribute("bis_skin_checked"))n.removeAttribute("bis_skin_checked");c(n)}}}}});o.observe(document.documentElement,{attributes:true,attributeFilter:["bis_skin_checked"],childList:true,subtree:true});window.addEventListener("load",function(){setTimeout(function(){o.disconnect()},2000)},{once:true});setTimeout(function(){o.disconnect()},6000)}catch(e){}})();`;

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="vi"
      suppressHydrationWarning
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <head suppressHydrationWarning>
        <script
          id="strip-ext-attrs"
          suppressHydrationWarning
          dangerouslySetInnerHTML={{ __html: STRIP_EXTENSION_ATTRS_SCRIPT }}
        />
      </head>
      <body
        suppressHydrationWarning
        className="min-h-full bg-zinc-100 text-zinc-950 font-sans"
      >
        {children}
        <Toaster position="top-center" richColors />
      </body>
    </html>
  );
}
