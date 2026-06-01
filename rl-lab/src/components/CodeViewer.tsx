// 可折叠的源码查看器：展示算法的真实 TS 实现，配语法高亮。
// 让“动画 + 直觉讲解 + 真实源码”三位一体。
import { useState } from "react";
import { Highlight, themes } from "prism-react-renderer";
import { Code2, ChevronDown, ChevronRight, Check, Copy } from "lucide-react";

interface Props {
  code: string;
  language?: string;
  /** 源码文件路径，显示在标题栏 */
  path?: string;
}

export default function CodeViewer({ code, language = "tsx", path }: Props) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  const lineCount = code.trimEnd().split("\n").length;

  return (
    <div className="glass rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center gap-2 px-4 py-3 text-left hover:bg-white/[0.02] transition-colors"
      >
        <Code2 className="w-4 h-4 text-[#00ff88]" />
        <span className="text-sm font-semibold text-slate-200">查看源码</span>
        {path && <span className="text-xs text-slate-600 font-mono hidden sm:inline">{path}</span>}
        <span className="text-xs text-slate-600 ml-auto mr-1">{lineCount} 行</span>
        {open ? (
          <ChevronDown className="w-4 h-4 text-slate-500" />
        ) : (
          <ChevronRight className="w-4 h-4 text-slate-500" />
        )}
      </button>

      {open && (
        <div className="relative border-t border-slate-800/60">
          <button
            onClick={copy}
            className="absolute top-2.5 right-2.5 z-10 inline-flex items-center gap-1 px-2 py-1 rounded text-xs border border-slate-700 bg-[#0a0e17]/80 text-slate-400 hover:text-[#00ff88] hover:border-[#00ff88]/40 transition-all"
          >
            {copied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
            {copied ? "已复制" : "复制"}
          </button>
          <Highlight code={code.trimEnd()} language={language} theme={themes.nightOwl}>
            {({ className, style, tokens, getLineProps, getTokenProps }) => (
              <pre
                className={`${className} text-xs leading-relaxed overflow-auto max-h-[28rem] p-4 m-0`}
                style={{ ...style, background: "transparent" }}
              >
                {tokens.map((line, i) => {
                  const lineProps = getLineProps({ line });
                  return (
                    <div key={i} {...lineProps} className={`${lineProps.className} table-row`}>
                      <span className="table-cell pr-4 text-right select-none text-slate-700 font-mono w-10">
                        {i + 1}
                      </span>
                      <span className="table-cell">
                        {line.map((token, key) => (
                          <span key={key} {...getTokenProps({ token })} />
                        ))}
                      </span>
                    </div>
                  );
                })}
              </pre>
            )}
          </Highlight>
        </div>
      )}
    </div>
  );
}
