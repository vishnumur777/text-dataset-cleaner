import { useEffect, useMemo, useRef } from "react";
import { diffWords } from "../diff";

function Span({ op, side }) {
  const text = op.value;
  if (op.type === "same") return <span className="tok same">{text}</span>;
  if (side === "left" && op.type === "del") {
    return (
      <span className="tok del" title="replaced in the cleaned text">
        {text}
      </span>
    );
  }
  if (side === "right" && op.type === "ins") {
    return (
      <span className="tok ins" title="added by the cleaner">
        {text}
      </span>
    );
  }
  return null;
}

export function DiffPanel({ original, edited, side, emptyText = "Nothing to show yet." }) {
  const ops = useMemo(() => diffWords(original, edited), [original, edited]);
  const visible = ops.filter((op) => (side === "left" ? op.type !== "ins" : op.type !== "del"));
  if (!visible.length) return <p className="muted">{emptyText}</p>;
  return (
    <div className={`diff diff-${side}`}>
      {visible.map((op, index) => (
        <Span key={index} op={op} side={side} />
      ))}
    </div>
  );
}

/** Editable panel: coloured diff view that becomes a plain editor on focus. */
export function EditableDiff({ original, value, onChange, placeholder }) {
  const ref = useRef(null);
  const lastValue = useRef(null);
  const focused = useRef(false);

  const ops = useMemo(() => diffWords(original, value), [original, value]);

  useEffect(() => {
    const node = ref.current;
    if (!node || focused.current) return;
    if (lastValue.current === value) return;
    lastValue.current = value;
    node.innerHTML = ops
      .filter((op) => op.type !== "del")
      .map((op) => {
        const safe = op.value
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")
          .replace(/\n/g, "<br/>");
        return op.type === "ins" ? `<span class="tok ins">${safe}</span>` : `<span class="tok same">${safe}</span>`;
      })
      .join("");
  }, [ops, value]);

  return (
    <div
      ref={ref}
      className="editable"
      contentEditable
      suppressContentEditableWarning
      spellCheck={false}
      role="textbox"
      aria-multiline="true"
      aria-label="Cleaned text, editable"
      data-placeholder={placeholder}
      onFocus={() => {
        focused.current = true;
      }}
      onBlur={() => {
        focused.current = false;
        lastValue.current = null;
        onChange(ref.current?.innerText ?? "");
      }}
      onInput={() => {
        onChange(ref.current?.innerText ?? "");
      }}
    />
  );
}
