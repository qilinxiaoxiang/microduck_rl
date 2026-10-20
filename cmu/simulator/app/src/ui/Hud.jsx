// In-game HUD: Back (top-left), pre-order button (top-right) and the
// telemetry stack (bottom-right).
//
// Chrome reads as small comic PANELS: each group sits in a thick cream
// keyline frame on a dark glass plate (1px ink inset between glass and
// frame), with a caption box (cartouche) overlapping the frame's corner -
// flat cream (orange for the shop CTA) fill, ink keyline, hard 2px offset
// shadow, half in / half out like a caption on a comic panel's edge. Same
// tokens as the landing (Anton, orange, cream), zero radius everywhere.
// Title-menu CTAs stay ComicButton.
//
// Whole HUD is hidden while the title/pause overlay is up; touch mode
// strips it down to the thumbs + Back.
import { useEffect, useRef, useState } from "react";
import Box from "@mui/material/Box";
import { useGame } from "../store.js";
import { ORANGE, MONO } from "../theme.js";
import { ANTON, COMIC_INK, CREAM } from "./comic.jsx";

// Matrix-style letter scramble: on change every glyph flips through random
// charset entries, then locks to its target left-to-right over ~0.45 s.
// Monospace keeps the width stable mid-scramble.
const SCRAMBLE_GLYPHS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#$%&*<>/=+";
function useScramble(target) {
  const [text, setText] = useState(target.toUpperCase());
  const timer = useRef(null);
  const shown = useRef(target.toUpperCase());
  useEffect(() => {
    const t = target.toUpperCase();
    if (shown.current === t) return;
    shown.current = t;
    if (timer.current) clearInterval(timer.current);
    const n = t.length;
    const DUR = 450; // ms until the last letter locks
    const rnd = () => SCRAMBLE_GLYPHS[(Math.random() * SCRAMBLE_GLYPHS.length) | 0];
    const t0 = performance.now();
    timer.current = setInterval(() => {
      const k = Math.floor(((performance.now() - t0) / DUR) * n);
      if (k >= n) {
        clearInterval(timer.current);
        timer.current = null;
        setText(t);
        return;
      }
      let out = t.slice(0, k);
      for (let i = k; i < n; i++) out += rnd();
      setText(out);
    }, 40);
    return () => timer.current && clearInterval(timer.current);
  }, [target]);
  return text;
}

const HUD_H = 48;
const FRAME_W = 2; // cream keyline thickness
const PANEL_PAD = 5; // air between the frame and the printed content
const FRAME = "rgba(250, 248, 242, 0.9)";
const GLASS = "rgba(8, 8, 12, 0.6)";

// The panel frame: a thick cream keyline (comic ink weight, HUD read)
// around a dark glass plate, with a 1px ink inset between frame and glass
// so it reads like a printed panel, and a few px of glass padding so the
// content has air inside the frame (the glass ground fills the padding).
// The inset is a pseudo-element so it draws over the cells (an inset
// box-shadow would be painted under them). Overflow clipping lives on the
// inner content wrapper, not the plate, so the caption box can hang over
// the frame's corner.
const plateSx = {
  position: "relative",
  display: "inline-flex",
  height: HUD_H,
  boxSizing: "border-box",
  border: `${FRAME_W}px solid ${FRAME}`,
  padding: `${PANEL_PAD}px`,
  borderRadius: 0,
  background: GLASS,
  "&::after": {
    content: '""',
    position: "absolute",
    inset: 0,
    pointerEvents: "none",
    zIndex: 2,
    boxShadow: `inset 0 0 0 1px ${COMIC_INK}`,
  },
};

// Caption box (cartouche) printed over the panel's top corner, half in /
// half out - the landing's print language: flat fill, ink keyline, hard
// offset shadow, zero radius. Decorative only: no pointer events, no
// layout footprint (absolute over the frame).
const CAPTION_FILLS = { cream: CREAM, orange: ORANGE };
const captionBaseSx = {
  position: "absolute",
  top: -9,
  zIndex: 3,
  pointerEvents: "none",
  userSelect: "none",
  whiteSpace: "nowrap",
  fontFamily: ANTON,
  fontSize: "0.58rem",
  letterSpacing: "0.09em",
  textTransform: "uppercase",
  lineHeight: 1,
  padding: "3px 6px 2px",
  color: COMIC_INK,
  border: `2px solid ${COMIC_INK}`,
  borderRadius: 0,
  boxShadow: `2px 2px 0 ${COMIC_INK}`,
};

function HudPlate({ caption, captionSide = "left", captionFill = "cream", children, sx }) {
  return (
    <Box sx={{ ...plateSx, ...sx }}>
      <Box
        sx={{
          display: "inline-flex",
          alignItems: "stretch",
          height: "100%",
          overflow: "hidden",
        }}
      >
        {children}
      </Box>
      {caption ? (
        <Box
          aria-hidden
          sx={{
            ...captionBaseSx,
            background: CAPTION_FILLS[captionFill],
            ...(captionSide === "right" ? { right: -8 } : { left: -8 }),
          }}
        >
          {caption}
        </Box>
      ) : null}
    </Box>
  );
}

function BackArrowIcon() {
  return (
    <Box
      component="svg"
      viewBox="0 0 24 24"
      aria-hidden
      fill="none"
      stroke="currentColor"
      strokeWidth={2.25}
      strokeLinecap="square"
      strokeLinejoin="miter"
      sx={{ width: "0.95em", height: "0.95em", display: "block", flex: "none" }}
    >
      <path d="M20 12H6" />
      <path d="M12 5l-7 7 7 7" />
    </Box>
  );
}

const hudHitSx = {
  appearance: "none",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  gap: "0.45em",
  height: "100%",
  px: "0.95rem",
  margin: 0,
  border: "none",
  borderRadius: 0,
  background: "transparent",
  color: CREAM,
  cursor: "pointer",
  fontFamily: ANTON,
  fontSize: "0.92rem",
  letterSpacing: "0.08em",
  textTransform: "uppercase",
  textDecoration: "none",
  lineHeight: 1,
  whiteSpace: "nowrap",
  WebkitTapHighlightColor: "transparent",
  userSelect: "none",
  transition: "background 0.15s ease, color 0.15s ease, filter 0.15s ease",
  "&:focus-visible": {
    outline: `2px dashed ${CREAM}`,
    outlineOffset: -5,
  },
};

function BackButton() {
  return (
    <HudPlate
      caption="Nav"
      sx={{ position: "fixed", top: "1.25rem", left: "1.5rem", zIndex: 10 }}
    >
      <Box
        component="button"
        type="button"
        onClick={() => useGame.setState({ menuOpen: true })}
        sx={{
          ...hudHitSx,
          "&:hover": {
            color: ORANGE,
            background: "rgba(255, 122, 47, 0.12)",
          },
          "&:active": { filter: "brightness(0.92)" },
        }}
      >
        <BackArrowIcon /> Back
      </Box>
    </HudPlate>
  );
}

const SHOP_URL = "https://store.pollen-robotics.com/collections/microduck";

function PreorderButton() {
  const text = useScramble("Pre-order");
  return (
    <HudPlate
      caption="Shop"
      captionSide="right"
      captionFill="orange"
      sx={{ position: "fixed", top: "1.25rem", right: "1.5rem", zIndex: 10 }}
    >
      <Box
        component="a"
        href={SHOP_URL}
        target="_blank"
        rel="noopener noreferrer"
        sx={{
          ...hudHitSx,
          background: ORANGE,
          color: COMIC_INK,
          "&:hover": { filter: "brightness(1.07)" },
          "&:active": { filter: "brightness(0.92)" },
        }}
      >
        {text}
      </Box>
    </HudPlate>
  );
}

// Immediate policy-state feedback. Several one-shot skills intentionally own
// the robot for a short window; showing that hand-off makes a valid keypress
// visible instead of making the controls feel unresponsive.
function ActionStatus() {
  const modeLabel = useGame((s) => s.modeLabel);
  const mouthOpen = useGame((s) => s.mouthOpen);
  const voiceLabel = useGame((s) => s.voiceLabel);
  const label = voiceLabel ?? (mouthOpen ? "Mouth open" : modeLabel);
  const scrambled = useScramble(label);
  // Mouth feedback is shorter than the 450 ms scramble, so render it
  // immediately; otherwise an entire quack would appear as random glyphs.
  const text = voiceLabel || mouthOpen ? label.toUpperCase() : scrambled;
  const active = label !== "Run" && label !== "Drive";
  return (
    <HudPlate
      caption="Action"
      sx={{
        position: "fixed",
        top: "1.25rem",
        left: "50%",
        transform: "translateX(-50%)",
        zIndex: 10,
        pointerEvents: "none",
      }}
    >
      <Box
        role="status"
        aria-live="polite"
        sx={{
          display: "flex",
          alignItems: "center",
          minWidth: "6.8rem",
          px: "0.9rem",
          color: active ? ORANGE : CREAM,
          fontFamily: ANTON,
          fontSize: "0.92rem",
          letterSpacing: "0.1em",
          lineHeight: 1,
          textTransform: "uppercase",
          justifyContent: "center",
          whiteSpace: "nowrap",
        }}
      >
        {text}
      </Box>
    </HudPlate>
  );
}

function Telemetry() {
  const t = useGame((s) => s.telemetry);
  const odo = t.odo < 1000 ? `${t.odo.toFixed(1)}M` : `${(t.odo / 1000).toFixed(2)}KM`;
  const lines = [];
  if (t.peers) lines.push(`${t.peers + 1} ONLINE`);
  lines.push(`${t.speed.toFixed(2)}M/S \u00b7 ODO ${odo}`);
  lines.push(`FPS ${t.fps} \u00b7 CTRL ${t.ctrlHz}HZ`);
  return (
    <Box
      sx={{
        position: "fixed",
        bottom: "1.25rem",
        right: "1.5rem",
        zIndex: 10,
        pointerEvents: "none",
        fontFamily: MONO,
        fontSize: "0.62rem",
        letterSpacing: "0.14em",
        textTransform: "uppercase",
        color: "rgba(255, 255, 255, 0.35)",
        fontVariantNumeric: "tabular-nums",
        textShadow: "0 0 6px rgba(255, 255, 255, 0.15)",
        whiteSpace: "pre-line",
        textAlign: "right",
        lineHeight: 1.8,
      }}
    >
      {lines.join("\n")}
    </Box>
  );
}

export default function Hud() {
  const entered = useGame((s) => s.entered);
  const menuOpen = useGame((s) => s.menuOpen);
  const touchMode = useGame((s) => s.touchMode);
  if (!entered || menuOpen) return null;
  return (
    <>
      <BackButton />
      <ActionStatus />
      <PreorderButton />
      <Box sx={{ position: "fixed", top: "5rem", left: "1.5rem", zIndex: 10,
        maxWidth: "19rem", color: CREAM, background: "rgba(8,8,12,0.8)", p: 1,
        font: "13px/1.5 system-ui", border: "1px solid #6a756e" }}>
        <Box>本模式 G 地面拾取仍属实验动作</Box>
        <Box component="a" href="/table-pick.html" sx={{ color: "#a0e8c1" }}>
          打开已验证的整机台沿叼取
        </Box>
      </Box>
      {!touchMode && (
        <>
          <Telemetry />
        </>
      )}
    </>
  );
}
