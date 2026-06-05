/** Stable colors per label id (per dimension). */

const PALETTE = [
  "#4e79a7",
  "#f28e2b",
  "#e15759",
  "#76b7b2",
  "#59a14f",
  "#edc948",
  "#b07aa1",
  "#ff9da7",
  "#9c755f",
  "#bab0ac",
  "#86bcb6",
  "#d37295",
  "#8cd17d",
  "#b6992d",
  "#499894",
];

const NEUTRAL_COLOR = "#e9ecef";
const NEUTRAL_IDS = new Set(["none", "not_visible", "not_seen"]);

function hashString(s) {
  let h = 0;
  for (let i = 0; i < s.length; i += 1) {
    h = (h * 31 + s.charCodeAt(i)) | 0;
  }
  return Math.abs(h);
}

export function colorForLabelId(dimension, labelId) {
  if (!labelId || NEUTRAL_IDS.has(labelId)) {
    return NEUTRAL_COLOR;
  }
  const idx = hashString(`${dimension}:${labelId}`) % PALETTE.length;
  return PALETTE[idx];
}

export function isNeutralLabelId(labelId) {
  return !labelId || NEUTRAL_IDS.has(labelId);
}
