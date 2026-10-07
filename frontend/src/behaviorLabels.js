/**
 * Behaviour label catalogs (shared with the backend): backend/config/behavior_labels.local.json
 * if present (git-ignored), otherwise the placeholder backend/config/behavior_labels.json.
 */
import placeholderConfig from "../../backend/config/behavior_labels.json";

const localConfig = Object.values(
  import.meta.glob("../../backend/config/behavior_labels.local.json", { eager: true, import: "default" })
)[0];
const behaviorConfig = localConfig || placeholderConfig;

export const BEHAVIOR_LABEL_NONE = "none";

export const BEHAVIOR_DIMENSIONS = ["activity", "label2", "label3"];

function toFrontendLabel(label) {
  const out = {
    id: label.id,
    nameFi: label.name_fi,
    descriptionFi: label.description_fi || "",
  };
  if (label.group_fi !== undefined) out.groupFi = label.group_fi;
  return out;
}

const dimensionConfig = behaviorConfig.dimensions;

export const BEHAVIOR_DIMENSION_TITLES = Object.fromEntries(
  BEHAVIOR_DIMENSIONS.map((dim) => [dim, dimensionConfig[dim].title_fi])
);

export const BEHAVIOR_DIMENSION_SHORT_TITLES = Object.fromEntries(
  BEHAVIOR_DIMENSIONS.map((dim) => [dim, dimensionConfig[dim].short_title_fi || dimensionConfig[dim].title_fi])
);

export const BEHAVIOR_LABELS_BY_DIMENSION = Object.fromEntries(
  BEHAVIOR_DIMENSIONS.map((dim) => [dim, dimensionConfig[dim].labels.map(toFrontendLabel)])
);

export const BEHAVIOR_LABELS_ACTIVITY = BEHAVIOR_LABELS_BY_DIMENSION.activity;
export const BEHAVIOR_LABELS_LABEL2 = BEHAVIOR_LABELS_BY_DIMENSION.label2;
export const BEHAVIOR_LABELS_LABEL3 = BEHAVIOR_LABELS_BY_DIMENSION.label3;

export const DEFAULT_BEHAVIOR_LABEL_ID = dimensionConfig.activity.default_label;
export const DEFAULT_LABEL2_LABEL_ID = dimensionConfig.label2.default_label;
export const DEFAULT_LABEL3_LABEL_ID = dimensionConfig.label3.default_label;

export const ANNOTATION_MODES = {
  STANDARD: "standard",
  BEHAVIOR: "behavior",
};

export function labelNameFi(labelId, dimension = "activity") {
  const catalog = BEHAVIOR_LABELS_BY_DIMENSION[dimension] || BEHAVIOR_LABELS_ACTIVITY;
  const found = catalog.find((l) => l.id === labelId);
  return found ? found.nameFi : labelId;
}

export function labelsForInitSelect(dimension) {
  const catalog = BEHAVIOR_LABELS_BY_DIMENSION[dimension];
  if (dimension === "activity") {
    return catalog.filter((l) => l.id !== "not_visible");
  }
  return catalog;
}
