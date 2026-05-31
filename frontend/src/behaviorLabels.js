/** Cow behaviour labels (UI names). Must match backend server.py label ids. */

export const BEHAVIOR_LABELS = [
  {
    id: "walk",
    nameFi: "Kävelee",
    descriptionFi:
      "Eläin ottaa useita askeleita siirtyäkseen paikasta toiseen. Eläin liikkuu kohtalaisen hitaasti siirtäen yhtä jalkaa kerrallaan eteenpäin.",
  },
  {
    id: "trot_gallop",
    nameFi: "Ravaa tai laukkaa",
    descriptionFi: "Eläin liikkuu nopeasti ravissa tai laukaten.",
  },
  {
    id: "stand",
    nameFi: "Seisoo",
    descriptionFi: "Eläin seisoo paikoillaan vähintään kolmen jalan ollessa kosketuksissa maahan.",
  },
  {
    id: "lie_down",
    nameFi: "Makuulle laskeutuminen",
    descriptionFi: "Makuulle laskeutumisen liikesarja.",
  },
  {
    id: "get_up",
    nameFi: "Ylös nouseminen",
    descriptionFi: "Ylös nousemisen liikesarja.",
  },
  {
    id: "abnormal_motion",
    nameFi: "Epänormaalit liikesarjat",
    descriptionFi: "Epänormaali laskeutuminen tai nouseminen.",
  },
  {
    id: "lying",
    nameFi: "Makaa",
    descriptionFi: "Eläimen vartalo lepää maassa.",
  },
  {
    id: "other_posture",
    nameFi: "Muu asento",
    descriptionFi: "Esim. selkään hyppääminen, kaatuminen tai kompurointi.",
  },
  {
    id: "not_visible",
    nameFi: "Ei näkyvissä",
    descriptionFi: "Eläin ei ole kuvassa.",
  },
];

export const DEFAULT_BEHAVIOR_LABEL_ID = "stand";

export const ANNOTATION_MODES = {
  STANDARD: "standard",
  BEHAVIOR: "behavior",
};

export function labelNameFi(labelId) {
  const found = BEHAVIOR_LABELS.find((l) => l.id === labelId);
  return found ? found.nameFi : labelId;
}
