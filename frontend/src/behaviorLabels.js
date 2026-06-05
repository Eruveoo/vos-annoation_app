/** Behaviour label catalogs — must match server.py ids per dimension. */

export const BEHAVIOR_LABEL_NONE = "none";

export const BEHAVIOR_DIMENSIONS = ["activity", "label2", "label3"];

export const BEHAVIOR_DIMENSION_TITLES = {
  activity: "Label 1: Aktivisuus",
  label2: "Label 2: Toimija / syöminen / hoito",
  label3: "Label 3: Vastaanottaja",
};

export const BEHAVIOR_LABELS_ACTIVITY = [
  {
    id: "walk",
    nameFi: "Kävelee",
    descriptionFi:
      "Eläin ottaa useita askeleita siirtyäkseen paikasta toiseen. Eläin liikkuu kohtalaisen hitaasti siirtäen yhtä jalkaa kerrallaan eteenpäin. Myös peruuttaminen ja kääntyminen.",
  },
  {
    id: "trot_gallop",
    nameFi: "Ravaa tai laukkaa",
    descriptionFi:
      "Eläin ottaa useita askeleita siirtyäkseen paikasta toiseen. Eläin liikkuu kohtalaisen nopeasti tai nopeasti joko symmetrisesti siirtäen eri puolen etu- ja takajalkaa (diagonaalisia jalkoja) yhtä aikaa eteenpäin (ravi) tai siirtäen eteenpäin yhtä takajalkaa, toista takajalkaa yhtä aikaa vastakkaisen puoleisen etujalan (diagonaaliset jalat) kanssa ja toista etujalkaa, mitä seuraa liitovaihe, jolloin kaikki sorkat ovat irti maasta (laukka).",
  },
  {
    id: "stand",
    nameFi: "Seisoo",
    descriptionFi:
      "Eläin seisoo paikoillaan vähintään kolmen jalan ollessa kosketuksissa maahan.",
  },
  {
    id: "lie_down",
    nameFi: "Makuulle laskeutuminen",
    descriptionFi:
      "Makuulle laskeutuminen alkaa, kun yksi eläimen kyynärnivel taipuu ja laskeutuu (ennen maahan kosketusta). Koko makuulle laskeutumisliike päättyy, kun eläimen takapuoli on laskenut maahan ja eläin on vetänyt etujalan alta.",
  },
  {
    id: "get_up",
    nameFi: "Ylös nouseminen",
    descriptionFi:
      "Ylös nouseminen alkaa eläimen kohottaessa päätään hiukan ylöspäin ja jännittäessä etuosansa lihaksia. Eläin heilauttaa päätään voimakkaasti alakautta eteenpäin ja nostaa samalla takapäätään niin, että se saa asetettua takajalat vartalon alle. Seuraavaksi eläin nojautuu etupolviensa varaan ja suoristaa takajalkansa. Lopuksi eläin ottaa toisella etujalallaan askelen eteenpäin ja nousee seisomaan.",
  },
  {
    id: "abnormal_motion",
    nameFi: "Epänormaalit liikesarjat",
    descriptionFi:
      "Laskeutuessaan makuulle eläin laskee ensin takapäänsä maahan, käyden istuvassa asennossa ennen etupolvien koukistamista ja etupään laskemista maahan. Ylös noustessaan eläin ojentaa ensin etujalkansa päätyen istuvaan asentoon, josta se ponnistaa taka-osansa ylös.",
  },
  {
    id: "lying",
    nameFi: "Makaa",
    descriptionFi:
      "Eläimen vartalo lepää maassa alemmanpuoleisen takajalan ja reiden, vatsan ja etujalkojen tai toisen kyljen varassa.",
  },
  {
    id: "other_posture",
    nameFi: "Muu asento",
    descriptionFi: "Esim. selkään hyppääminen (kahdella jalalla), kaatuminen tai kompurointi.",
  },
  {
    id: "not_visible",
    nameFi: "Ei näkyvissä",
    descriptionFi: "Eläin ei ole kuvassa (poistui ruudusta).",
  },
];

export const BEHAVIOR_LABELS_LABEL2 = [
  {
    id: BEHAVIOR_LABEL_NONE,
    nameFi: "Ei valittu",
    descriptionFi: "Ei käyttäytymistä tässä kategoriassa (oletus).",
    groupFi: "",
  },
  {
    id: "inactive_ruminate",
    nameFi: "Toimeton tai märehtii (ei labelia)",
    descriptionFi: "Eläin on toimeton tai märehtii; ei muuta Label 2 -käyttäytymistä.",
    groupFi: "",
  },
  {
    id: "feed_head_down",
    nameFi: "Ruokintapöydällä pää alhaalla",
    descriptionFi:
      "Eläin seisoo pää ruokintapöydän vieressä pää ruokintaesteen etupuolella, pää alhaalla (syö, tutkii rehua ym.). Nopea rehun heittely sisältyy.",
    groupFi: "Syömiskäyttäytyminen",
  },
  {
    id: "feed_head_up",
    nameFi: "Ruokintapöydällä pää ylhäällä",
    descriptionFi:
      "Eläin seisoo pää ruokintapöydän vieressä pää ruokintaesteen etupuolella, pää ylhäällä (pureskelee, on toimeton ym.).",
    groupFi: "Syömiskäyttäytyminen",
  },
  {
    id: "drink",
    nameFi: "Juo",
    descriptionFi:
      "Juo, laskee päänsä kuppiin/altaaseen – nostaa sen pois kupista/altaasta. Sisältää muutamien sekuntien tauot.",
    groupFi: "Syömiskäyttäytyminen",
  },
  {
    id: "groom_self",
    nameFi: "Kehon hoito itse",
    descriptionFi: "Nuolee, raapii tai hankaa itseään.",
    groupFi: "Kehon hoito",
  },
  {
    id: "scratch_neck_rail",
    nameFi: "Rapsuttelu karjaharjalla",
    descriptionFi: "Kehon rapsuttelu karjaharjaan.",
    groupFi: "Kehon hoito",
  },
  {
    id: "scratch_other",
    nameFi: "Rapsuttelu muuhun",
    descriptionFi: "Kehon rapsuttelu muuhun kuin karjaharjaan (parret ym.).",
    groupFi: "Kehon hoito",
  },
  {
    id: "social_lick_actor",
    nameFi: "Sosiaalinen nuoleminen (toimija)",
    descriptionFi: "Eläin nuolee toista eläintä yleensä päästä, kaulasta tai hartioista.",
    groupFi: "Sosiaalinen käyttäytyminen",
  },
  {
    id: "pushing",
    nameFi: "Puskeminen",
    descriptionFi:
      "Eläin sysää otsan tai pään ylöspäin suuntautuvalla liikkeellä vasten toisen eläimen niskaa, hartioita, kylkeä tai takaosaa.",
    groupFi: "Sosiaalinen käyttäytyminen",
  },
  {
    id: "displacement_actor",
    nameFi: "Syrjäyttäminen (toimija)",
    descriptionFi:
      "Eläin puskee tai esim. vartalollaan tönimällä syrjäyttää toisen eläimen pois ruokintapaikalta, juomakupilta tai makuupaikalta.",
    groupFi: "Sosiaalinen käyttäytyminen",
  },
  {
    id: "chin_rest_actor",
    nameFi: "Leuan lepuuttaminen (toimija)",
    descriptionFi:
      "Eläin testaa toisen lehmän seisomisrefleksiä ennen selkään hyppäämistä painamalla leukaansa ja kurkkuaan lehmän takapuolen tai selän päälle.",
    groupFi: "Kiimakäyttäytyminen",
  },
  {
    id: "mount_actor",
    nameFi: "Selkään hyppääminen (toimija)",
    descriptionFi:
      "Eläin ponnistaa etujalkansa irti maasta ja nostaa ryntäänsä toisen lehmän selän päälle, sijoittaen etujalkansa juuri lehmän lonkkakyhmyjen etupuolelle pitäen kiinni lehmästä.",
    groupFi: "Kiimakäyttäytyminen",
  },
  {
    id: "other_label2",
    nameFi: "Muu",
    descriptionFi: "Ei ole mitään yllä mainittua (ei toimeton tai märehdi).",
    groupFi: "Kiimakäyttäytyminen",
  },
  {
    id: "not_seen",
    nameFi: "Ei näy",
    descriptionFi: "Käyttäytymistä ei näe.",
    groupFi: "",
  },
];

export const BEHAVIOR_LABELS_LABEL3 = [
  {
    id: BEHAVIOR_LABEL_NONE,
    nameFi: "Ei valittu",
    descriptionFi: "Ei käyttäytymistä tässä kategoriassa (oletus).",
    groupFi: "",
  },
  {
    id: "social_lick_receiver",
    nameFi: "Sosiaalinen nuoleminen (vastaanottaja)",
    descriptionFi:
      "Toinen eläin nuolee eläintä yleensä päästä, kaulasta tai hartioista. Nuoltavana oleva eläin ojentaa usein kaulaansa ja päätään eteen.",
    groupFi: "Sosiaalinen käyttäytyminen",
  },
  {
    id: "displacement_receiver",
    nameFi: "Syrjäyttäminen (vastaanottaja)",
    descriptionFi:
      "Toinen eläin puskee tai esim. vartalollaan tönimällä syrjäyttää eläimen pois ruokintapaikalta, juomakupilta tai makuupaikalta. Syrjäytettävä siirtyy noin lehmän mitan pois.",
    groupFi: "Sosiaalinen käyttäytyminen",
  },
  {
    id: "chin_rest_receiver",
    nameFi: "Leuan lepuuttaminen (vastaanottaja)",
    descriptionFi: "Toinen eläin lepuuttaa leukaa tämän lehmän selän tai takapuolen päällä.",
    groupFi: "Kiimakäyttäytyminen",
  },
  {
    id: "mount_receiver",
    nameFi: "Selkään hyppääminen (vastaanottaja)",
    descriptionFi: "Toinen eläin hyppää tämän lehmän selkään.",
    groupFi: "Kiimakäyttäytyminen",
  },
];

export const BEHAVIOR_LABELS_BY_DIMENSION = {
  activity: BEHAVIOR_LABELS_ACTIVITY,
  label2: BEHAVIOR_LABELS_LABEL2,
  label3: BEHAVIOR_LABELS_LABEL3,
};

export const DEFAULT_BEHAVIOR_LABEL_ID = "stand";
export const DEFAULT_LABEL2_LABEL_ID = BEHAVIOR_LABEL_NONE;
export const DEFAULT_LABEL3_LABEL_ID = BEHAVIOR_LABEL_NONE;

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
