TYPE_TO_TAG = {
"AKTIONSART":  ["ACCMP", "ACH", "ACTY", "ATEL", "DUR", "DYN", "PCT", "SEMEL", "STAT", "TEL"],
"ANIMACY":  ["ANIM", "HUM", "INAN", "NHUM"],
"ASPECT":  ["HAB", "IPFV", "ITER", "PFV", "PRF", "PROG", "PROSP"],
"CASE":  ["ABL", "ABS", "ACC", "ALL", "ANTE", "APPRX", "APUD", "AT", "AVR", "BEN", "BYWAY", "CIRC",
        "COM", "COMPV", "DAT", "EQTV", "ERG", "ESS", "FRML", "GEN", "IN", "INS", "INTER", "NOM",
        "NOMS", "ON", "ONHR", "ONVR", "POST", "PRIV", "PROL", "PROPR", "PROX", "PRP", "PRT", "REL",
        "REM", "SUB", "TERM", "TRANS", "VERS", "VOC"],
"COMPARISON":  ["AB", "CMPR", "EQT", "RL", "SPRL"],
"DEFINITENESS":  ["DEF", "INDF", "NSPEC", "SPEC"],
"DEIXIS":  ["ABV", "BEL", "EVEN", "MED", "NOREF", "NVIS", "PHOR", "PROX", "REF1", "REF2", "REMT", "VIS"],
"EVIDENTIALITY":  ["ASSUM", "AUD", "DRCT", "FH", "HRSY", "INFER", "NFH", "NVSEN", "QUOT", "RPRT", "SEN"],
"FINITENESS":  ["FIN", "NFIN", "{FIN/NFIN}"],
"GENDER":  ["BANTU1-23", "FEM", "MASC", "NAKH1-8", "NEUT"],
"INFORMATION_STRUCTURE":  ["FOC", "TOP"],
"INTERROGATIVITY":  ["DECL", "INT"],
"MOOD":  ["ADM", "AUNPRP", "AUPRP", "COND", "DEB", "DED", "IMP", "IND", "INTEN", "IRR", "LKLY", "OBLIG",
        "OPT", "PERM", "POT", "PURP", "REAL", "SBJV", "SIM"],
"NUMBER":  ["DU", "GPAUC", "GRPL", "INVN", "PAUC", "PL", "SG"],
"POS":  ["ADJ", "ADP", "ADV", "ART", "AUX", "CLF", "COMP", "CONJ", "DET", "INTJ", "N", "NUM", "PART", "PRO",
       "PROPN", "V", "V.CVB", "V.MSDR", "V.PTCP"],
"PERSON":  ["0", "1", "2", "3", "4", "EXCL", "INCL", "OBV", "PRX"],
"POLARITY":  ["POS", "NEG"],
"POLITENESS":  ["AVOID", "COL", "ELEV", "FOREG", "FORM", "HIGH", "HUMB", "INFM", "LIT", "LOW", "POL", "STELEV", "STSUPR"],
"POSSESSION":  ["ALN", "NALN", "PSS1D", "PSS1DE", "PSS1DI", "PSS1P", "PSS1PE", "PSS1PI", "PSS1S", "PSS2D", "PSS2DF",
              "PSS2DM", "PSS2P", "PSS2PF", "PSS2PM", "PSS2S", "PSS2SF", "PSS2SFORM", "PSS2SINFM", "PSS2SM", "PSS3D",
              "PSS3DF", "PSS3DM", "PSS3P", "PSS3PF", "PSS3PM", "PSS3S", "PSS3SF", "PSS3SM", "PSSD"],
"SWITCH_REFERENCE":  ["CN_R_MN", "DS", "DSADV", "LOG", "OR", "SEQMA", "SEMMA", "SS", "SSADV"],
"TENSE":  ["1DAY", "FUT", "HOD", "IMMED", "PRS", "PST", "RCT", "RMT"],
"VALENCY":  ["APPL", "CAUS", "DITR", "IMPRS", "INTR", "RECP", "REFL", "TR"],
"VOICE":  ["ACFOC", "ACT", "AGFOC", "ANTIP", "BFOC", "CFOC", "DIR", "IFOC", "INV", "LFOC", "MID", "PASS", "PFOC"]}

def get_type_to_tag(ty):
    return TYPE_TO_TAG[ty]

def get_tag_to_type(tag):
    for key in TYPE_TO_TAG:
        if tag in TYPE_TO_TAG[key]:
            return key

def get_all_tags():
    tags = []
    for ty in TYPE_TO_TAG:
        tags.extend(TYPE_TO_TAG[ty])
    return tags
