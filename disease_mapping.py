"""
disease_mapping.py — model output class → diseases, severity, description, action.

Hard classes (lymphoblast / reactive_lymphocyte / lymphocyte) are annotated with
a note explaining the distinction, since the model now receives extra augmentation
for these and may still report them as close alternatives to each other.

Edit DISEASE_MAP freely as your dataset grows.
"""

DISEASE_MAP = {
    # ── WBC / Leukemia ───────────────────────────────────────────────────────
    "myeloblast": {
        "diseases": ["Acute Myeloid Leukemia (AML)"],
        "severity": "critical",
        "description": (
            "Myeloblasts are immature myeloid precursors that should not appear "
            "in peripheral blood. Their presence strongly suggests AML, a rapidly "
            "progressing blood cancer requiring immediate treatment."
        ),
        "action": "Immediate hematology referral required. Do not delay.",
        "hard_class": False,
    },
    "lymphoblast": {
        "diseases": ["Acute Lymphoblastic Leukemia (ALL)"],
        "severity": "critical",
        "description": (
            "Lymphoblasts are immature lymphoid precursors. They resemble reactive "
            "lymphocytes but have a higher nucleus-to-cytoplasm ratio and more "
            "prominent nucleoli. Elevated counts indicate ALL, most common in children "
            "but also seen in adults. Confirm with flow cytometry."
        ),
        "action": "Urgent oncology consultation. Confirm with bone marrow biopsy and flow cytometry.",
        "hard_class": True,
        "hard_note": "Easily confused with reactive_lymphocyte and lymphocyte — always confirm with lab tests.",
    },
    "neutrophil": {
        "diseases": ["Bacterial infection", "Inflammation", "Normal (if in range)"],
        "severity": "low",
        "description": (
            "Neutrophils are the most abundant WBC and the first responders to bacterial "
            "infection. Counts above 7.5×10⁹/L (neutrophilia) suggest infection or "
            "inflammation; below 1.5×10⁹/L (neutropenia) raises infection risk."
        ),
        "action": "Correlate with clinical symptoms, temperature, and CRP/ESR levels.",
        "hard_class": False,
    },
    "eosinophil": {
        "diseases": ["Allergic reaction", "Parasitic infection", "Eosinophilia"],
        "severity": "medium",
        "description": (
            "Eosinophils defend against parasites and mediate allergic responses. "
            "Counts above 0.5×10⁹/L suggest an allergic or parasitic cause; "
            "above 1.5×10⁹/L constitutes hypereosinophilia with potential organ damage."
        ),
        "action": "Review allergy and travel history. Order stool O&P and IgE panel.",
        "hard_class": False,
    },
    "basophil": {
        "diseases": ["Chronic Myeloid Leukemia (CML)", "Allergic inflammation"],
        "severity": "medium",
        "description": (
            "Basophils are rare in normal blood (<0.1×10⁹/L). Persistent basophilia "
            "is a hallmark feature of CML and should not be ignored even at low levels."
        ),
        "action": "Order BCR-ABL PCR test if basophilia is persistent. Refer to hematology.",
        "hard_class": False,
    },
    "monocyte": {
        "diseases": ["Chronic infection", "Monocytic leukemia", "Inflammatory disease"],
        "severity": "medium",
        "description": (
            "Monocytes are long-lived phagocytes. Monocytosis (>0.8×10⁹/L) persisting "
            "for weeks suggests chronic infection, autoimmune disease, or rarely "
            "monocytic leukemia (CMML)."
        ),
        "action": "Assess for TB, CMV, EBV, or autoimmune conditions. Repeat CBC in 4 weeks.",
        "hard_class": False,
    },
    "lymphocyte": {
        "diseases": ["Viral infection", "Chronic Lymphocytic Leukemia (CLL)", "Normal"],
        "severity": "low",
        "description": (
            "Lymphocytes are the primary adaptive immune cells. A transient rise is "
            "normal during viral illness. Persistent lymphocytosis (>4×10⁹/L in adults) "
            "with small, mature-looking cells may indicate CLL."
        ),
        "action": "Recheck CBC in 4–6 weeks. Investigate with flow cytometry if count stays elevated.",
        "hard_class": True,
        "hard_note": "Morphologically similar to lymphoblast and reactive_lymphocyte. Context and lab confirmation are essential.",
    },
    "lymphocyte_reactive": {
        "diseases": ["EBV (Infectious Mononucleosis)", "CMV", "Hepatitis", "COVID-19"],
        "severity": "medium",
        "description": (
            "Reactive (atypical) lymphocytes are activated T-cells with abundant pale "
            "cytoplasm that molds around adjacent RBCs. They indicate active viral "
            "infection and are the hallmark of infectious mononucleosis."
        ),
        "action": "Order Monospot test, EBV IgM/IgG, CMV serology, and hepatitis panel.",
        "hard_class": True,
        "hard_note": "Can be confused with lymphoblast — key difference is cytoplasm volume and lack of prominent nucleoli.",
    },
    "normoblast": {
        "diseases": ["Severe anemia", "Bone marrow stress", "Thalassemia", "Myelofibrosis"],
        "severity": "high",
        "description": (
            "Nucleated red blood cells (normoblasts) should not appear in adult peripheral "
            "blood. Their presence signals severe bone marrow stress, hemolysis, or "
            "infiltrative marrow disease."
        ),
        "action": "Urgent hematology evaluation. Order reticulocyte count and bone marrow assessment.",
        "hard_class": False,
    },
    "myelocyte": {
    "diseases": ["Chronic Myeloid Leukemia (CML)", "Myeloid metaplasia", "Severe infection"],
    "severity": "high",
    "description": (
        "Myelocytes are immature granulocyte precursors normally confined to the bone marrow. "
        "Their presence in peripheral blood indicates bone marrow stress or abnormal myeloid "
        "proliferation, most commonly seen in CML or severe systemic infection."
    ),
    "action": "Order BCR-ABL PCR and bone marrow evaluation. Refer to hematology.",
    "hard_class": False,
    },

    # ── RBC anomalies ─────────────────────────────────────────────────────────
    "sickle_cell": {
        "diseases": ["Sickle Cell Anemia (SCA)", "HbS trait"],
        "severity": "high",
        "description": (
            "Sickle (crescent) shaped RBCs result from polymerization of abnormal "
            "HbS hemoglobin. They obstruct capillaries, causing vaso-occlusive pain "
            "crises, stroke, and progressive organ damage."
        ),
        "action": "Confirm with hemoglobin electrophoresis. Genetic counseling and hematology follow-up.",
        "hard_class": False,
    },
    "target_cell": {
        "diseases": ["Thalassemia", "Iron deficiency anemia", "Liver disease", "HbC disease"],
        "severity": "medium",
        "description": (
            "Target cells (codocytes) have excess membrane relative to cell volume, "
            "creating a bullseye appearance. They appear in hemoglobin disorders, "
            "obstructive liver disease, and post-splenectomy states."
        ),
        "action": "Order hemoglobin electrophoresis and liver function tests.",
        "hard_class": False,
    },
    "schistocyte": {
        "diseases": ["TTP", "HUS", "DIC", "Microangiopathic hemolytic anemia (MAHA)"],
        "severity": "critical",
        "description": (
            "Schistocytes are red cell fragments produced by mechanical shearing inside "
            "damaged or narrowed vessels. Even a small percentage (>1%) is clinically "
            "significant and demands urgent evaluation."
        ),
        "action": "Emergency evaluation. TTP requires immediate plasmapheresis — delays are fatal.",
        "hard_class": False,
    },

    # ── Platelet anomalies ────────────────────────────────────────────────────
    "platelet_clump": {
        "diseases": ["Pseudothrombocytopenia", "Platelet activation"],
        "severity": "low",
        "description": (
            "Platelet clumps form in EDTA anticoagulated blood in some individuals "
            "(EDTA-dependent pseudothrombocytopenia). They cause falsely low platelet "
            "counts on automated analyzers but carry no clinical risk."
        ),
        "action": "Recheck platelet count using sodium citrate tube. Manual smear review recommended.",
        "hard_class": False,
    },

    # ── Parasites ─────────────────────────────────────────────────────────────
    "malaria_ring": {
        "diseases": ["Malaria (Plasmodium falciparum / vivax / malariae / ovale)"],
        "severity": "critical",
        "description": (
            "Ring-form trophozoites inside RBCs are diagnostic of malaria. "
            "P. falciparum can cause cerebral malaria and multi-organ failure "
            "within hours if untreated."
        ),
        "action": "Start antimalarial treatment immediately. Notify public health authority if required.",
        "hard_class": False,
    },

    # ── Normal ────────────────────────────────────────────────────────────────
    "normal": {
        "diseases": [],
        "severity": "none",
        "description": (
            "Cells appear morphologically normal with no detectable anomalies. "
            "This indicates healthy cell morphology in the analysed region of the smear. "
            "A normal result here does not replace a full CBC or clinical assessment."
        ),
        "action": "No immediate action required. Routine follow-up as clinically indicated.",
        "hard_class": False,
    },
}

SEVERITY_EMOJI = {
    "none":     "✅",
    "low":      "🟡",
    "medium":   "🟠",
    "high":     "🔴",
    "critical": "🚨",
}

# Classes that receive extra augmentation during training
HARD_CLASSES = {"lymphoblast", "reactive_lymphocyte", "lymphocyte"}


def get_disease_info(class_name: str) -> dict:
    """Return disease info for a predicted class. Falls back gracefully for unknown classes."""
    return DISEASE_MAP.get(
        class_name,
        {
            "diseases": ["Unknown — class not in mapping"],
            "severity": "medium",
            "description": f"Class '{class_name}' has no disease mapping yet.",
            "action": "Consult a hematologist and add this class to disease_mapping.py.",
            "hard_class": False,
        },
    )


def is_hard_class(class_name: str) -> bool:
    """Returns True if this class is in the hard/ambiguous group."""
    return DISEASE_MAP.get(class_name, {}).get("hard_class", False)


def get_hard_note(class_name: str) -> str | None:
    """Returns the disambiguation note for hard classes, or None."""
    return DISEASE_MAP.get(class_name, {}).get("hard_note", None)
