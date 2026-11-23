"""Build the Bird Voice Recognition System project report as a DOCX file."""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


OUT = Path("docs/Bird_Voice_Recognition_Project_Report.docx")
BLUE = "17365D"
PALE_BLUE = "DCE6F1"
LIGHT = "F2F6FA"


def shade(cell, fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    properties.append(shading)


def borders(table) -> None:
    properties = table._tbl.tblPr
    border = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        item = OxmlElement(f"w:{side}")
        item.set(qn("w:val"), "single")
        item.set(qn("w:sz"), "6")
        item.set(qn("w:color"), "D9D9D9")
        border.append(item)
    properties.append(border)


def set_cell_text(cell, text: str, bold: bool = False, color: str = "000000", size: int = 10) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(0)
    run = paragraph.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    paragraph = doc.add_heading(text, level=level)
    for run in paragraph.runs:
        run.font.color.rgb = RGBColor(0, 0, 0)
    paragraph.paragraph_format.space_before = Pt(14 if level == 1 else 8)
    paragraph.paragraph_format.space_after = Pt(6)


def add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        paragraph = doc.add_paragraph(style="List Bullet")
        paragraph.add_run(item)
        paragraph.paragraph_format.space_after = Pt(3)


def add_process_box(doc: Document, title: str, body: str) -> None:
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    borders(table)
    cell = table.cell(0, 0)
    shade(cell, PALE_BLUE)
    cell.text = ""
    title_p = cell.paragraphs[0]
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_p.add_run(title)
    title_run.bold = True
    title_run.font.color.rgb = RGBColor.from_string(BLUE)
    title_run.font.size = Pt(11)
    body_p = cell.add_paragraph()
    body_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    body_p.paragraph_format.space_after = Pt(2)
    body_run = body_p.add_run(body)
    body_run.font.size = Pt(9)
    arrow = doc.add_paragraph("↓")
    arrow.alignment = WD_ALIGN_PARAGRAPH.CENTER
    arrow.paragraph_format.space_before = Pt(1)
    arrow.paragraph_format.space_after = Pt(1)
    arrow.runs[0].font.size = Pt(14)
    arrow.runs[0].font.color.rgb = RGBColor.from_string(BLUE)


def add_table(doc: Document, headers: list[str], rows: list[list[str]], widths: list[float] | None = None) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    borders(table)
    for index, header in enumerate(headers):
        shade(table.rows[0].cells[index], BLUE)
        set_cell_text(table.rows[0].cells[index], header, bold=True, color="FFFFFF")
        if widths:
            table.rows[0].cells[index].width = Inches(widths[index])
    for row_index, row in enumerate(rows):
        cells = table.add_row().cells
        for index, value in enumerate(row):
            if row_index % 2 == 1:
                shade(cells[index], LIGHT)
            set_cell_text(cells[index], value)
            if widths:
                cells[index].width = Inches(widths[index])
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def main() -> None:
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.top_margin = section.bottom_margin = Inches(0.75)
    section.left_margin = section.right_margin = Inches(0.8)
    styles = doc.styles
    styles["Normal"].font.name = "Aptos"
    styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "Aptos")
    styles["Normal"].font.size = Pt(11)
    styles["Normal"].paragraph_format.space_after = Pt(6)
    styles["Title"].font.name = "Aptos Display"
    styles["Title"].font.size = Pt(24)
    styles["Title"].font.color.rgb = RGBColor(0, 0, 0)

    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("Bird Voice Recognition System")
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.add_run("Project Report and Technical Architecture").bold = True
    subtitle.runs[0].font.size = Pt(14)
    subtitle.paragraph_format.space_after = Pt(14)
    status = doc.add_paragraph()
    status.alignment = WD_ALIGN_PARAGRAPH.CENTER
    status.add_run("Implementation-ready design report | November 2026").italic = True
    status.paragraph_format.space_after = Pt(28)

    add_heading(doc, "Executive summary")
    doc.add_paragraph(
        "This report specifies a reproducible machine-learning pipeline for identifying bird vocalizations in noisy field recordings. "
        "The system transforms audio into log-mel spectrograms, applies a residual convolutional neural network to classify species, "
        "and produces a pitch-consistency diagnostic for error analysis. The repository contains the implementation, tests, experiment "
        "configuration, preprocessing scripts, inference entry point, and reporting scaffolding. It does not contain a downloaded dataset, "
        "trained checkpoint, measured metrics, or ecological findings."
    )
    doc.add_paragraph(
        "The intended benchmark layout is BirdCLEF 2021, which supplies focal recordings and labelled soundscape windows. The study must "
        "report measured performance only after a licensed dataset download and a completed training run."
    )

    add_heading(doc, "Project objectives")
    add_bullets(doc, [
        "Build a robust species-recognition baseline for noisy field recordings.",
        "Make preprocessing, data splits, model configuration, and evaluation reproducible.",
        "Test whether pitch consistency improves confidence calibration or error diagnosis.",
        "Validate a trained model on reviewed local recordings before making deployment or ecological claims.",
    ])

    doc.add_page_break()
    add_heading(doc, "System architecture")
    doc.add_paragraph("The architecture separates data handling, acoustic feature extraction, classification, diagnostic analysis, and reporting.")
    for title_text, body in [
        ("1  Source audio and metadata", "BirdCLEF-compatible OGG recordings and metadata manifest. Audio files remain outside Git."),
        ("2  Audio standardization", "Stereo to mono, resample to 32 kHz, peak normalize, then select or pad a five-second window."),
        ("3  Acoustic representation", "Create a 128-bin log-mel spectrogram using a 2048-point FFT and 512-sample hop length."),
        ("4  CNN classifier", "Residual convolutional blocks learn time-frequency patterns and output a species probability distribution."),
        ("5  Pitch diagnostic", "Autocorrelation estimates dominant periodic frequency; robust species priors support consistency analysis."),
        ("6  Evaluation and reporting", "Macro F1, PR-AUC, calibration error, class-level analysis, checkpoints, configs, and split manifests."),
    ]:
        add_process_box(doc, title_text, body)

    add_heading(doc, "Data preparation")
    doc.add_paragraph(
        "The preparation script reads the source metadata, validates the primary label and filename fields, constructs a relative audio path, "
        "and creates a stable species-to-integer label map. This keeps raw audio separate from version-controlled code while allowing every "
        "training run to identify its exact classes."
    )
    add_table(doc, ["Stage", "Operation", "Reason"], [
        ["Load", "Read OGG waveform with torchaudio", "Preserves original audio while allowing a consistent pipeline."],
        ["Channel handling", "Average channels into mono", "Avoids model dependence on microphone channel layout."],
        ["Resampling", "Convert to 32 kHz", "Standardizes temporal and frequency resolution."],
        ["Normalization", "Divide by peak amplitude", "Reduces gain differences between recorders."],
        ["Windowing", "Random five seconds for training; center window for validation", "Adds training variation while keeping validation deterministic."],
        ["Spectrogram", "Log-mel transform with 128 mel bins", "Represents vocal frequency patterns in a CNN-friendly format."],
        ["Augmentation", "Frequency and time masking during training", "Improves resilience to partial calls and localized interference."],
    ], [1.0, 2.55, 3.0])

    add_heading(doc, "Model design")
    doc.add_paragraph(
        "The classifier accepts a single-channel log-mel spectrogram. An initial 5 × 5 convolution learns local time-frequency features. "
        "Residual blocks retain information through skip connections while expanding the channel width from 32 to 64 and then 128. Adaptive "
        "global average pooling produces a fixed-length representation regardless of the number of time frames. Dropout regularizes the final "
        "linear classifier. The BirdCLEF 2021 configuration defines 397 output classes."
    )
    add_table(doc, ["Layer group", "Output role", "Design decision"], [
        ["Input", "1 × 128 × time", "Log-mel spectrogram rather than raw waveform."],
        ["Stem", "32 feature maps", "5 × 5 convolution, batch normalization, GELU activation."],
        ["Residual stages", "32, 64, then 128 feature maps", "Skip connections stabilize learning while downsampling."],
        ["Pooling", "128-dimensional embedding", "Adaptive global average pooling reduces spatial dimensions."],
        ["Classifier", "397 logits", "Dropout followed by a linear layer produces species probabilities."],
        ["Pitch head", "Scalar auxiliary output", "Reserved for controlled pitch-related experiments; not a substitute for CNN classification."],
    ], [1.2, 1.55, 3.8])

    add_heading(doc, "Training workflow")
    doc.add_paragraph(
        "The training entry point reads a YAML experiment configuration, fixes random seeds, creates a stratified development split, loads batches "
        "of spectrograms, and optimizes multiclass cross-entropy with AdamW. It evaluates macro F1 after each epoch and saves the best checkpoint. "
        "The checkpoint retains the model state, configuration, and measured metrics from that run."
    )
    add_table(doc, ["Component", "Configured baseline"], [
        ["Optimizer", "AdamW"],
        ["Learning rate", "0.0003"],
        ["Weight decay", "0.0001"],
        ["Batch size", "24"],
        ["Epochs", "20"],
        ["Primary model-selection metric", "Validation macro F1"],
    ], [2.55, 4.0])

    add_heading(doc, "Pitch consistency analysis")
    doc.add_paragraph(
        "Pitch is treated as a diagnostic signal, not a fixed identification rule. The pipeline computes autocorrelation of a normalized waveform, "
        "searches for periodic lags corresponding to approximately 80 to 2,000 Hz, and converts the strongest lag to a frequency estimate. "
        "A later script can compute the median and median absolute deviation of estimates for each species using training recordings only. "
        "Predictions with unusual pitch can then be flagged for review or evaluated in a calibration ablation. This is more defensible than declaring a species correct only when a single frequency matches."
    )

    doc.add_page_break()
    add_heading(doc, "Evaluation plan")
    add_table(doc, ["Question", "Metric or evidence", "Interpretation"], [
        ["Does the classifier handle imbalanced classes?", "Macro F1 and per-species recall", "Each species contributes equally to the primary score."],
        ["How good are predicted probabilities?", "Expected calibration error", "Confidence should align with empirical correctness."],
        ["Can it rank positive examples?", "Macro precision-recall AUC", "Useful when species classes have uneven prevalence."],
        ["Does it generalize beyond similar sources?", "Source-held-out split", "Reduces leakage from a shared recordist or recording context."],
        ["Does pitch help?", "CNN-only versus diagnostic/re-ranking ablation", "Report improvement, no change, or degradation by class."],
        ["Does it transfer locally?", "Manually reviewed campus sample", "Measures domain shift before deployment claims."],
    ], [2.2, 2.0, 2.35])

    add_heading(doc, "Repository implementation map")
    add_table(doc, ["Path", "Purpose"], [
        ["src/birdvoice/audio.py", "Audio normalization, windowing, log-mel transformation, and pitch estimation."],
        ["src/birdvoice/model.py", "Residual CNN classifier architecture."],
        ["src/birdvoice/train.py", "Training loop, augmentation, evaluation, checkpointing, and metric storage."],
        ["src/birdvoice/infer.py", "Top-k prediction for one recorded audio file."],
        ["scripts/prepare_birdclef.py", "BirdCLEF metadata manifest and label-map construction."],
        ["scripts/eda.py", "Class balance and metadata-quality summary."],
        ["scripts/create_splits.py", "Stratified and source-held-out split creation."],
        ["scripts/build_pitch_priors.py", "Robust per-species pitch-prior estimation from training audio."],
        ["configs/birdclef2021_baseline.yaml", "Versioned hyperparameters and input paths."],
    ], [2.65, 3.9])

    add_heading(doc, "Limitations and responsible claims")
    doc.add_paragraph(
        "A species classifier does not estimate bird population size. Population or occurrence claims require repeated surveys, fixed recorder locations, "
        "sampling effort, local validation, and detection-aware ecological modelling. The proposed benchmark does not supply validated sex labels for "
        "all species, so the system must not claim gender classification. Likewise, call-type classification requires a separately curated annotation set. "
        "Locations for sensitive species should be generalized in any public report or application."
    )

    doc.add_page_break()
    add_heading(doc, "Reproducible execution sequence")
    for command in [
        "python scripts/prepare_birdclef.py --input data/raw/birdclef-2021 --output data/processed/birdclef-2021",
        "python scripts/eda.py --metadata data/processed/birdclef-2021/metadata.csv --output artifacts/eda",
        "python scripts/create_splits.py --metadata data/processed/birdclef-2021/metadata.csv --output artifacts/splits --strategy stratified",
        "birdvoice-train --config configs/birdclef2021_baseline.yaml",
        "python scripts/build_pitch_priors.py --metadata artifacts/splits/train.csv --audio-root data/raw/birdclef-2021/train_short_audio --output artifacts/pitch_priors.csv",
    ]:
        paragraph = doc.add_paragraph()
        run = paragraph.add_run(command)
        run.font.name = "Courier New"; run.font.size = Pt(9)
        paragraph.paragraph_format.left_indent = Inches(0.2)

    add_heading(doc, "References")
    references = [
        "BirdCLEF 2021 Birdcall Identification. Kaggle competition data and rules. https://www.kaggle.com/competitions/birdclef-2021",
        "Conde, M. V., et al. Weakly-Supervised Classification and Detection of Bird Sounds in the Wild. CLEF 2021 Working Notes.",
        "Project proposal deck supplied by the project owner, GC002.pptx.",
    ]
    for reference in references:
        doc.add_paragraph(reference, style="List Number")

    for section in doc.sections:
        footer = section.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        footer.add_run("Bird Voice Recognition System | Technical Project Report")
        footer.runs[0].font.size = Pt(8)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
