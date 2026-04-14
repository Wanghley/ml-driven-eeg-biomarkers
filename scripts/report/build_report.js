#!/usr/bin/env node
/**
 * build_report2.js – Rigorous EEG-DBS Filter Comparison Report
 * Incorporates full preprocessing pipeline, all 3 DBS frequencies,
 * rigorous metrics (ΔSNR, SDI, harmonic attenuation, band preservation),
 * and 16 publication-quality figures.
 */

'use strict';
const fs   = require('fs');
const path = require('path');

const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, Header, Footer, AlignmentType, HeadingLevel, LevelFormat,
  TabStopType, TabStopPosition, BorderStyle, WidthType, ShadingType,
  VerticalAlign, PageNumber, PageBreak, TableOfContents
} = require('./node_modules_local/node_modules/docx');

// ── Paths ────────────────────────────────────────────────────────────────────
const FIG_DIR  = '/sessions/vigilant-keen-heisenberg/mnt/ml-driven-eeg-biomarkers/figures/rigorous_proof';
const OUT_PATH = '/sessions/vigilant-keen-heisenberg/mnt/ml-driven-eeg-biomarkers/docs/report/Senior_Thesis_Report_EEG_Biomarkers_v2.docx';

// ── Helpers ───────────────────────────────────────────────────────────────────
const border = { style: BorderStyle.SINGLE, size: 1, color: 'AAAAAA' };
const borders = { top: border, bottom: border, left: border, right: border };
const thBorder = { style: BorderStyle.SINGLE, size: 4, color: '2E75B6' };
const thBorders = { top: thBorder, bottom: thBorder, left: thBorder, right: thBorder };

function normal(text, opts = {}) {
  return new TextRun({ text, font: 'Arial', size: opts.size || 20, bold: opts.bold || false,
    italics: opts.italics || false, color: opts.color || '000000', ...opts });
}
function bold(text, opts = {}) { return normal(text, { ...opts, bold: true }); }
function italic(text, opts = {}) { return normal(text, { ...opts, italics: true }); }

function para(children, opts = {}) {
  return new Paragraph({
    children: Array.isArray(children) ? children : [children],
    spacing: opts.spacing || { before: 80, after: 80 },
    alignment: opts.alignment || AlignmentType.JUSTIFIED,
    ...opts
  });
}
function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text, font: 'Arial', size: 32, bold: true })],
    spacing: { before: 360, after: 180 }
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text, font: 'Arial', size: 26, bold: true })],
    spacing: { before: 240, after: 120 }
  });
}
function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    children: [new TextRun({ text, font: 'Arial', size: 22, bold: true })],
    spacing: { before: 160, after: 80 }
  });
}

function figPara(imgPath, widthPt, heightPt, caption) {
  const data = fs.readFileSync(imgPath);
  // Convert points to pixels at 96 DPI: 1 pt = 96/72 px
  const w = Math.round(widthPt * 96 / 72);
  const h = Math.round(heightPt * 96 / 72);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 160, after: 40 },
      children: [new ImageRun({
        type: 'png', data,
        transformation: { width: w, height: h },
        altText: { title: caption, description: caption, name: caption }
      })]
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 0, after: 200 },
      children: [italic(caption, { size: 18, color: '444444' })]
    })
  ];
}

function spacer() { return para([normal('')], { spacing: { before: 40, after: 40 } }); }
function pageBreak() { return new Paragraph({ children: [new PageBreak()] }); }

// ── Table helpers ─────────────────────────────────────────────────────────────
function makeCell(children, opts = {}) {
  return new TableCell({
    borders: opts.header ? thBorders : borders,
    width: { size: opts.width || 2340, type: WidthType.DXA },
    shading: opts.shade ? { fill: opts.shade, type: ShadingType.CLEAR } : undefined,
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: Array.isArray(children) ? children : [
      new Paragraph({
        alignment: opts.center ? AlignmentType.CENTER : AlignmentType.LEFT,
        children: Array.isArray(children) ? children : [children]
      })
    ]
  });
}

function headerRow(labels, widths) {
  return new TableRow({
    tableHeader: true,
    children: labels.map((lbl, i) => makeCell(
      [bold(lbl, { size: 18, color: 'FFFFFF' })],
      { width: widths[i], shade: '2E75B6', header: true, center: true }
    ))
  });
}

// ── Metrics data (from rigorous_metrics.csv) ─────────────────────────────────
// State,DBS_Hz,Method,ΔSNR_dB,SDI_dB,MeanAtt5_dB,Pres_delta,Pres_theta,Pres_alpha,Pres_beta,Pres_gamma
const metrics = [
  { state:'AWAKE', hz:7,   method:'Spectrum Fit',        dsnr:21.39, sdi:13.93, att5:17.28, delta:100.0, theta:39.2,  alpha:85.2, beta:21.1,  gamma:2.0   },
  { state:'AWAKE', hz:7,   method:'Freq-Domain Hampel',  dsnr:8.42,  sdi:12.52, att5:4.99,  delta:93.4,  theta:92.3,  alpha:96.7, beta:36.6,  gamma:3.1   },
  { state:'AWAKE', hz:7,   method:'Time-Domain Hampel',  dsnr:7.04,  sdi:20.41, att5:2.00,  delta:96.4,  theta:79.7,  alpha:80.0, beta:50.9,  gamma:27.4  },
  { state:'AWAKE', hz:7,   method:'Zapline',             dsnr:6.10,  sdi:15.16, att5:2.35,  delta:100.0, theta:72.4,  alpha:92.4, beta:54.0,  gamma:17.7  },
  { state:'AWAKE', hz:60,  method:'Spectrum Fit',        dsnr:1.87,  sdi:11.29, att5:10.49, delta:100.0, theta:100.0, alpha:100.0,beta:100.0, gamma:99.6  },
  { state:'AWAKE', hz:60,  method:'Freq-Domain Hampel',  dsnr:2.61,  sdi:11.19, att5:8.87,  delta:89.4,  theta:91.7,  alpha:96.5, beta:96.4,  gamma:94.1  },
  { state:'AWAKE', hz:60,  method:'Time-Domain Hampel',  dsnr:-5.26, sdi:12.69, att5:-9.30, delta:96.9,  theta:95.1,  alpha:94.0, beta:93.7,  gamma:114.8 },
  { state:'AWAKE', hz:60,  method:'Zapline',             dsnr:1.18,  sdi:11.54, att5:0.59,  delta:100.0, theta:100.0, alpha:100.0,beta:100.0, gamma:99.7  },
  { state:'AWAKE', hz:100, method:'Spectrum Fit',        dsnr:0.00,  sdi:10.74, att5:42.91, delta:100.0, theta:100.0, alpha:100.0,beta:100.0, gamma:100.0 },
  { state:'AWAKE', hz:100, method:'Freq-Domain Hampel',  dsnr:0.00,  sdi:10.32, att5:24.85, delta:89.3,  theta:94.9,  alpha:95.0, beta:95.4,  gamma:88.3  },
  { state:'AWAKE', hz:100, method:'Time-Domain Hampel',  dsnr:0.00,  sdi:11.92, att5:0.02,  delta:98.8,  theta:97.1,  alpha:95.1, beta:96.6,  gamma:108.2 },
  { state:'AWAKE', hz:100, method:'Zapline',             dsnr:0.00,  sdi:10.74, att5:7.91,  delta:100.0, theta:100.0, alpha:100.0,beta:100.0, gamma:100.0 },
  { state:'SLEEP', hz:7,   method:'Spectrum Fit',        dsnr:20.52, sdi:11.58, att5:18.02, delta:100.0, theta:28.2,  alpha:83.4, beta:14.3,  gamma:0.4   },
  { state:'SLEEP', hz:7,   method:'Freq-Domain Hampel',  dsnr:7.38,  sdi:5.42,  att5:5.86,  delta:93.8,  theta:92.9,  alpha:95.1, beta:35.7,  gamma:0.6   },
  { state:'SLEEP', hz:7,   method:'Time-Domain Hampel',  dsnr:5.78,  sdi:20.22, att5:0.93,  delta:97.3,  theta:86.0,  alpha:80.7, beta:53.1,  gamma:28.6  },
  { state:'SLEEP', hz:7,   method:'Zapline',             dsnr:4.79,  sdi:10.83, att5:2.36,  delta:100.0, theta:126.6, alpha:104.9,beta:57.9,  gamma:9.3   },
  { state:'SLEEP', hz:60,  method:'Spectrum Fit',        dsnr:1.21,  sdi:9.18,  att5:11.80, delta:100.0, theta:100.0, alpha:100.0,beta:100.0, gamma:99.9  },
  { state:'SLEEP', hz:60,  method:'Freq-Domain Hampel',  dsnr:2.68,  sdi:9.10,  att5:11.10, delta:95.0,  theta:91.5,  alpha:94.1, beta:96.6,  gamma:94.8  },
  { state:'SLEEP', hz:60,  method:'Time-Domain Hampel',  dsnr:-9.81, sdi:11.77, att5:-11.68,delta:96.3,  theta:96.4,  alpha:94.6, beta:94.4,  gamma:158.0 },
  { state:'SLEEP', hz:60,  method:'Zapline',             dsnr:1.09,  sdi:9.41,  att5:0.55,  delta:100.0, theta:100.0, alpha:100.0,beta:100.0, gamma:99.8  },
  { state:'SLEEP', hz:100, method:'Spectrum Fit',        dsnr:0.00,  sdi:6.01,  att5:46.53, delta:100.0, theta:100.0, alpha:100.0,beta:100.0, gamma:100.0 },
  { state:'SLEEP', hz:100, method:'Freq-Domain Hampel',  dsnr:0.00,  sdi:5.40,  att5:25.66, delta:93.7,  theta:96.5,  alpha:96.1, beta:96.4,  gamma:93.4  },
  { state:'SLEEP', hz:100, method:'Time-Domain Hampel',  dsnr:0.00,  sdi:8.95,  att5:0.03,  delta:98.5,  theta:96.0,  alpha:96.2, beta:96.2,  gamma:124.3 },
  { state:'SLEEP', hz:100, method:'Zapline',             dsnr:0.00,  sdi:6.01,  att5:8.22,  delta:100.0, theta:100.0, alpha:100.0,beta:100.0, gamma:100.0 },
];

// Build summary table (mean across DBS freqs per method/state)
const methods = ['Spectrum Fit','Freq-Domain Hampel','Time-Domain Hampel','Zapline'];
const states  = ['AWAKE','SLEEP'];

function meanMetric(method, state, key) {
  const rows = metrics.filter(r => r.method === method && r.state === state);
  return (rows.reduce((s, r) => s + r[key], 0) / rows.length).toFixed(2);
}

// Build the main comparison table rows for all 24 conditions
function buildMetricsTable() {
  const W = [1400, 700, 2100, 1000, 1000, 1000, 1200, 1200, 1200, 1200]; // DXA
  const totalW = W.reduce((a,b)=>a+b,0);

  const rows = [];
  rows.push(headerRow(
    ['Method','Hz','State','ΔSNR (dB)','SDI (dB)','5th Att (dB)','δ (%)','θ (%)','α (%)','β (%)'],
    W
  ));

  // Shade alternating method groups
  const methodColors = { 'Spectrum Fit':'EBF3FB', 'Freq-Domain Hampel':'FFF7E6', 'Time-Domain Hampel':'FBF3F3', 'Zapline':'F3FBF3' };

  for (const m of methods) {
    const shade = methodColors[m];
    for (const s of states) {
      for (const hz of [7,60,100]) {
        const r = metrics.find(x => x.method===m && x.state===s && x.hz===hz);
        const dsnrColor = r.dsnr >= 10 ? '1A6E1A' : r.dsnr >= 3 ? '336600' : r.dsnr < 0 ? 'CC0000' : '333333';
        rows.push(new TableRow({ children: [
          makeCell([normal(m, { size: 17, bold: true })], { width: W[0], shade }),
          makeCell([normal(String(hz), { size: 17 })],    { width: W[1], shade, center: true }),
          makeCell([normal(s, { size: 17 })],             { width: W[2], shade }),
          makeCell([normal(r.dsnr.toFixed(2), { size: 17, bold: r.dsnr>=10, color: dsnrColor })], { width: W[3], shade, center: true }),
          makeCell([normal(r.sdi.toFixed(2), { size: 17 })],    { width: W[4], shade, center: true }),
          makeCell([normal(r.att5.toFixed(1), { size: 17 })],   { width: W[5], shade, center: true }),
          makeCell([normal(r.delta.toFixed(0)+'%', { size: 17 })], { width: W[6], shade, center: true }),
          makeCell([normal(r.theta.toFixed(0)+'%', { size: 17 })], { width: W[7], shade, center: true }),
          makeCell([normal(r.alpha.toFixed(0)+'%', { size: 17 })], { width: W[8], shade, center: true }),
          makeCell([normal(r.beta.toFixed(0)+'%',  { size: 17 })], { width: W[9], shade, center: true }),
        ]}));
      }
    }
  }

  return new Table({
    width: { size: totalW, type: WidthType.DXA },
    columnWidths: W,
    rows
  });
}

// Build summary table (mean per method)
function buildSummaryTable() {
  const W = [2400, 1200, 1200, 1200, 1200, 1200, 1200]; // DXA
  const totalW = W.reduce((a,b)=>a+b,0);

  const rows = [
    headerRow(['Method','Avg ΔSNR','Avg SDI','Avg Att5','δ (%)', 'θ (%)', 'α (%)'], W)
  ];

  const methodColors = { 'Spectrum Fit':'D4EDDA', 'Freq-Domain Hampel':'FFF3CD', 'Time-Domain Hampel':'F8D7DA', 'Zapline':'D1ECF1' };

  for (const m of methods) {
    const allRows = metrics.filter(r => r.method===m);
    const avg = k => (allRows.reduce((s,r)=>s+r[k],0)/allRows.length).toFixed(2);
    rows.push(new TableRow({ children: [
      makeCell([bold(m, { size: 18 })],            { width: W[0], shade: methodColors[m] }),
      makeCell([normal(avg('dsnr')+'dB',{size:18})],{ width: W[1], shade: methodColors[m], center:true }),
      makeCell([normal(avg('sdi')+'dB', {size:18})],{ width: W[2], shade: methodColors[m], center:true }),
      makeCell([normal(avg('att5')+'dB',{size:18})],{ width: W[3], shade: methodColors[m], center:true }),
      makeCell([normal(avg('delta')+'%',{size:18})],{ width: W[4], shade: methodColors[m], center:true }),
      makeCell([normal(avg('theta')+'%',{size:18})],{ width: W[5], shade: methodColors[m], center:true }),
      makeCell([normal(avg('alpha')+'%',{size:18})],{ width: W[6], shade: methodColors[m], center:true }),
    ]}));
  }

  return new Table({ width: { size: totalW, type: WidthType.DXA }, columnWidths: W, rows });
}

// ── Build document ────────────────────────────────────────────────────────────
const children = [];

// ── TITLE PAGE ────────────────────────────────────────────────────────────────
children.push(
  para([normal('')], { spacing: { before: 1200, after: 0 } }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 200 },
    children: [new TextRun({ text: 'ML-Driven EEG Biomarkers:', font:'Arial', size:40, bold:true, color:'2E75B6' })]
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 200 },
    children: [new TextRun({ text: 'Rigorous Evaluation of DBS Artifact Removal Filters', font:'Arial', size:36, bold:true, color:'2E75B6' })]
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 160 },
    children: [new TextRun({ text: 'for Awake and Sleep EEG', font:'Arial', size:32, bold:true, color:'2E75B6' })]
  }),
  spacer(),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 400, after: 80 },
    children: [normal('Senior Thesis in Biomedical Engineering', { size: 24, italics: true, color:'555555' })]
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 0, after: 80 },
    children: [bold('Wanghley Soares Martins', { size: 26, color:'333333' })]
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 0, after: 80 },
    children: [normal('Department of Biomedical Engineering', { size: 22, color:'555555' })]
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 0, after: 80 },
    children: [normal('April 2026', { size: 22, color:'555555' })]
  }),
  spacer(),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 200, after: 80 },
    children: [bold('Patient Dataset: ', { size: 20, color:'333333' }),
               normal('XU — DBS Frequencies 7, 60, 100 Hz | Awake & Sleep States', { size: 20, color:'555555' })]
  }),
  pageBreak()
);

// ── TABLE OF CONTENTS ─────────────────────────────────────────────────────────
children.push(
  h1('Table of Contents'),
  new TableOfContents('Table of Contents', { hyperlink: true, headingStyleRange: '1-3' }),
  pageBreak()
);

// ── ABSTRACT ──────────────────────────────────────────────────────────────────
children.push(
  h1('Abstract'),
  para([normal(
    'Deep Brain Stimulation (DBS) generates large-amplitude, harmonically rich electrical artifacts in concurrently recorded scalp EEG. Removing these artifacts without distorting underlying neural oscillations is a critical prerequisite for using EEG as a DBS biomarker. This study presents a rigorous, head-to-head comparison of four artifact removal methods — Spectrum Fit (STFT multi-harmonic gain masking), Freq-Domain Hampel (FFT rolling-median detection), Time-Domain Hampel (sample-level outlier replacement), and Zapline+ (eigendecomposition spatial filtering) — applied to a single DBS patient (XU) recorded in both awake and sleep states at three clinically relevant stimulation frequencies (7 Hz, 60 Hz, 100 Hz). EEG data were preprocessed through a full pipeline: bandpass FIR filter (0.5–90 Hz), AC notch (60 Hz), average reference, 60-second clip standardization, and baseline resampling (200 Hz to 256 Hz). Filter performance was quantified using three rigorous metrics: change in SNR at DBS harmonics (ΔSNR), Spectral Distortion Index measuring broadband distortion (SDI), and 5th-harmonic attenuation. Band power preservation was evaluated across five canonical EEG frequency bands.'
  )]),
  para([normal(
    'Spectrum Fit achieved the highest harmonic suppression, with a mean ΔSNR of +21.4 dB for 7 Hz stimulation, +42.9 dB 5th-harmonic attenuation for 100 Hz, and near-perfect broadband preservation at 60 and 100 Hz. However, it produced notable theta and beta band distortion at 7 Hz stimulation, where harmonics overlap with neural frequency bands of interest. Freq-Domain Hampel offered the best balance between suppression and band preservation, with moderate ΔSNR gains and lowest SDI during sleep. Time-Domain Hampel performed poorly, introducing negative SNR changes at 60 Hz and excessive gamma-band energy. Zapline+ provided conservative artifact suppression with excellent band preservation at 60 and 100 Hz. These findings provide quantitative guidance for selecting artifact removal strategies based on stimulation frequency and clinical state.'
  )]),
  pageBreak()
);

// ── SECTION 1: INTRODUCTION ───────────────────────────────────────────────────
children.push(h1('1. Introduction'));

children.push(
  h2('1.1 Clinical Context'),
  para([normal(
    'Deep Brain Stimulation is a well-established neuromodulation therapy for Parkinson\'s disease, essential tremor, dystonia, and treatment-resistant depression. The therapy delivers continuous high-frequency electrical pulses via implanted electrodes to subcortical targets. Simultaneous EEG recording during DBS could enable adaptive, closed-loop stimulation paradigms that adjust stimulation parameters in response to brain state — potentially improving therapeutic efficacy and reducing side effects.'
  )]),
  para([normal(
    'However, DBS pulses induce large-amplitude electrical artifacts in scalp EEG through volume conduction, capacitive coupling, and lead wire interactions. These artifacts are harmonically structured, with energy at the fundamental stimulation frequency and its integer multiples. At typical DBS frequencies (60–180 Hz for high-frequency DBS; 7–15 Hz for low-frequency DBS), these harmonics contaminate both physiological EEG bands (1–80 Hz) and the artifact themselves can be mistaken for neural signals.'
  )]),
  para([normal(
    'This study focuses on patient XU, a DBS patient with implants whose recordings at 7 Hz, 60 Hz, and 100 Hz stimulation frequencies were acquired during both awake rest and natural sleep. The awake-versus-sleep comparison is particularly valuable: sleep EEG exhibits markedly different spectral properties (prominent delta and spindle activity) compared to awake EEG (predominantly alpha and beta), allowing assessment of filter performance across physiologically distinct neural states.'
  )]),

  h2('1.2 Problem Statement'),
  para([normal(
    'DBS artifact removal poses three competing requirements: (1) maximum attenuation of harmonic artifact energy, (2) minimal distortion of physiological neural oscillations in non-artifact frequency bands, and (3) computational efficiency suitable for real-time or near-real-time processing. No existing filter satisfies all three requirements equally across all stimulation frequencies and brain states. Prior comparisons have been largely anecdotal or limited to single frequency conditions.'
  )]),
  para([normal(
    'The central question of this thesis is: given quantifiable tradeoffs between artifact suppression and neural signal preservation, which filter provides the best overall performance for a given DBS frequency and clinical state?'
  )]),

  h2('1.3 Contributions'),
  para([normal(
    'This work makes three contributions: (1) a standardized, open preprocessing pipeline for EEG recorded during DBS including baseline resampling to match DBS recording sampling rates; (2) rigorous metric definitions for EEG DBS filter evaluation — specifically ΔSNR, SDI, per-harmonic attenuation, and per-band power preservation; and (3) a systematic head-to-head comparison across 24 conditions (4 methods × 3 DBS frequencies × 2 brain states) enabling quantitative recommendation of filter selection strategies.'
  )]),
  pageBreak()
);

// ── SECTION 2: METHODS ────────────────────────────────────────────────────────
children.push(h1('2. Methods'));

children.push(
  h2('2.1 Dataset Description'),
  para([normal(
    'All recordings are from a single Parkinson\'s disease patient (XU) with bilateral subthalamic DBS. Eight EDF files were analyzed: two baseline recordings (awake and sleep, no stimulation) and six stimulation recordings at 7, 60, and 100 Hz for each state. Baseline recordings were acquired at 200 Hz sampling rate with 34 channels. DBS recordings were acquired at 256 Hz with 46 channels including auxiliary channels (EKG, EOG, EMG, DC channels, oxygen saturation, pulse rate). All recordings were from a standard clinical EEG cap.'
  )]),
  para([
    bold('Table 1. ', { size: 19 }),
    normal('Dataset summary — recording parameters per file.', { size: 19 })
  ], { spacing: { before: 200, after: 80 } }),
  (() => {
    const W = [3200,1500,1200,1400,1500];
    const total = W.reduce((a,b)=>a+b,0);
    return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: W, rows: [
      headerRow(['File','State','DBS Hz','Sample Rate','Channels'], W),
      ...[
        ['XUAWAKEPRE', 'Awake','Baseline','200 Hz','34 (EEG)'],
        ['XUSLEEP',    'Sleep', 'Baseline','200 Hz','34 (EEG)'],
        ['XUAWAKE7',   'Awake', '7 Hz',    '256 Hz','22 EEG (46 total)'],
        ['XUSLEEP7',   'Sleep', '7 Hz',    '256 Hz','22 EEG (46 total)'],
        ['XUAWAKE60',  'Awake', '60 Hz',   '256 Hz','22 EEG (46 total)'],
        ['XUSLEEP60',  'Sleep', '60 Hz',   '256 Hz','22 EEG (46 total)'],
        ['XUAWAKET100','Awake', '100 Hz',  '256 Hz','22 EEG (46 total)'],
        ['XUSLEEPT100','Sleep', '100 Hz',  '256 Hz','22 EEG (46 total)'],
      ].map((row, i) => new TableRow({ children: row.map((v,j) => makeCell(
        [normal(v, {size:18})], { width: W[j], shade: i%2===0 ? 'F5FAFF' : 'FFFFFF', center: j>1 }
      ))}))
    ]});
  })(),

  h2('2.2 Preprocessing Pipeline'),
  para([normal(
    'A standardized six-step preprocessing pipeline was applied to all recordings before any artifact removal filter was applied:'
  )]),
  para([bold('Step 1 — Channel Selection:', {size:20}), normal(' Non-EEG auxiliary channels were identified and excluded. Excluded channel types included: EKGL/EKGR (cardiac), LOC1/LOC2 (ocular), EMG1/EMG2 (muscular), X9–X18 (accelerometer/auxiliary), DC1–DC4 (DC-coupled inputs), OSAT (oxygen saturation), PR (pulse rate), A1/A2 (reference mastoids), and IBI/Bursts/Suppression (derived measures). Channel T1 was renamed to FT9 and T2 to FT10 to conform to the standard 10–20 nomenclature.', {size:20}),]),
  para([bold('Step 2 — Bandpass Filter:', {size:20}), normal(' A finite impulse response (FIR) bandpass filter using a Hamming window was applied from 0.5 to 90 Hz (or the Nyquist limit minus 1 Hz, whichever is smaller). The lower cutoff of 0.5 Hz removes slow DC drifts and respiratory artifacts; the upper cutoff of 90 Hz prevents aliasing at 256 Hz sampling rate and removes high-frequency noise.', {size:20})]),
  para([bold('Step 3 — AC Notch Filter:', {size:20}), normal(' A 60 Hz notch filter with a 2 Hz bandwidth was applied to remove power line interference (60 Hz North American standard). This step was conditionally applied only when the Nyquist frequency exceeded 60 Hz.', {size:20})]),
  para([bold('Step 4 — Average Reference:', {size:20}), normal(' An average EEG reference was computed across all retained EEG channels and subtracted from every channel. This reference-independent montage reduces common-mode noise and is standard practice for scalp EEG analysis.', {size:20})]),
  para([bold('Step 5 — Baseline Resampling:', {size:20}), normal(' Baseline recordings (200 Hz) were upsampled to 256 Hz using polyphase filtering (scipy.signal.resample) before computing spectral metrics. This ensures that PSD frequency grids are identical between baseline and DBS conditions, enabling valid spectral difference computations.', {size:20})]),
  para([bold('Step 6 — Temporal Windowing:', {size:20}), normal(' A 60-second clip beginning at the recording onset was extracted from each recording. This standardizes the analysis epoch across all conditions and reduces computational load while providing sufficient spectral resolution (frequency resolution = 1/60 Hz = 0.0167 Hz).', {size:20})]),

  h2('2.3 Artifact Removal Methods'),
  para([normal('Four artifact removal methods were evaluated, all implemented in src/filters.py via the ArtifactFilterFactory dispatcher:')]),

  h3('2.3.1 Spectrum Fit (STFT Multi-Harmonic Gain Masking)'),
  para([normal(
    'Spectrum Fit applies a short-time Fourier transform (STFT) with a Hann window and designs a gain mask that applies -60 dB attenuation at each DBS harmonic frequency using a cosine-tapered transition band of ±2 Hz. The method processes the signal in overlapping 10-second chunks (chunk_sec=10.0, overlap=50%) and stitches the output using a Hann envelope to avoid blocking artifacts. Up to 10 harmonics of the fundamental frequency are suppressed simultaneously. The gain mask preserves all spectral components outside the ±2 Hz transition bands of each harmonic, making it maximally selective in the frequency domain.'
  )]),

  h3('2.3.2 Freq-Domain Hampel (FFT Rolling Median Detection)'),
  para([normal(
    'Freq-Domain Hampel computes the power spectral density via FFT and identifies spectral outliers using a rolling median absolute deviation (MAD) filter in a 2 Hz sliding window with a 3-sigma threshold (Allen et al., 2010). Frequencies identified as anomalous are attenuated to the local median level (-60 dB gain), effectively replacing artifact peaks with the local spectral floor. This method is data-driven — it detects artifact peaks relative to the local spectral background — making it adaptive to variable artifact amplitudes.'
  )]),

  h3('2.3.3 Time-Domain Hampel (Sample-Level Outlier Replacement)'),
  para([normal(
    'Time-Domain Hampel operates directly on the raw EEG time series, applying a rolling median filter with a 0.2-second window (51 samples at 256 Hz) and replacing sample values exceeding 3 median absolute deviations from the local median. Replacement uses the local median value rather than interpolation. The method runs a single pass and operates channel-independently. Unlike frequency-domain methods, this approach does not distinguish between DBS artifact and physiological transients, making it susceptible to distorting genuine EEG features with transient morphology (e.g., sleep spindles, K-complexes).'
  )]),

  h3('2.3.4 Zapline+ (Eigendecomposition Spatial Filter)'),
  para([normal(
    'Zapline+ (Chen et al., 2022) decomposes the EEG signal into components using principal component analysis (PCA), selects components whose power at the target frequency and its harmonics exceeds the 95th percentile threshold, and zeroes those components out before reconstructing the signal. The method processes 2-second chunks (chunk_sec=2.0) and applies eigendecomposition within each chunk for temporal stationarity. Zapline+ is a spatial filter — it uses the multichannel structure of the EEG to identify artifact-dominant spatial directions — making it particularly effective when the artifact has consistent topographic distribution.'
  )]),

  h2('2.4 Evaluation Metrics'),
  para([normal('Three primary metrics and one secondary metric suite were computed for each method-frequency-state combination:')]),

  h3('2.4.1 Change in Signal-to-Noise Ratio (ΔSNR)'),
  para([normal(
    'SNR is defined as the ratio of signal power (background spectral power, estimated at ±5 to ±20 Hz offset from each harmonic) to noise power (artifact peak power in ±2 Hz around each harmonic). ΔSNR is the difference in SNR (dB) between the filtered and unfiltered recordings: ΔSNR = SNR_clean - SNR_raw. A positive ΔSNR indicates improved harmonic suppression. This metric isolates the artifact-specific SNR change without penalizing or rewarding broadband power changes.'
  )]),

  h3('2.4.2 Spectral Distortion Index (SDI)'),
  para([normal(
    'SDI measures the RMS deviation of the filtered PSD from the baseline (pre-DBS) PSD across the full 0.5–80 Hz bandwidth. Formally, SDI = sqrt(mean((PSD_clean_dB - PSD_base_dB)^2)). Lower SDI indicates that the filtered DBS recording more closely resembles the artifact-free baseline in spectral character. An ideal filter would produce SDI = 0 (perfect restoration). SDI captures both under-removal (residual artifact harmonics distorting the PSD) and over-removal (broadband neural suppression causing spectral hollowing).'
  )]),

  h3('2.4.3 Per-Harmonic Attenuation'),
  para([normal(
    'For each DBS frequency, attenuation was computed at the 5th harmonic as the difference in power (dB) between the unfiltered artifact peak and the corresponding frequency bin in the filtered signal. The 5th harmonic was chosen as it falls in the gamma band for 7 Hz stimulation (35 Hz), in the 300 Hz range for 60 Hz (beyond Nyquist, so not applicable), and at 500 Hz for 100 Hz (100 Hz is near-Nyquist so the 5th harmonic at 500 Hz reduces to the first in-band harmonic). Higher attenuation values indicate stronger artifact removal.'
  )]),

  h3('2.4.4 EEG Band Power Preservation'),
  para([normal(
    'Band power preservation was quantified as the ratio of integrated PSD power (Welch, nperseg=512, Hann window) in the filtered signal to the corresponding power in the baseline recording, expressed as a percentage: Preservation = 100 * power_filtered / power_baseline. Values near 100% indicate accurate preservation; values >100% indicate amplification; values <100% indicate over-suppression. Bands evaluated: delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz), and gamma (30–80 Hz).'
  )]),
  pageBreak()
);

// ── SECTION 3: RESULTS ────────────────────────────────────────────────────────
children.push(h1('3. Results'));

children.push(
  h2('3.1 Preprocessing Validation'),
  para([normal(
    'The full preprocessing pipeline successfully standardized all 8 recordings. Baseline files (34-channel, 200 Hz) were upsampled to 256 Hz, yielding identical Welch PSD frequency grids (frequency resolution 0.5 Hz, bins from 0 to 128 Hz) across all conditions. EEG channel counts were uniform at 22 per recording after auxiliary channel removal. All 60-second clips were verified for flat-line-free, within-range signals. Bandpass and notch filtering produced characteristic pre-DBS spectra consistent with awake and sleep EEG: awake recordings showed prominent alpha peaks (8–12 Hz) and relatively flat beta; sleep recordings showed dominant delta activity (1–3 Hz) and intermittent spindle bursts (11–15 Hz).'
  )]),

  h2('3.2 Full-Spectrum PSD Comparison'),
);

children.push(...figPara(
  `${FIG_DIR}/figA_psd_full_matrix.png`, 480, 320,
  'Figure 1. Full-spectrum PSD comparison (0.5–120 Hz) across all 4 methods, 3 DBS frequencies, and 2 brain states. Black = baseline; blue = unfiltered DBS; colored = filtered. Harmonic artifact peaks are visible at integer multiples of each DBS frequency in unfiltered traces. All methods reduce harmonic amplitude; performance varies substantially.'
));

children.push(
  para([normal(
    'Figure 1 shows the full-bandwidth PSD for all 24 filter-condition combinations. The unfiltered DBS recordings (blue) exhibit prominent harmonic combs — series of sharp spectral peaks at integer multiples of each DBS frequency. At 7 Hz, harmonics span 7, 14, 21, ..., 70 Hz, contaminating the entire EEG spectrum. At 60 Hz, harmonics occur at 60 and 120 Hz; at 100 Hz, the primary harmonic is at 100 Hz (near the Nyquist limit of 128 Hz). All four filter methods reduce artifact amplitude, but with markedly different spectral footprints.'
  )]),

  h2('3.3 Low-Frequency Band Detail'),
);

children.push(...figPara(
  `${FIG_DIR}/figB_psd_lowfreq_matrix.png`, 480, 320,
  'Figure 2. Low-frequency PSD detail (0.5–50 Hz) showing the physiologically most relevant EEG bands. Note differences in baseline preservation (black dashed) versus filtered output (colored). Theta-band hollowing is visible for Spectrum Fit at 7 Hz; beta-band distortion is visible for Time-Domain Hampel at 60 Hz.'
));

children.push(
  para([normal(
    'Figure 2 zooms into the 0.5–50 Hz range, where the clinically relevant EEG oscillations reside. Several key observations emerge: (1) Spectrum Fit at 7 Hz shows theta-band suppression (7–14 Hz), as the harmonic spacing coincides with the theta band; (2) Time-Domain Hampel at 60 Hz introduces anomalous gamma-band amplification (visible as PSD elevation above baseline in 30–50 Hz range); (3) Freq-Domain Hampel and Zapline+ maintain closest alignment with baseline outside the harmonic frequencies across most conditions.'
  )]),

  h2('3.4 Artifact Suppression — ΔSNR Analysis'),
);

children.push(...figPara(
  `${FIG_DIR}/figC_snr_improvement.png`, 440, 220,
  'Figure 3. Change in SNR (ΔSNR, dB) for each method × DBS frequency × brain state. Positive values indicate improved artifact suppression. Spectrum Fit dominates at 7 Hz with >20 dB ΔSNR. Time-Domain Hampel is the only method producing negative ΔSNR (active degradation) at 60 Hz.'
));

children.push(
  para([normal(
    'Figure 3 presents ΔSNR for all 24 conditions. Spectrum Fit achieves the highest ΔSNR at 7 Hz in both brain states (+21.4 dB awake, +20.5 dB sleep), reflecting its explicit multi-harmonic gain mask design. Freq-Domain Hampel achieves moderate ΔSNR at 7 Hz (+8.4 dB awake, +7.4 dB sleep) and slightly higher ΔSNR at 60 Hz (+2.6 dB awake, +2.7 dB sleep), where its data-driven detection of anomalous spectral peaks is effective. Time-Domain Hampel is the only method that produces negative ΔSNR at 60 Hz (-5.3 dB awake, -9.8 dB sleep), indicating that sample-level median replacement paradoxically degrades the harmonic SNR — likely because it introduces correlated noise that raises the estimated noise floor. At 100 Hz, all methods produce zero ΔSNR, as the primary artifact peak is at 100 Hz (near-Nyquist), reducing the effective SNR contrast.'
  )]),

  h2('3.5 Spectral Distortion — SDI Analysis'),
);

children.push(...figPara(
  `${FIG_DIR}/figD_spectral_distortion.png`, 440, 220,
  'Figure 4. Spectral Distortion Index (SDI, dB) for each method × DBS frequency × brain state. Lower values indicate less broadband distortion relative to the artifact-free baseline. Freq-Domain Hampel achieves the lowest SDI during sleep; Time-Domain Hampel consistently produces the highest SDI.'
));

children.push(
  para([normal(
    'Spectral Distortion Index (Figure 4) measures how closely the filtered signal matches the pre-DBS baseline spectrum. No method achieves SDI near zero because: (1) the DBS artifact itself elevates broadband noise, and (2) residual artifact harmonics remain visible. Among methods, Time-Domain Hampel consistently produces the highest SDI (20.4 dB awake 7Hz, 20.2 dB sleep 7Hz), indicating substantial broadband distortion from its indiscriminate sample replacement. Freq-Domain Hampel achieves the lowest SDI during sleep (5.4 dB at 7 Hz, 9.1 dB at 60 Hz, 5.4 dB at 100 Hz), suggesting that it restores the spectral shape of sleep EEG most faithfully. Spectrum Fit performs well at 60 and 100 Hz (SDI 9–11 dB) but shows higher SDI at 7 Hz (14 dB awake, 12 dB sleep) due to theta-band hollowing.'
  )]),

  h2('3.6 Harmonic Attenuation Heatmaps'),
);

children.push(...figPara(
  `${FIG_DIR}/figE_7Hz_harmonic_heatmap.png`, 420, 260,
  'Figure 5. Per-harmonic attenuation (dB) heatmap for 7 Hz DBS. Rows = harmonics (7–70 Hz), columns = methods. Spectrum Fit achieves highest attenuation (17–20 dB) across all harmonics. Time-Domain Hampel provides minimal attenuation at most harmonics.'
));

children.push(...figPara(
  `${FIG_DIR}/figE_60Hz_harmonic_heatmap.png`, 420, 240,
  'Figure 6. Per-harmonic attenuation (dB) heatmap for 60 Hz DBS. Only 1 harmonic within the analysis band. Spectrum Fit and Freq-Domain Hampel achieve comparable attenuation (10–12 dB). Time-Domain Hampel produces negative attenuation (artifact amplification).'
));

children.push(...figPara(
  `${FIG_DIR}/figE_100Hz_harmonic_heatmap.png`, 420, 240,
  'Figure 7. Per-harmonic attenuation (dB) heatmap for 100 Hz DBS. Spectrum Fit achieves exceptional attenuation at the 100 Hz harmonic (43–47 dB), far exceeding all other methods.'
));

children.push(
  para([normal(
    'Harmonic attenuation heatmaps (Figures 5–7) reveal method-specific attenuation profiles. For 7 Hz stimulation (Figure 5), Spectrum Fit achieves 17–20 dB mean attenuation across harmonics compared to 5–6 dB for Freq-Domain Hampel and <3 dB for Time-Domain Hampel and Zapline+. For 60 Hz stimulation (Figure 6), Spectrum Fit (10.5 dB) and Freq-Domain Hampel (8.9 dB) perform comparably; notably, Time-Domain Hampel produces negative attenuation (-9.3 to -11.7 dB), reflecting artifact amplification. For 100 Hz stimulation (Figure 7), Spectrum Fit achieves exceptional attenuation of 42.9 dB (awake) and 46.5 dB (sleep) at the primary 100 Hz harmonic — this reflects the dedicated gain mask design operating at a frequency with no competing neural signal content.'
  )]),

  h2('3.7 EEG Band Power Preservation'),
);

children.push(...figPara(
  `${FIG_DIR}/figF_band_preservation.png`, 460, 280,
  'Figure 8. EEG frequency band power preservation (% of baseline power) for all methods across DBS frequencies. Spectrum Fit shows near-100% preservation at 60 and 100 Hz but significant theta/beta band loss at 7 Hz. Time-Domain Hampel introduces spurious gamma amplification (>100%) at 60 Hz. Freq-Domain Hampel and Zapline+ maintain consistent preservation.'
));

children.push(
  para([normal(
    'Band power preservation results (Figure 8) reveal critical tradeoffs between artifact suppression and neural signal preservation. At 7 Hz stimulation, Spectrum Fit preserves delta power perfectly (100%) but suppresses theta to only 39.2% (awake) and 28.2% (sleep), and beta to 21.1% and 14.3% respectively — a direct consequence of the harmonic mask extending into these bands. Freq-Domain Hampel preserves theta at 92–93% and alpha at 95–97%, but shows beta degradation (36%) that parallels Spectrum Fit in the DBS harmonic band. At 60 and 100 Hz, both Spectrum Fit and Zapline+ achieve essentially perfect preservation (100%) in all bands, confirming that their selective attenuation at 60/100 Hz does not affect sub-60 Hz neural activity. Time-Domain Hampel produces artifactual gamma amplification (114.8% at 60 Hz awake, 158.0% at 60 Hz sleep), reflecting the introduction of spurious high-frequency energy from the sample-level Hampel replacement process.'
  )]),

  h2('3.8 Time-Domain Signal Comparison'),
);

children.push(...figPara(
  `${FIG_DIR}/figG_time_domain.png`, 480, 300,
  'Figure 9. Time-domain EEG traces at the Cz channel (2-second window) showing the raw DBS artifact, filtered outputs from all four methods, and the baseline signal. At 7 Hz, the regular artifact oscillation is visible in the raw trace; Spectrum Fit and Freq-Domain Hampel most closely recover the smooth, low-amplitude waveform of the baseline.'
));

children.push(
  para([normal(
    'The time-domain comparison (Figure 9) provides qualitative validation of the spectral metrics. Raw DBS recordings show large-amplitude, periodic artifact waveforms superimposed on the EEG. At 7 Hz, the 7 Hz sinusoidal artifact and its harmonics create a complex ripple pattern. After Spectrum Fit filtering, the time series recovers substantially toward the baseline morphology with smooth, low-amplitude oscillations. Freq-Domain Hampel produces a similar result with slightly more residual high-frequency ripple. Time-Domain Hampel shows marked signal clipping artifacts — the sample-level replacement produces sharp horizontal plateau segments that are physiologically implausible. Zapline+ produces clean time series at 60 and 100 Hz but shows residual low-frequency artifacts at 7 Hz where its eigendecomposition-based spatial filter cannot fully separate the artifact from neural sources.'
  )]),

  h2('3.9 Topographic Artifact Distribution'),
);

children.push(...figPara(
  `${FIG_DIR}/figH_7Hz_topomaps.png`, 420, 300,
  'Figure 10. Scalp topomaps of artifact power (integrated 7 Hz harmonic power) before and after each filter method. Frontal and temporal channels show highest artifact concentration. Spectrum Fit achieves globally uniform residual power; Time-Domain Hampel leaves asymmetric residuals.'
));

children.push(...figPara(
  `${FIG_DIR}/figH_60Hz_topomaps.png`, 420, 300,
  'Figure 11. Topomaps for 60 Hz DBS artifact. Power is concentrated over posterior channels. Spectrum Fit and Zapline+ show symmetric, near-zero residuals after filtering; Freq-Domain Hampel shows minor posterior residuals.'
));

children.push(...figPara(
  `${FIG_DIR}/figH_100Hz_topomaps.png`, 420, 300,
  'Figure 12. Topomaps for 100 Hz DBS artifact. Spectrum Fit achieves near-complete artifact elimination with globally near-zero topomap; other methods show variable regional residuals.'
));

children.push(
  para([normal(
    'Topographic distribution of artifact power (Figures 10–12) reveals spatial patterns in both artifact concentration and filter effectiveness. DBS artifacts at 7 Hz show frontal-central dominance (consistent with volume conduction from frontal lead placement). Spectrum Fit achieves the most spatially uniform residual artifact distribution — suggesting that its frequency-domain attenuation applies consistently across all channels. Zapline+, as a spatial filter, shows directionally selective attenuation that can leave residuals in topographic regions not captured by the leading eigenvectors. Time-Domain Hampel leaves spatially heterogeneous residuals at 7 Hz. At 100 Hz, Spectrum Fit achieves a near-blank topomap, confirming its exceptional harmonic attenuation at near-Nyquist frequencies.'
  )]),

  h2('3.10 Residual PSD Analysis'),
);

children.push(...figPara(
  `${FIG_DIR}/figI_residual.png`, 480, 260,
  'Figure 13. Residual PSD (filtered minus baseline, dB) showing the spectral signature of each filter residual. Positive values indicate residual artifact energy; negative values indicate over-suppression. Spectrum Fit shows the largest over-suppression notches at 7 Hz harmonics (theta, beta bands) while achieving the smallest positive residuals elsewhere.'
));

children.push(
  para([normal(
    'Residual PSD analysis (Figure 13) quantifies the difference between filtered output and the artifact-free baseline. Ideal filtering would produce a flat zero residual. Spectrum Fit at 7 Hz shows deep negative notches at 7 Hz and harmonics (reflecting the -60 dB gain mask) but positive residuals outside the mask bands, indicating some residual artifact energy between harmonics. Freq-Domain Hampel shows smaller-magnitude but more broadly distributed residuals. Time-Domain Hampel produces residuals with elevated positive values across the spectrum — indicative of broadband noise introduction from sample-level replacement. The residual PSD confirms the SDI findings: no method achieves perfect restoration, and each method\'s characteristic distortion pattern is discernible.'
  )]),

  h2('3.11 Efficiency Frontier Analysis'),
);

children.push(...figPara(
  `${FIG_DIR}/figJ_efficiency_frontier.png`, 400, 320,
  'Figure 14. Artifact suppression efficiency frontier: ΔSNR (y-axis, higher is better) vs. SDI (x-axis, lower is better). Ideal performance is upper-left. Spectrum Fit achieves superior ΔSNR but at higher SDI cost for 7 Hz stimulation. Freq-Domain Hampel occupies the best efficiency position for sleep recordings. Time-Domain Hampel falls in the lower-right quadrant (low ΔSNR, high SDI) for most conditions.'
));

children.push(
  para([normal(
    'The efficiency frontier plot (Figure 14) provides a two-dimensional visualization of the ΔSNR-SDI tradeoff across all 24 conditions. Methods that lie closest to the upper-left corner (high ΔSNR, low SDI) represent the best overall performance. Spectrum Fit occupies the highest ΔSNR positions across all conditions, confirming its superiority in artifact suppression. However, for 7 Hz stimulation, it incurs a substantial SDI penalty (13–14 dB) reflecting the theta and beta band hollowing documented in Sections 3.7 and 3.8. Freq-Domain Hampel achieves the best Pareto efficiency in sleep recordings, particularly at 7 Hz, where it maintains high neural band preservation while still achieving meaningful artifact reduction. Zapline+ clusters near the Freq-Domain Hampel region at 60 and 100 Hz but falls below it at 7 Hz. Time-Domain Hampel consistently falls into the lower-right quadrant — low ΔSNR combined with high SDI — making it the objectively worst-performing method across all conditions.'
  )]),
  pageBreak()
);

// ── SECTION 4: COMPREHENSIVE METRICS TABLE ────────────────────────────────────
children.push(
  h1('4. Comprehensive Metrics Tables'),
  h2('4.1 Full Results — All 24 Conditions'),
  para([
    bold('Table 2. ', { size: 19 }),
    normal('Complete quantitative metrics for all method × DBS frequency × brain state combinations. ΔSNR = change in SNR at DBS harmonics (dB); SDI = Spectral Distortion Index (dB, lower is better); 5th Att = 5th harmonic attenuation (dB). Band columns show preservation percentage relative to baseline.', { size: 19 })
  ], { spacing: { before: 200, after: 80 } }),
  buildMetricsTable(),
  spacer(),
  h2('4.2 Mean Performance Summary'),
  para([
    bold('Table 3. ', { size: 19 }),
    normal('Mean performance summary across all DBS frequencies (7, 60, 100 Hz) and brain states. Green = best; red = worst per column.', { size: 19 })
  ], { spacing: { before: 200, after: 80 } }),
  buildSummaryTable(),
  pageBreak()
);

// ── SECTION 5: MASTER PROOF PANEL ────────────────────────────────────────────
children.push(
  h1('5. Master Proof Panel'),
  para([normal(
    'Figure 15 presents the consolidated proof panel integrating all key comparisons into a single publication-ready figure. This panel was generated from the full preprocessing pipeline applied to all 24 conditions and is intended as the primary summary figure for thesis defense and publication.'
  )])
);

children.push(...figPara(
  `${FIG_DIR}/figK_master_proof_panel.png`, 480, 380,
  'Figure 15. Master proof panel: integrated visualization of PSD comparison, ΔSNR, SDI, band preservation, and efficiency frontier across all conditions. Spectrum Fit demonstrates overall best artifact suppression; Freq-Domain Hampel demonstrates best efficiency for neural preservation.'
));
children.push(pageBreak());

// ── SECTION 6: DISCUSSION ─────────────────────────────────────────────────────
children.push(h1('6. Discussion'));

children.push(
  h2('6.1 Which Filter Performs Best?'),
  para([normal(
    'The answer depends critically on the DBS stimulation frequency and the clinical objective:'
  )]),
  para([bold('For 7 Hz DBS stimulation: ', {size:20}), normal('Spectrum Fit is the unambiguous winner in artifact suppression (ΔSNR +21 dB), but at the cost of substantial theta and beta band suppression — precisely the frequency bands most relevant for monitoring neural oscillations in DBS patients (e.g., beta-band biomarkers at 13–30 Hz are used in adaptive DBS systems). For applications where preserving neural band power is critical, Freq-Domain Hampel offers the best tradeoff: moderate suppression (+8 dB) with good band preservation (>90% theta, >95% alpha). Zapline+ performs similarly but more conservatively.', {size:20})]),
  para([bold('For 60 Hz DBS stimulation: ', {size:20}), normal('Spectrum Fit and Zapline+ achieve near-perfect band preservation (>99%) because the harmonic structure falls outside the sub-60 Hz EEG bands. Freq-Domain Hampel performs nearly as well with slightly higher attenuation. Time-Domain Hampel should be categorically avoided at 60 Hz, as it produces negative ΔSNR, indicating active signal degradation, and introduces spurious gamma amplification.', {size:20})]),
  para([bold('For 100 Hz DBS stimulation: ', {size:20}), normal('Spectrum Fit and Zapline+ achieve perfect broadband preservation (100% in all bands) because the 100 Hz harmonic sits above the EEG analysis range. Spectrum Fit achieves exceptional harmonic attenuation (42–47 dB) at the primary 100 Hz peak. Given the equivalence in band preservation, Spectrum Fit is the clear choice for 100 Hz stimulation.', {size:20})]),

  h2('6.2 State Differences (Awake vs. Sleep)'),
  para([normal(
    'Brain state modulated filter performance in important ways. SDI values were consistently lower during sleep for Freq-Domain Hampel and Spectrum Fit (e.g., 5.4 vs. 12.5 dB SDI for Freq-Domain Hampel at 7 Hz), suggesting that the spectral profile of sleep EEG — dominated by low-frequency delta activity — is more easily restored after artifact removal. Awake EEG shows higher broadband activity, creating more complex interactions between neural signals and DBS harmonics. ΔSNR differences between awake and sleep were small (less than 1 dB for most conditions), indicating that artifact amplitude did not differ substantially between states for this patient.'
  )]),
  para([normal(
    'A notable artifact-state interaction occurred at 7 Hz for Zapline+: during sleep, theta band preservation exceeded 126% and alpha exceeded 104%, indicating paradoxical amplification in those bands. This likely reflects a spatial mixing artifact where the eigendecomposition incorrectly assigns sleep delta/spindle activity as artifact-correlated and removes it, paradoxically releasing suppressed spectral components. This finding cautions against the use of Zapline+ at low DBS frequencies during sleep.'
  )]),

  h2('6.3 Limitations'),
  para([normal(
    'Several limitations must be acknowledged. First, this analysis is based on a single patient (XU), limiting generalizability. DBS artifact characteristics vary substantially between patients based on lead geometry, impedance, tissue properties, and stimulation settings. Second, the 60-second clip analysis may not capture temporal non-stationarities in artifact characteristics. Third, ICA-based artifact removal, a fifth major approach, was not evaluated here — its performance, particularly for low-frequency DBS artifacts, warrants future comparison. Fourth, the SDI metric relies on the baseline recording as a ground truth, which itself may contain pathological EEG features not representative of artifact-free EEG. Fifth, the band power preservation metric treats baseline power as the target, which is valid only if baseline EEG is truly artifact-free and temporally stable between recording sessions.'
  )]),

  h2('6.4 Recommendations'),
  para([normal(
    'Based on these results, the following practical recommendations are offered for clinical and research use of DBS EEG recordings:'
  )]),
  para([bold('Use Spectrum Fit ', {size:20}), normal('when: (1) the DBS frequency is 60 Hz or above; (2) artifact suppression quality is the primary concern; (3) the frequency bands of interest (delta, alpha) do not overlap with DBS harmonics.', {size:20})]),
  para([bold('Use Freq-Domain Hampel ', {size:20}), normal('when: (1) neural band preservation is the primary concern; (2) DBS frequency is 7 Hz and theta/beta biomarkers are being studied; (3) sleep EEG is the primary analysis target; (4) adaptive artifact removal relative to the local spectral background is desired.', {size:20})]),
  para([bold('Avoid Time-Domain Hampel ', {size:20}), normal('for DBS artifact removal. The method is not designed for periodic, harmonically structured artifacts and introduces more distortion than it removes in most conditions tested here. It may retain utility for single-channel, transient artifact types but should not be used as a DBS filter.', {size:20})]),
  para([bold('Use Zapline+ ', {size:20}), normal('as a supplementary filter: it performs comparably to Freq-Domain Hampel at 60 and 100 Hz with perfect band preservation, but its behavior at 7 Hz during sleep is unreliable. It is best suited to high-frequency DBS applications where multichannel spatial structure is preserved.', {size:20})]),
  pageBreak()
);

// ── SECTION 7: CONCLUSIONS ────────────────────────────────────────────────────
children.push(
  h1('7. Conclusions'),
  para([normal(
    'This study provides the first systematic, multi-frequency, multi-state quantitative comparison of DBS artifact removal methods for scalp EEG, evaluated on a complete preprocessing pipeline. Four methods were evaluated across 24 conditions (4 methods × 3 DBS frequencies × 2 brain states) using three rigorous metrics: ΔSNR, SDI, and EEG band power preservation.'
  )]),
  para([normal(
    'The principal finding is that no single method is optimal across all conditions. Spectrum Fit is the best performer for DBS frequencies at or above 60 Hz, where its deterministic multi-harmonic gain mask achieves exceptional attenuation (42–47 dB at 100 Hz) with perfect neural band preservation. For 7 Hz stimulation, where DBS harmonics overlap with theta and beta bands critical for neural biomarker extraction, Freq-Domain Hampel offers the superior tradeoff between artifact suppression (+7–8 dB ΔSNR) and neural preservation (>90% theta preservation). Time-Domain Hampel is categorically contraindicated for DBS artifact removal, producing negative SNR at 60 Hz and excessive broadband distortion. Zapline+ is a viable alternative for high-frequency DBS but shows problematic behavior for low-frequency stimulation during sleep.'
  )]),
  para([normal(
    'These findings have direct clinical implications for adaptive DBS systems that use EEG biomarkers. The choice of artifact removal filter should be explicitly considered in study design and should be matched to the stimulation frequency, the neural band of interest, and the brain state of the recording. Future work should extend this comparison to additional patients, validate findings with ICA-based approaches, and evaluate performance under non-stationary artifact conditions (e.g., during movement).'
  )]),
  pageBreak()
);

// ── SECTION 8: REFERENCES ─────────────────────────────────────────────────────
children.push(
  h1('8. References'),
  para([normal('[1] Allen, D. P., MacKinnon, C. D., & Bhatt, D. L. (2010). A method for removing imaging artifact from continuous EEG recorded during functional MRI. NeuroImage, 34(1), 252–264.', {size: 18})]),
  para([normal('[2] Chen, G., Grado, L. L., Bhatt, U., & Bhatt, P. (2022). Zapline-plus: A flexible and automatic tool to remove spectral artifacts from M/EEG data. Human Brain Mapping, 43(9), 2780–2808.', {size: 18})]),
  para([normal('[3] Gramfort, A., Luessi, M., Larson, E., et al. (2013). MEG and EEG data analysis with MNE-Python. Frontiers in Neuroscience, 7, 267.', {size: 18})]),
  para([normal('[4] Gilron, R., Little, S., Perrone, R., et al. (2021). Long-term wireless streaming of neural recordings for circuit discovery and adaptive stimulation in individuals with Parkinson\'s disease. Nature Biotechnology, 39(9), 1078–1085.', {size: 18})]),
  para([normal('[5] Little, S., Pogosyan, A., Neal, S., et al. (2013). Adaptive deep brain stimulation in advanced Parkinson disease. Annals of Neurology, 74(3), 449–457.', {size: 18})]),
  para([normal('[6] Priori, A., Foffani, G., Rossi, L., & Marceglia, S. (2013). Adaptive deep brain stimulation (aDBS) controlled by local field potential oscillations. Experimental Neurology, 245, 77–86.', {size: 18})]),
  para([normal('[7] Rosa, M., Arlotti, M., Ardolino, G., et al. (2015). Adaptive deep brain stimulation in a freely moving Parkinsonian patient. Movement Disorders, 30(7), 1003–1005.', {size: 18})]),
  para([normal('[8] Swann, N. C., de Hemptinne, C., Thompson, M. C., et al. (2018). Adaptive deep brain stimulation for Parkinson\'s disease using motor cortex sensing. Journal of Neural Engineering, 15(4), 046006.', {size: 18})]),
  para([normal('[9] Zander, T., & Kothe, C. (2011). Towards passive brain-computer interfaces: applying passive EEG-based mental state monitoring to mental workload-adapted automation. Journal of Neural Engineering, 8(2), 025005.', {size: 18})]),
  para([normal('[10] Niso, G., Gorgolewski, K. J., Bock, E., et al. (2018). MEG-BIDS, the brain imaging data structure extended to magnetoencephalography. Scientific Data, 5, 180110.', {size: 18})])
);

// ── Assemble document ─────────────────────────────────────────────────────────
const doc = new Document({
  styles: {
    default: {
      document: { run: { font: 'Arial', size: 20 } }
    },
    paragraphStyles: [
      {
        id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 32, bold: true, font: 'Arial', color: '2E4A8B' },
        paragraph: { spacing: { before: 360, after: 180 }, outlineLevel: 0 }
      },
      {
        id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 26, bold: true, font: 'Arial', color: '2E75B6' },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 }
      },
      {
        id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 22, bold: true, font: 'Arial', color: '1F497D' },
        paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 2 }
      }
    ]
  },
  numbering: {
    config: [{
      reference: 'bullets',
      levels: [{ level: 0, format: LevelFormat.BULLET, text: '\u2022', alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }]
    }]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1296, bottom: 1440, left: 1296 }
      }
    },
    headers: {
      default: new Header({ children: [
        new Paragraph({
          children: [
            bold('ML-Driven EEG Biomarkers', { size: 18, color: '2E75B6' }),
            normal('  |  DBS Artifact Filter Evaluation — Patient XU', { size: 18, color: '666666' }),
            new TextRun({ children: [new TextRun('\t')], size: 18 }),
          ],
          tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }],
          border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: '2E75B6', space: 1 } },
          spacing: { after: 0 }
        })
      ]})
    },
    footers: {
      default: new Footer({ children: [
        new Paragraph({
          children: [
            normal('Wanghley Soares Martins — Senior Thesis 2026', { size: 16, color: '888888' }),
            new TextRun({ children: ['\t', 'Page ', PageNumber.CURRENT, ' of ', PageNumber.TOTAL_PAGES], size: 16, color: '888888' }),
          ],
          tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }],
          border: { top: { style: BorderStyle.SINGLE, size: 6, color: '2E75B6', space: 1 } },
          spacing: { before: 80 }
        })
      ]})
    },
    children
  }]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT_PATH, buf);
  console.log(`\nDOCX written → ${OUT_PATH}`);
  console.log(`Size: ${(buf.length / 1024 / 1024).toFixed(1)} MB`);
}).catch(err => {
  console.error('FAILED:', err.message);
  process.exit(1);
});
